import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel


class AttentionPooling(nn.Module):
    """Learned attention pooling over patch tokens.
    Computes attention scores over sequence tokens and returns weighted sum.
    """
    def __init__(self, embed_dim=768, hidden_dim=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, tokens):
        # tokens: (B, N, C)
        scores = self.attn(tokens).squeeze(-1)  # (B, N)
        weights = torch.softmax(scores, dim=1)   # (B, N)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)  # (B, C)
        return pooled


class SpectrogramDecoder(nn.Module):
    """Decoder to reconstruct spectrogram from patch embeddings."""
    def __init__(self, embed_dim=768, output_size=(128, 512)):
        super().__init__()
        self.output_size = output_size
        num_pixels = output_size[0] * output_size[1]
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_pixels)
        )
    
    def forward(self, patch_tokens):
        # patch_tokens: (B, N, C)
        # Average pool over all patches to get global feature
        global_feat = patch_tokens.mean(dim=1)  # (B, C)
        
        # Decode to full spectrogram
        recon_flat = self.decoder(global_feat)  # (B, num_pixels)
        recon = recon_flat.view(-1, self.output_size[0], self.output_size[1])  # (B, H, W)
        
        return recon

class MultiScaleCNNFrontend(nn.Module):
    
    def __init__(self, input_channels=1, embed_dim=768):
        super().__init__()
        
        self.fine_scale = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.mid_scale = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.coarse_scale = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
    
    def forward(self, x):
        fine = self.fine_scale(x)
        mid = self.mid_scale(x)
        coarse = self.coarse_scale(x)
        
        fine = F.interpolate(fine, size=mid.shape[2:], mode='bilinear', align_corners=False)
        coarse = F.interpolate(coarse, size=mid.shape[2:], mode='bilinear', align_corners=False)
        
        fused = torch.cat([fine, mid, coarse], dim=1)
        fused = self.fusion(fused)
        fused = self.pool(fused)
        
        B, C, H, W = fused.shape
        fused = fused.flatten(2).transpose(1, 2)
        
        return fused


class CNNModel(nn.Module):
    """CNN model for spectrogram classification."""
    
    def __init__(self, imageHeight, imageWidth, outputDim):
        super(CNNModel, self).__init__()
        
        apply_same_padding = imageHeight < 120 or imageWidth < 120
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding='same' if apply_same_padding else 'valid')
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding='same' if apply_same_padding else 'valid')
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding='same' if apply_same_padding else 'valid')
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding='same' if apply_same_padding else 'valid')
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout(0.2)
        
        self.flatten_size = self.get_flatten_size(imageHeight, imageWidth)
        
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, outputDim)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def get_flatten_size(self, height, width):
        x = torch.zeros(1, 1, height, width)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.pool2(x)
        
        x = self.conv4(x)
        x = self.pool3(x)
        
        x = self.conv5(x)
        x = self.pool4(x)
        
        return x.numel()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.relu(self.conv4(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.relu(self.conv5(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        # Return logits (no softmax) for use with CrossEntropyLoss
        
        return x


class MultiScaleAST(nn.Module):
    
    def __init__(self, num_classes, multilabel=False, input_size=None, dropout=0.1, use_reconstruction=False):
        super().__init__()
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.input_size = input_size if input_size else (128, 512)
        self.use_reconstruction = use_reconstruction
        
        self.multiscale_frontend = MultiScaleCNNFrontend(input_channels=1, embed_dim=768)
        
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPooling(embed_dim=768, hidden_dim=256)
        self.classifier = nn.Linear(768, num_classes)
        
        if use_reconstruction:
            self.decoder = SpectrogramDecoder(embed_dim=768, output_size=self.input_size)
    
    def forward(self, x):
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        import config
        x = (x - config.AST_MEAN) / config.AST_STD
        
        multiscale_features = self.multiscale_frontend(x)
        
        B, N, C = multiscale_features.shape
        cls_token = self.ast.embeddings.cls_token.expand(B, -1, -1)
        dist_token = self.ast.embeddings.distillation_token.expand(B, -1, -1)
        
        embeddings = torch.cat([cls_token, dist_token, multiscale_features], dim=1)
        
        pos_embed = self.ast.embeddings.position_embeddings
        if embeddings.shape[1] != pos_embed.shape[1]:
            n_special = 2
            special_pos = pos_embed[:, :n_special, :]
            patch_pos = pos_embed[:, n_special:, :]
            
            current_patches = N
            original_patches = patch_pos.shape[1]
            
            if current_patches != original_patches:
                embed_dim = patch_pos.shape[2]
                
                if original_patches == 1212:
                    h_old, w_old = 12, 101
                elif original_patches == 800:
                    h_old, w_old = 8, 100
                elif original_patches == 256:
                    h_old, w_old = 16, 16
                else:
                    h_old = w_old = int(original_patches ** 0.5)
                    if h_old * w_old != original_patches:
                        raise ValueError(f"Cannot determine grid size for {original_patches} patches")
                
                h_new = w_new = int(current_patches ** 0.5)
                if h_new * w_new != current_patches:
                    raise ValueError(f"Cannot determine grid size for {current_patches} patches")
                
                patch_pos = patch_pos.reshape(1, h_old, w_old, embed_dim).permute(0, 3, 1, 2)
                patch_pos = F.interpolate(patch_pos, size=(h_new, w_new), mode='bicubic', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h_new * w_new, embed_dim)
            
            pos_embed = torch.cat([special_pos, patch_pos], dim=1)
        
        embeddings = embeddings + pos_embed
        embeddings = self.ast.embeddings.dropout(embeddings)
        
        encoder_outputs = self.ast.encoder(embeddings)
        hidden_states = encoder_outputs.last_hidden_state
        # Exclude special tokens (cls + dist) for pooling
        patch_tokens = hidden_states[:, 2:, :]
        features = self.pool(patch_tokens)
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if self.use_reconstruction:
            recon = self.decoder(patch_tokens)
            return logits, recon
        return logits
    
    def interpolate_pos_embed(self, target_size):
        print(f"MultiScaleAST uses adaptive positional embeddings - no interpolation needed")
        pass


class AST(nn.Module):
    
    def __init__(self, num_classes, multilabel=False, input_size=None, dropout=0.1, use_reconstruction=False):
        super().__init__()
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.use_reconstruction = use_reconstruction
        self.input_size = input_size if input_size else (128, 512)
        
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPooling(embed_dim=768, hidden_dim=256)
        self.classifier = nn.Linear(768, num_classes)
        
        if use_reconstruction:
            self.decoder = SpectrogramDecoder(embed_dim=768, output_size=self.input_size)

    def forward(self, x, sparse_mode=False, positions=None, mask=None):
        """Forward pass with AST paper normalization.
        
        Input x should be log-mel spectrogram (already log-transformed).
        Applies mean-std normalization with AudioSet statistics.
        
        Args:
            x: Input tensor
               - Standard mode: (B, 1, H, W) spectrogram
               - Sparse mode: (B, K, 1, 16, 16) pre-extracted patches
            sparse_mode: If True, x contains pre-extracted patches
            positions: (B, K, 2) grid positions for sparse patches (row, col)
            mask: (B, K) boolean mask indicating valid patches (True) vs padding (False)
        """
        x = x.float()
        
        if sparse_mode:
            # Sparse patch mode: x is (B, K, 1, 16, 16)
            return self.forward_sparse(x, positions, mask)
        
        # Standard mode: process full spectrogram
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        import config
        x = (x - config.AST_MEAN) / config.AST_STD
        
        hidden_states = self.ast(x).last_hidden_state
        # Exclude special tokens (cls + dist) for pooling
        patch_tokens = hidden_states[:, 2:, :]
        features = self.pool(patch_tokens)
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if self.use_reconstruction:
            recon = self.decoder(patch_tokens)
            return logits, recon
        return logits
    
    def forward_sparse(self, patches, positions, mask):
        """
        Forward pass with sparse patches.
        
        Args:
            patches: (B, K, 1, 16, 16) - pre-extracted patches
            positions: (B, K, 2) - grid positions (row, col) for each patch
            mask: (B, K) - boolean mask for valid patches
        
        Returns:
            logits: (B, num_classes)
        """
        import config
        
        B, K, C, H, W = patches.shape
        assert C == 1, f"Expected 1 channel, got {C}"
        assert H == 16 and W == 16, f"Expected 16x16 patches, got {H}x{W}"
        
        # Flatten patches to (B*K, 1, 16, 16) then to (B*K, 256) for projection
        patches_flat = patches.contiguous().view(B * K, C * H * W)  # (B*K, 256)
        
        # Normalize the flattened patches
        # Note: AST normalization expects (H, W) format, but we have flattened patches
        # Apply normalization per-patch
        patches_reshaped = patches.view(B * K, H, W)  # (B*K, 16, 16)
        patches_normalized = (patches_reshaped - config.AST_MEAN) / config.AST_STD
        patches_flat = patches_normalized.view(B * K, H * W)  # (B*K, 256)
        
        # Project flattened patches to embedding dimension
        # AST uses 768-dimensional embeddings
        # We need to create a linear projection if it doesn't exist
        if not hasattr(self, 'patch_projection'):
            # Create a linear layer to project flattened patches to embedding dim
            embed_dim = 768
            self.patch_projection = nn.Linear(H * W, embed_dim).to(patches.device)
            # Initialize with small weights
            nn.init.xavier_uniform_(self.patch_projection.weight)
            nn.init.zeros_(self.patch_projection.bias)
        
        patch_embeddings = self.patch_projection(patches_flat)  # (B*K, 768)
        patch_embeddings = patch_embeddings.view(B, K, -1)  # (B, K, 768)
        
        # Add positional embeddings based on actual grid positions
        embed_dim = patch_embeddings.shape[-1]
        pos_embed = self.ast.embeddings.position_embeddings  # (1, N+2, 768) where N is original num patches
        
        # Extract position embeddings for spatial tokens (skip cls and dist tokens)
        spatial_pos_embed = pos_embed[:, 2:, :]  # (1, N, 768)
        
        # Infer grid size from the actual positional embedding tensor
        N = spatial_pos_embed.shape[1]
        
        # Common AST grids after interpolation
        if N == 1212:
            h_grid, w_grid = 12, 101
        elif N == 800:
            h_grid, w_grid = 8, 100
        elif N == 1050:
            h_grid, w_grid = 21, 50
        elif N == 512:
            h_grid, w_grid = 8, 64
        elif N == 448:
            h_grid, w_grid = 14, 32
        else:
            # Try to infer from input size
            # Calculate what the grid should be based on AST's patch extraction
            # AST uses 16x16 patches with stride (10, 10) typically
            # But for our interpolated embeddings, just factor N
            h_grid = int(N ** 0.5)
            w_grid = N // h_grid
            while h_grid * w_grid != N and h_grid > 1:
                h_grid -= 1
                if N % h_grid == 0:
                    w_grid = N // h_grid
                    break
            
            if h_grid * w_grid != N:
                raise ValueError(f"Cannot factorize {N} positional embeddings into a grid. N={N}")
        
        # Reshape to grid: (1, N, 768) -> (1, h, w, 768)
        spatial_pos_embed = spatial_pos_embed.view(1, h_grid, w_grid, embed_dim)
        
        # Index into positional embeddings using provided positions
        # positions: (B, K, 2) where [..., 0] is row, [..., 1] is col
        selected_pos_embed = []
        for b in range(B):
            batch_pos_embeds = []
            for k in range(K):
                if mask[b, k]:
                    row = positions[b, k, 0].item()
                    col = positions[b, k, 1].item()
                    # Clamp to valid range
                    row = min(row, h_grid - 1)
                    col = min(col, w_grid - 1)
                    pos_emb = spatial_pos_embed[0, row, col, :]
                else:
                    # Padding patch - use zero position embedding
                    pos_emb = torch.zeros(embed_dim, device=patches.device)
                batch_pos_embeds.append(pos_emb)
            selected_pos_embed.append(torch.stack(batch_pos_embeds))
        selected_pos_embed = torch.stack(selected_pos_embed)  # (B, K, 768)
        
        # Add positional embeddings
        patch_embeddings = patch_embeddings + selected_pos_embed
        
        # Add special tokens (cls and dist)
        cls_token = self.ast.embeddings.cls_token.expand(B, -1, -1)
        dist_token = self.ast.embeddings.distillation_token.expand(B, -1, -1)
        
        # Concatenate: [cls, dist, patch1, patch2, ..., patchK]
        embeddings = torch.cat([cls_token, dist_token, patch_embeddings], dim=1)  # (B, K+2, 768)
        
        # Add special token positional embeddings
        special_pos_embed = pos_embed[:, :2, :]  # (1, 2, 768)
        embeddings[:, :2, :] = embeddings[:, :2, :] + special_pos_embed
        
        # Apply dropout
        embeddings = self.ast.embeddings.dropout(embeddings)
        
        # Pass through transformer encoder - just give it the embeddings
        encoder_outputs = self.ast.encoder(embeddings)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Pool over patch tokens (excluding special tokens)
        patch_tokens = hidden_states[:, 2:, :]
        
        # Mask out padding before pooling
        masked_patch_tokens = patch_tokens.clone()
        masked_patch_tokens[~mask] = 0
        
        # Use attention pooling
        features = self.pool(masked_patch_tokens)
        features = self.dropout(features)
        logits = self.classifier(features)
        
        if self.use_reconstruction:
            recon = self.decoder(patch_tokens)
            return logits, recon
        return logits
    
    def interpolate_pos_embed(self, target_size):
        """Interpolate positional embeddings to match target input size.
        
        Follows ViT/AST paper methodology: bicubic interpolation over both spatial dimensions.
        
        Args:
            target_size: Tuple of (height, width) for target spectrogram size
        """
        pos_embed = self.ast.embeddings.position_embeddings
        device = pos_embed.device
        dtype = pos_embed.dtype
        B, N, C = pos_embed.shape
        
        # Detect number of special tokens (cls, or cls+dist)
        # AST typically uses 2 special tokens (cls + dist) from DeiT backbone
        n_special = 2
        if N - n_special <= 0:
            n_special = 1
        
        num_old_patches = N - n_special
        
        # Infer original grid dimensions from known AST configurations
        # Standard AST: 128 mel bins, 1024 time steps -> 12x101 patches (with 16x16 patches, stride 10)
        # We need to infer h_old, w_old from num_old_patches
        # Common configurations: 12x101=1212 (standard), 8x100=800, etc.
        h_old, w_old = None, None
        
        # Try to get from model config if available
        if hasattr(self.ast.config, 'num_mel_bins') and hasattr(self.ast.config, 'max_length'):
            projection = self.ast.embeddings.patch_embeddings.projection
            patch_size = projection.kernel_size
            stride = projection.stride
            h_old = (self.ast.config.num_mel_bins - patch_size[0]) // stride[0] + 1
            w_old = (self.ast.config.max_length - patch_size[1]) // stride[1] + 1
        else:
            # Fallback: infer from common AST grid sizes
            if num_old_patches == 1212:  # 12 x 101
                h_old, w_old = 12, 101
            elif num_old_patches == 800:  # 8 x 100
                h_old, w_old = 8, 100
            elif num_old_patches == 980:  # 10 x 98
                h_old, w_old = 10, 98
            else:
                # Try to factor assuming roughly 12:100 aspect ratio (height:width for audio)
                # This is approximate for mel bins (freq) vs time
                for h in range(1, int(num_old_patches**0.5) + 1):
                    if num_old_patches % h == 0:
                        w = num_old_patches // h
                        if 8 <= h <= 16 and 80 <= w <= 120:  # reasonable audio grid
                            h_old, w_old = h, w
                            break
        
        if h_old is None or w_old is None:
            raise ValueError(f"Cannot infer original patch grid from {num_old_patches} patches")
        
        # Calculate new grid dimensions
        projection = self.ast.embeddings.patch_embeddings.projection
        patch_size = projection.kernel_size
        stride = projection.stride
        
        h_new = (target_size[0] - patch_size[0]) // stride[0] + 1
        w_new = (target_size[1] - patch_size[1]) // stride[1] + 1
        
        if h_old == h_new and w_old == w_new:
            print(f"Position embeddings already match target size: {h_old}x{w_old}")
            return
        
        print(f"Interpolating position embeddings from {h_old}x{w_old} to {h_new}x{w_new}")
        
        # Split special tokens and position tokens
        special_tokens = pos_embed[:, :n_special, :]  # cls (and possibly dist)
        pos_tokens = pos_embed[:, n_special:, :]  # spatial position embeddings
        
        # Reshape to 2D grid: (1, num_patches, C) -> (1, C, h_old, w_old)
        pos_tokens = pos_tokens.reshape(1, h_old, w_old, C).permute(0, 3, 1, 2)
        
        # Interpolate only time dimension if height unchanged
        if h_old == h_new and w_old != w_new:
            pos_tokens = F.interpolate(
                pos_tokens,
                size=(h_new, w_new),
                mode='bicubic',
                align_corners=False
            )
        elif h_old != h_new or w_old != w_new:
            pos_tokens = F.interpolate(
                pos_tokens,
                size=(h_new, w_new),
                mode='bicubic',
                align_corners=False
            )
        
        # Reshape back to sequence: (1, C, h_new, w_new) -> (1, num_new_patches, C)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, h_new * w_new, C)
        
        # Concatenate special tokens and interpolated position tokens
        new_pos_embed = torch.cat([special_tokens, pos_tokens], dim=1)
        
        # Validate shape
        expected_length = n_special + h_new * w_new
        assert new_pos_embed.shape[1] == expected_length, \
            f"Shape mismatch: got {new_pos_embed.shape[1]}, expected {expected_length}"
        
        # Ensure correct device and dtype
        new_pos_embed = new_pos_embed.to(device=device, dtype=dtype)
        
        # Update the model's position embeddings
        self.ast.embeddings.position_embeddings = nn.Parameter(new_pos_embed)
        print(f"Position embeddings updated: {pos_embed.shape} -> {new_pos_embed.shape}")


class PixelPredictionCNN(nn.Module):
    
    def __init__(self, imageHeight, imageWidth):
        super(PixelPredictionCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output


class ASTPixelPredictor(nn.Module):
    
    def __init__(self, input_size=(224, 512)):
        super().__init__()
        self.input_size = input_size
        
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        projection = self.ast.embeddings.patch_embeddings.projection
        patch_size = projection.kernel_size
        stride = projection.stride
        
        h_patches = (input_size[0] - patch_size[0]) // stride[0] + 1
        w_patches = (input_size[1] - patch_size[1]) // stride[1] + 1
        self.h_patches = h_patches
        self.w_patches = w_patches
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        x = x.float()
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        import config
        x = (x - config.AST_MEAN) / config.AST_STD
        
        hidden_states = self.ast(x).last_hidden_state
        
        patch_tokens = hidden_states[:, 2:, :]
        
        B = patch_tokens.shape[0]
        patch_features = patch_tokens.reshape(B, self.h_patches, self.w_patches, 768)
        patch_features = patch_features.permute(0, 3, 1, 2)
        
        output = self.decoder(patch_features)
        
        output = F.interpolate(output, size=self.input_size, mode='bilinear', align_corners=False)
        
        return output