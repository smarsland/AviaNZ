import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel
import timm


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
        
        # Initialize weights with proper scaling to prevent gradient explosion
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
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
    
    def forward(self, x, sparse_mode=False, positions=None, mask=None):
        if sparse_mode:
            raise NotImplementedError("MultiScaleAST does not support sparse patch mode. Use standard AST with use_multiscale=False.")
        
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
    
    def __init__(self, num_classes, multilabel=False, input_size=None, dropout=0.1, use_reconstruction=False,
                 use_adapters=False, adapter_dim=64, per_chunk_norm=False, num_chunks=2):
        super().__init__()
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.use_reconstruction = use_reconstruction
        self.use_adapters = use_adapters
        self.per_chunk_norm = per_chunk_norm
        self.num_chunks = num_chunks
        self.input_size = input_size if input_size else (128, 512)
        
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        if use_adapters:
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(768, adapter_dim),
                    nn.ReLU(),
                    nn.Linear(adapter_dim, 768)
                ) for _ in range(12)
            ])
            for adapter in self.adapters:
                nn.init.zeros_(adapter[2].weight)
                nn.init.zeros_(adapter[2].bias)
        
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPooling(embed_dim=768, hidden_dim=256)
        self.classifier = nn.Linear(768, num_classes)
        
        self.patch_projection = nn.Linear(256, 768)
        
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
        
        if self.per_chunk_norm:
            B, H, W = x.shape
            chunk_width = W // self.num_chunks
            chunks = []
            for i in range(self.num_chunks):
                start = i * chunk_width
                end = start + chunk_width if i < self.num_chunks - 1 else W
                chunk = x[:, :, start:end]
                chunk_min = chunk.reshape(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(2)
                chunk_max = chunk.reshape(B, -1).max(dim=1, keepdim=True)[0].unsqueeze(2)
                chunk_normalized = (chunk - chunk_min) / (chunk_max - chunk_min + 1e-6)
                chunks.append(chunk_normalized)
            x = torch.cat(chunks, dim=2)
        else:
            x = (x - config.AST_MEAN) / config.AST_STD
        
        hidden_states = self.ast(x).last_hidden_state
        
        if self.use_adapters:
            for i in range(12):
                adapter_output = self.adapters[i](hidden_states)
                hidden_states = hidden_states + adapter_output
        
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
        
        # Project flattened patches to embedding dimension using the trained projection layer
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
        
        # Calculate new grid dimensions
        projection = self.ast.embeddings.patch_embeddings.projection
        patch_size = projection.kernel_size
        stride = projection.stride
        
        h_new = (target_size[0] - patch_size[0]) // stride[0] + 1
        w_new = (target_size[1] - patch_size[1]) // stride[1] + 1
        
        # Infer original grid dimensions from checkpoint
        # Try to get from model config if available
        h_old, w_old = None, None
        
        if hasattr(self.ast.config, 'num_mel_bins') and hasattr(self.ast.config, 'max_length'):
            h_old = (self.ast.config.num_mel_bins - patch_size[0]) // stride[0] + 1
            w_old = (self.ast.config.max_length - patch_size[1]) // stride[1] + 1
        else:
            # Fallback: infer from common AST grid sizes and number of patches
            if num_old_patches == 1212:  # 12 x 101 - standard
                h_old, w_old = 12, 101
            elif num_old_patches == 2122:  # 46 x 46 or similar for larger models
                for h in range(1, int(num_old_patches**0.5) + 2):
                    if num_old_patches % h == 0:
                        w = num_old_patches // h
                        # Check if this is a reasonable aspect ratio for audio spectrograms
                        h_old, w_old = h, w
                        break
            else:
                # Try to factor: prefer height in range [8-16] (freq bins), width in [80-150] (time)
                found = False
                for h in range(16, 7, -1):  # Try from 16 down to 8
                    if num_old_patches % h == 0:
                        w = num_old_patches // h
                        if 50 <= w <= 200:  # reasonable time dimension
                            h_old, w_old = h, w
                            found = True
                            break
                
                if not found:
                    # Last resort: simple factorization
                    for h in range(1, int(num_old_patches**0.5) + 1):
                        if num_old_patches % h == 0:
                            w = num_old_patches // h
                            h_old, w_old = h, w
                            break
        
        if h_old is None or w_old is None:
            raise ValueError(f"Cannot infer original patch grid from {num_old_patches} patches. "
                           f"Position embeddings shape: {pos_embed.shape}. "
                           f"Try specifying freq_bins and time_bins in config.")
        
        if h_old == h_new and w_old == w_new:
            print(f"Position embeddings already match target size: {h_old}x{w_old}")
            return
        
        print(f"Interpolating position embeddings from {h_old}x{w_old} to {h_new}x{w_new}")
        
        # Split special tokens and position tokens
        special_tokens = pos_embed[:, :n_special, :]  # cls (and possibly dist)
        pos_tokens = pos_embed[:, n_special:, :]  # spatial position embeddings
        
        # Reshape to 2D grid: (1, num_patches, C) -> (1, C, h_old, w_old)
        pos_tokens = pos_tokens.reshape(1, h_old, w_old, C).permute(0, 3, 1, 2)
        
        # Interpolate
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


class KaytooClassifierHead(nn.Module):
    """Kaytoo's per-timestep classifier head."""
    def __init__(self, in_channels, num_classes, dropout_rate=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)
        return x


class KaytooAttentionBlock(nn.Module):
    """Kaytoo's attention-weighted aggregation over time chunks."""
    def __init__(self, in_features, out_features, image_shape=(1,1)):
        super().__init__()
        
        self.attention = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        with torch.no_grad():
            self.attention.weight.fill_(1.0 / (self.attention.kernel_size[0] * in_features))
            self.attention.bias.zero_()
        
        self.classify = KaytooClassifierHead(in_channels=in_features, num_classes=out_features)
        self.image_shape = image_shape
        self.num_chunks = int(self.image_shape[0] * self.image_shape[1])

    def forward(self, x):
        batch_size = x.shape[0]
        split_length = x.shape[2] // self.num_chunks
        
        x = torch.split(x, split_length, dim=2)
        x = torch.cat(x, dim=0)
        
        attn = self.attention(x)
        attn = torch.clamp(attn, min=-50.0, max=50.0)  # Prevent overflow in softmax
        norm_att = torch.softmax(torch.tanh(attn), dim=-1) / self.num_chunks
        split_attn = torch.split(norm_att, batch_size, dim=0)
        norm_att = torch.cat(split_attn, dim=2)

        seg_logits = self.classify(x)
        seg_logits = torch.clamp(seg_logits, min=-50.0, max=50.0)  # Prevent overflow in sigmoid
        seg_logits = F.dropout(seg_logits, p=0.3, training=self.training)
        classify = torch.sigmoid(seg_logits)

        split_logits = torch.split(seg_logits, batch_size, dim=0)
        seg_logits = torch.cat(split_logits, dim=2)

        split_classify = torch.split(classify, batch_size, dim=0)
        classify = torch.cat(split_classify, dim=2)
        
        weighted_preds = norm_att * classify
        weighted_seg_logits = norm_att * seg_logits
        preds = weighted_preds.sum(dim=-1)
        logit = weighted_seg_logits.sum(dim=-1)
        seg_logits = seg_logits.transpose(1, 2)

        return logit, seg_logits, preds


class KaytooModel(nn.Module):
    """Kaytoo architecture: timm EfficientNet backbone + attention pooling.
    
    This is Olly Powell's bird classifier architecture, using:
    - timm EfficientNet backbone (ImageNet pretrained)
    - Per-chunk min-max normalization
    - ImageNet normalization (mean/std)
    - Custom attention pooling over time chunks
    - Sigmoid multilabel output
    """
    def __init__(self, num_classes, multilabel=True, input_size=(128, 1024), 
                 backbone_name='tf_efficientnet_b2.ns_jft_in1k', image_shape=(2,1),
                 dropout=0.2):
        super().__init__()
        
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.input_size = input_size
        self.image_shape = image_shape
        
        # ImageNet normalization constants (for EfficientNet backbone)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.bn0 = nn.BatchNorm2d(3)
        
        self.base_model = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=3
        )
        
        layers = list(self.base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        
        try:
            classifier = self.base_model.get_classifier()
            if isinstance(classifier, nn.Identity):
                in_features = self.base_model.num_features
            else:
                in_features = classifier.in_features
        except AttributeError:
            if hasattr(self.base_model, "fc") and hasattr(self.base_model.fc, "in_features"):
                in_features = self.base_model.fc.in_features
            elif hasattr(self.base_model, "head") and hasattr(self.base_model.head, "fc"):
                in_features = self.base_model.head.fc.in_features
            elif hasattr(self.base_model, "classifier") and hasattr(self.base_model.classifier, "in_features"):
                in_features = self.base_model.classifier.in_features
            else:
                in_features = 1408
        
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = KaytooAttentionBlock(in_features, self.num_classes, image_shape=self.image_shape)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            self.fc1.bias.data.fill_(0.)
        self.bn0.bias.data.fill_(0.)
        self.bn0.weight.data.fill_(1.0)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W) or (B,1,H,W) spectrograms, got {x.shape}")

        # Apply per-chunk min-max normalization (Kaytoo's key preprocessing)
        B, C, H, W = x.shape
        num_chunks = int(self.image_shape[0] * self.image_shape[1])
        chunk_width = W // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_width
            end = start + chunk_width if i < num_chunks - 1 else W
            chunk = x[:, :, :, start:end]
            chunk_min = chunk.reshape(B, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(3)
            chunk_max = chunk.reshape(B, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(3)
            chunk_normalized = (chunk - chunk_min) / (chunk_max - chunk_min + 1e-6)
            chunks.append(chunk_normalized)
        x = torch.cat(chunks, dim=3)
        
        # Apply ImageNet normalization (for EfficientNet backbone)
        x = (x - self.imagenet_mean) / self.imagenet_std
        
        x = self.bn0(x)
        x = self.encoder(x)
        
        # Clamp encoder output to prevent NaN propagation under AMP
        x = torch.clamp(x, min=-50.0, max=50.0)
        
        if self.image_shape == (2,2):
            half = x.shape[2] // 2
            x0 = x[:,:,:half,:half]
            x1 = x[:,:,:half,half:]
            x2 = x[:,:,half:,:half]
            x3 = x[:,:,half:,half:]
            x = torch.cat((x0,x1,x2,x3), dim=2)
        elif self.image_shape == (1,4):
            quarter = x.shape[3] // 4
            x0 = x[:,:,:,:quarter]
            x1 = x[:,:,:,quarter:2*quarter]
            x2 = x[:,:,:,2*quarter:3*quarter]
            x3 = x[:,:,:,3*quarter:]
            x = torch.cat((x0,x1,x2,x3), dim=2)
        elif self.image_shape == (1,2):
            half = x.shape[3] // 2
            x0 = x[:,:,:,:half]
            x1 = x[:,:,:,half:]
            x = torch.cat((x0,x1), dim=2)
        elif self.image_shape == (2, 0.5):
            half = x.shape[2] // 2
            x0 = x[:,:,:half,:]
            x1 = x[:,:,half:,:]
            x = torch.cat((x0,x1), dim=3)

        dimension = 2 if self.image_shape == (2, 0.5) else 3
        x = torch.mean(x, dim=dimension)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)

        logit, segment_logits, preds = self.att_block(x)
        
        if self.multilabel:
            return logit
        else:
            return logit


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_param = 1.0
    
    def set_lambda(self, lambda_param):
        self.lambda_param = lambda_param
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_param)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class DANNModel(nn.Module):
    def __init__(self, num_classes, backbone_name='regnety_008', pretrained=True, freeze_backbone=False):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        base_model = timm.create_model(backbone_name, pretrained=pretrained)
        
        if hasattr(base_model, 'fc'):
            feature_dim = base_model.fc.in_features
            backbone_layers = list(base_model.children())[:-1]
        elif hasattr(base_model, 'head'):
            if hasattr(base_model.head, 'fc'):
                feature_dim = base_model.head.fc.in_features
            else:
                feature_dim = base_model.num_features
            backbone_layers = list(base_model.children())[:-1]
        elif hasattr(base_model, 'classifier'):
            feature_dim = base_model.classifier.in_features
            backbone_layers = list(base_model.children())[:-1]
        else:
            feature_dim = base_model.num_features
            backbone_layers = list(base_model.children())
        
        self.feature_extractor = nn.Sequential(*backbone_layers)
        self.feature_dim = feature_dim
        
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.class_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.grl = GradientReversalLayer()
        self.domain_classifier = DomainDiscriminator(input_dim=feature_dim)
    
    def forward(self, x, alpha=1.0):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        features = self.feature_extractor(x)
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        
        class_output = self.class_classifier(features)
        
        self.grl.set_lambda(alpha)
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)
        
        return class_output, domain_output, features