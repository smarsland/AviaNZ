import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel
import timm
from src.core import config


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for Domain Adversarial Neural Networks (DANN).
    
    Reverses gradients during backprop to make features domain-invariant.
    """
    def __init__(self):
        super().__init__()
        self.lambda_param = 1.0
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_param)
    
    def set_lambda(self, lambda_param):
        self.lambda_param = lambda_param


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_param * grad_output, None


class DomainDiscriminator(nn.Module):
    """Domain discriminator for DANN - tries to classify source vs target domain."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.discriminator(x)


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


class TemporalAttentionHead(nn.Module):
    """Per-class temporal attention head for SED-style classification.

    Collapses the frequency axis of a 2D feature map, then computes independent
    attention weights over the time axis for each output class.  This lets each
    class focus on the time windows where its evidence actually lives, rather than
    averaging over the whole clip (which mixes signal with background noise).

    Forward input:  (B, C, F', T')   — spatial backbone output, NOT globally pooled
    Forward output: (B, num_classes) — logits
    """
    def __init__(self, in_channels, num_classes, proj_dim=None):
        super().__init__()
        if proj_dim is None:
            proj_dim = in_channels
        self.proj = nn.Conv1d(in_channels, proj_dim, kernel_size=1)
        self.attn_conv = nn.Conv1d(proj_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(proj_dim, num_classes, kernel_size=1)
        nn.init.zeros_(self.cls_conv.bias)
        nn.init.zeros_(self.attn_conv.bias)

    def forward(self, x):
        # x: (B, C, F', T')
        x = x.mean(dim=2)                                  # collapse freq → (B, C, T')
        x = F.relu(self.proj(x))                           # (B, proj_dim, T')
        attn = torch.softmax(self.attn_conv(x), dim=-1)    # (B, K, T')
        cls = self.cls_conv(x)                              # (B, K, T')
        logits = (attn * cls).sum(dim=-1)                  # (B, K)
        return logits


class CnnAdapter(nn.Module):
    """Lightweight trainable CNN adapter prepended to a frozen backbone.

    Learns a residual correction: output = input + f(input).
    The final 1x1 conv is zero-initialized so the adapter starts as identity.
    """
    def __init__(self, in_chans=1, num_layers=2, hidden_channels=32):
        super().__init__()
        self.in_chans = in_chans
        layers = []
        in_ch = in_chans
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            ]
            in_ch = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, in_chans, kernel_size=1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(1)
        out = x + self.net(x)
        if squeeze:
            out = out.squeeze(1)
        return out


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


class AST(nn.Module):
    """Audio Spectrogram Transformer. Always uses multilabel classification."""
    
    def __init__(self, num_classes, input_size=None, dropout=0.1, use_reconstruction=False,
                 use_adapters=False, adapter_dim=64, per_chunk_norm=False, num_chunks=2,
                 use_cnn_adapter=False, use_sed_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_reconstruction = use_reconstruction
        self.use_adapters = use_adapters
        self.per_chunk_norm = per_chunk_norm
        self.num_chunks = num_chunks
        self.input_size = input_size if input_size else (128, 512)
        self.use_sed_head = use_sed_head

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
        # pool + classifier always present (used by get_features / DANN)
        self.pool = AttentionPooling(embed_dim=768, hidden_dim=256)
        self.classifier = nn.Linear(768, num_classes)

        if use_sed_head:
            # Per-class attention over tokens: each class gets its own
            # attention distribution over the N patch tokens.
            self.attn_proj = nn.Linear(768, num_classes)
            self.val_proj = nn.Linear(768, num_classes)
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.val_proj.bias)

        if use_reconstruction:
            self.decoder = SpectrogramDecoder(embed_dim=768, output_size=self.input_size)
        self.cnn_adapter = CnnAdapter() if use_cnn_adapter else None

    def forward(self, x):
        x = x.float()

        if self.cnn_adapter is not None:
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = self.cnn_adapter(x)

        # Standard mode: process full spectrogram
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        
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

        if self.use_sed_head:
            # Per-class attention: each class attends independently over N tokens
            attn = torch.softmax(self.attn_proj(patch_tokens), dim=1)   # (B, N, K)
            vals = self.val_proj(patch_tokens)                            # (B, N, K)
            logits = (attn * vals).sum(dim=1)                            # (B, K)
        else:
            features = self.pool(patch_tokens)
            features = self.dropout(features)
            logits = self.classifier(features)

        if self.use_reconstruction:
            recon = self.decoder(patch_tokens)
            return logits, recon
        return logits
    
    def get_features(self, x):
        """Extract feature representation before classification layer.
        
        Returns pooled 768-dim features for DANN domain discrimination.
        """
        x = x.float()
        
        if self.cnn_adapter is not None:
            if x.dim() == 3:
                x = x.unsqueeze(1)
            x = self.cnn_adapter(x)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

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
        
        patch_tokens = hidden_states[:, 2:, :]
        features = self.pool(patch_tokens)
        
        return features
    
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


class GatedSpeciesHead(nn.Module):
    """Two-stage bird-presence gate + species classifier head.

    Takes pooled backbone features and produces:
      - gate_logit:     [B]           binary logit — is any bird present?
      - species_logits: [B, num_classes]  per-class multilabel logits

    The gate and species branches share the same backbone representation but
    are trained with separate objectives so they can specialise independently.
    Gate bias is zero-initialised so it starts at p=0.5 (no prior).
    """

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.gate = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.zeros_(self.gate.bias)

    def forward(self, features):
        gate_logit = self.gate(features).squeeze(-1)   # [B]
        species_logits = self.classifier(features)      # [B, num_classes]
        return species_logits, gate_logit


class RegNetModel(nn.Module):
    """RegNetY model for bird audio classification (BirdClef fine-tuning)."""

    def __init__(self, num_classes, pretrained_path=None, model_name='regnety_008', freeze_backbone=False, freeze_stages=0, use_cnn_adapter=False, use_sed_head=False, use_gated_head=False, in_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.use_sed_head = use_sed_head
        self.use_gated_head = use_gated_head
        self.in_chans = in_chans

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_chans,
            drop_rate=0.0,
            drop_path_rate=0.0
        )

        if 'efficientnet' in model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'regnet' in model_name:
            backbone_out = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.feature_dim = backbone_out

        if use_gated_head:
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.gated_head = GatedSpeciesHead(backbone_out, num_classes)
        elif use_sed_head:
            self.sed_head = TemporalAttentionHead(backbone_out, num_classes)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(backbone_out, num_classes)

        self.cnn_adapter = CnnAdapter(in_chans=in_chans) if use_cnn_adapter else None

        if pretrained_path:
            self._load_pretrained_weights(pretrained_path, freeze_backbone, freeze_stages)
    
    def _load_pretrained_weights(self, pretrained_path, freeze_backbone, freeze_stages):
        print(f"Loading BirdClef pretrained weights from {pretrained_path}")
        
        import sys
        
        # Create dummy CFG class for unpickling BirdClef checkpoint
        class CFG:
            pass
        
        # Add to __main__ module so unpickler can find it
        sys.modules['__main__'].CFG = CFG
        
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        orig_num_classes = checkpoint['model_state_dict']['classifier.weight'].shape[0]
        print(f"  Original model: {orig_num_classes} classes (BirdClef)")
        print(f"  Target model: {self.num_classes} classes (your dataset)")
        
        backbone_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                if 'classifier' not in new_key and 'fc' not in new_key:
                    backbone_dict[new_key] = v
        
        # Drop any keys whose shape doesn't match the current model (e.g. stem
        # conv when in_chans != 1); strict=False skips missing/extra keys but
        # still raises RuntimeError on size mismatches.
        current_shapes = {k: v.shape for k, v in self.backbone.state_dict().items()}
        backbone_dict = {
            k: v for k, v in backbone_dict.items()
            if k not in current_shapes or v.shape == current_shapes[k]
        }

        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_dict, strict=False)
        missing_keys = [k for k in missing_keys if 'classifier' not in k and 'fc' not in k]
        unexpected_keys = [k for k in unexpected_keys if 'classifier' not in k and 'fc' not in k]
        
        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")
        
        print("  ✓ Loaded pretrained backbone weights successfully")
        
        if freeze_backbone:
            print("  Freezing entire backbone - only training classifier head")
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_stages > 0:
            print(f"  Freezing first {freeze_stages} stages of backbone")
            stage_names = ['stem', 's1', 's2', 's3', 's4']
            for stage_name in stage_names[:freeze_stages]:
                if hasattr(self.backbone, stage_name):
                    stage = getattr(self.backbone, stage_name)
                    for param in stage.parameters():
                        param.requires_grad = False
                    print(f"    Froze stage: {stage_name}")
        else:
            print("  All layers trainable (full fine-tuning)")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def forward(self, x):
        if self.cnn_adapter is not None:
            x = self.cnn_adapter(x)
        if self.use_sed_head:
            features = self.backbone.forward_features(x)   # (B, C, F', T')
            logits = self.sed_head(features)
            return logits

        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        if self.use_gated_head:
            species_logits, gate_logit = self.gated_head(features)
            if self.training:
                return species_logits, gate_logit
            # Eval / inference: soft gate — add log P(bird present) to each species logit.
            # Equivalent to P(species_k) = P(bird present) * P(species_k | bird present).
            # When gate is confident of silence, species probs approach 0.
            # When uncertain, they are scaled down but not hard-zeroed.
            gate_log_prob = F.logsigmoid(gate_logit).unsqueeze(1)  # [B, 1], always <= 0
            return species_logits + gate_log_prob

        logits = self.classifier(features)
        return logits