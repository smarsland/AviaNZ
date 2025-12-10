# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# PyTorch model architectures

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import ASTModel

# AST normalization constants from AudioSet training
# These are used to normalize log-mel spectrograms for AST models
AST_MEAN = -4.2677393
AST_STD = 4.5689974


class AST(nn.Module):
    """Audio Spectrogram Transformer for spectrogram-based bird sound classification.
    
    Uses AST pretrained on AudioSet to process log-mel spectrograms.
    """
    
    def __init__(self, num_classes, multilabel=False, input_size=None, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.multilabel = multilabel
        
        # Load pretrained AST model from AudioSet
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Replace classifier head with dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        """Forward pass with AST paper normalization.
        
        Input x should be log-mel spectrogram (already log-transformed).
        Applies mean-std normalization with AudioSet statistics.
        """
        x = x.float()
        
        # Remove channel dimension if present (AST expects [B, H, W])
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        # Apply AST normalization with AudioSet statistics
        x = (x - AST_MEAN) / AST_STD
        
        # Pass through AST encoder
        hidden_states = self.ast(x).last_hidden_state
        
        # Use only cls token (first token) per AST paper standard practice
        features = hidden_states[:, 0]
        features = self.dropout(features)
        logits = self.classifier(features)
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


class CNNModel(nn.Module):
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
        x = self.softmax(x)
        
        return x
