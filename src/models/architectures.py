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
from transformers import ViTModel, ViTConfig


class AST(nn.Module):
    """Vision Transformer for spectrogram-based bird sound classification.
    
    Uses a ViT pretrained on ImageNet to process spectrograms as images.
    """
    
    def __init__(self, num_classes, multilabel=False, input_size=None, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.input_size = input_size  # Store expected input size (H, W) - for reference only
        
        # Load pretrained ViT model (ImageNet-21k)
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # Update config to accept 1 channel
        self.vit.config.num_channels = 1
        
        # Adapt patch embedding to 1 input channel by averaging RGB weights
        proj = self.vit.embeddings.patch_embeddings.projection
        if isinstance(proj, nn.Conv2d) and proj.in_channels == 3:
            new_proj = nn.Conv2d(
                1,
                proj.out_channels,
                kernel_size=proj.kernel_size,
                stride=proj.stride,
                padding=proj.padding,
                bias=proj.bias is not None,
            )
            with torch.no_grad():
                w = proj.weight.mean(dim=1, keepdim=True)
                new_proj.weight.copy_(w)
                if proj.bias is not None:
                    new_proj.bias.copy_(proj.bias)
            self.vit.embeddings.patch_embeddings.projection = new_proj
            
            # Also update the patch_embeddings attributes
            self.vit.embeddings.patch_embeddings.num_channels = 1
        
        # Replace classifier head with dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        x = x.float()

        # 1) Per-sample min-max normalization to [0,1]
        # Hardcoded normalization settings for ViT input
        # Compute min/max over spatial dims (and channels)
        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
        denom = (x_max - x_min).clamp_min(1e-6)  # Numerical stability
        x = (x - x_min) / denom
        x = x.clamp(0.0, 1.0)

        # 2) Apply ViT input normalization (ImageNet convention): (x - mean) / std
        # For ImageNet ViT checkpoints, mean=std=0.5 maps [0,1] -> [-1,1]
        mean = torch.tensor([0.5], device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor([0.5], device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        x = (x - mean) / std

        # Pass through ViT with interpolate_pos_encoding to handle variable sizes
        outputs = self.vit(pixel_values=x, interpolate_pos_encoding=True)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return logits


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
