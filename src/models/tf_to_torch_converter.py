
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

import torch
import torch.nn as nn
import h5py
import json
import numpy as np


class ChannelsLastFlatten(nn.Module):
    """
    Custom flatten layer that reorders dimensions to match TensorFlow's 
    channels-last flattening order.
    
    TensorFlow: (batch, height, width, channels) -> (batch, height * width * channels)
    PyTorch: (batch, channels, height, width) -> (batch, height * width * channels)
    
    This layer permutes (batch, C, H, W) -> (batch, H, W, C) before flattening
    to match TensorFlow's memory layout.
    """
    def forward(self, x):
        # Permute from (batch, channels, height, width) to (batch, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        # Flatten (batch, height, width, channels) to (batch, height * width * channels)
        return x.reshape(x.size(0), -1)


class TFtoPyTorchConverter:
    def __init__(self, json_path, h5_path=None):
        self.json_path = json_path
        self.h5_path = h5_path
        
        with open(json_path, 'r') as f:
            self.tf_config = json.load(f)
        
        self.layers_config = self.tf_config['config']['layers']
        
    def convert(self):
        pytorch_layers = []
        
        for i, layer_config in enumerate(self.layers_config):
            class_name = layer_config['class_name']
            config = layer_config['config']
            
            # Skip InputLayer - PyTorch doesn't need it
            if class_name == 'InputLayer':
                continue
                
            pytorch_layer = self._convert_layer(class_name, config)
            if pytorch_layer is not None:
                pytorch_layers.append(pytorch_layer)
        
        model = nn.Sequential(*pytorch_layers)
        
        # Load weights if HDF5 file provided
        if self.h5_path:
            self._load_weights(model)
        
        return model
    
    def _convert_layer(self, class_name, config):
        if class_name == 'Conv2D':
            return self._convert_conv2d(config)
        elif class_name == 'MaxPooling2D':
            return self._convert_maxpool2d(config)
        elif class_name == 'Dropout':
            return self._convert_dropout(config)
        elif class_name == 'Flatten':
            return self._convert_flatten(config)
        elif class_name == 'Dense':
            return self._convert_dense(config)
        else:
            raise NotImplementedError(f"Layer type {class_name} not yet supported")
    
    def _convert_conv2d(self, config):
        filters = config['filters']
        kernel_size = tuple(config['kernel_size'])
        strides = tuple(config['strides'])
        
        padding_mode = config['padding']
        if padding_mode == 'same':
            padding = tuple((k - 1) // 2 for k in kernel_size)
        elif padding_mode == 'valid':
            padding = (0, 0)
        else:
            raise ValueError(f"Unsupported padding mode: {padding_mode}")
        
        conv = nn.Conv2d(
            in_channels=1,  # Will be updated during weight loading
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            bias=config['use_bias']
        )
        
        activation = config.get('activation')
        if activation == 'relu':
            return nn.Sequential(conv, nn.ReLU())
        elif activation == 'linear' or activation is None:
            return conv
        else:
            raise NotImplementedError(f"Activation {activation} not yet supported for Conv2D")
    
    def _convert_maxpool2d(self, config):
        pool_size = tuple(config['pool_size'])
        strides = tuple(config['strides'])
        
        return nn.MaxPool2d(
            kernel_size=pool_size,
            stride=strides
        )
    
    def _convert_dropout(self, config):
        rate = config['rate']
        return nn.Dropout(p=rate)
    
    def _convert_flatten(self, config):
        """Convert TensorFlow Flatten to custom channels-last flatten."""
        return ChannelsLastFlatten()
    
    def _convert_dense(self, config):
        units = config['units']
        use_bias = config['use_bias']
        
        linear = nn.Linear(
            in_features=1,  # Will be updated during weight loading
            out_features=units,
            bias=use_bias
        )
        
        # Add activation if specified
        activation = config.get('activation')
        if activation == 'relu':
            return nn.Sequential(linear, nn.ReLU())
        elif activation == 'softmax':
            return nn.Sequential(linear, nn.Softmax(dim=1))
        elif activation == 'linear' or activation is None:
            return linear
        else:
            raise NotImplementedError(f"Activation {activation} not yet supported for Dense")
    
    def _load_weights(self, model):
        if not self.h5_path:
            return
        
        with h5py.File(self.h5_path, 'r') as h5file:
            # Get the top-level children (not all nested modules)
            modules_list = list(model.children())
            
            pytorch_module_idx = 0
            
            for layer_config in self.layers_config:
                class_name = layer_config['class_name']
                layer_name = layer_config['config']['name']
                
                # Skip InputLayer
                if class_name == 'InputLayer':
                    continue
                
                # Layers without weights - just advance the index
                if class_name in ['MaxPooling2D', 'Flatten', 'Dropout']:
                    pytorch_module_idx += 1
                    continue
                
                # Conv2D layers
                if class_name == 'Conv2D':
                    module = modules_list[pytorch_module_idx]
                    # Check if it's wrapped in Sequential (due to activation)
                    if isinstance(module, nn.Sequential):
                        conv_layer = module[0]
                    else:
                        conv_layer = module
                    
                    kernel_path = f"{layer_name}/{layer_name}/kernel:0"
                    if kernel_path in h5file:
                        kernel_data = h5file[kernel_path][:]
                        # TensorFlow: (H, W, in_channels, out_channels)
                        # PyTorch: (out_channels, in_channels, H, W)
                        kernel_torch = np.transpose(kernel_data, (3, 2, 0, 1))
                        
                        if conv_layer.in_channels != kernel_torch.shape[1]:
                            conv_layer.in_channels = kernel_torch.shape[1]
                        
                        conv_layer.weight.data = torch.from_numpy(kernel_torch).float()
                    
                    bias_path = f"{layer_name}/{layer_name}/bias:0"
                    if bias_path in h5file and conv_layer.bias is not None:
                        bias_data = h5file[bias_path][:]
                        conv_layer.bias.data = torch.from_numpy(bias_data).float()
                    
                    pytorch_module_idx += 1
                
                # Dense layers
                elif class_name == 'Dense':
                    module = modules_list[pytorch_module_idx]
                    # Check if it's wrapped in Sequential (due to activation)
                    if isinstance(module, nn.Sequential):
                        linear_layer = module[0]
                    else:
                        linear_layer = module
                    
                    kernel_path = f"{layer_name}/{layer_name}/kernel:0"
                    if kernel_path in h5file:
                        kernel_data = h5file[kernel_path][:]
                        # TensorFlow: (in_features, out_features)
                        # PyTorch: (out_features, in_features) - needs transpose
                        kernel_torch = kernel_data.T
                        
                        if linear_layer.in_features != kernel_torch.shape[1]:
                            linear_layer.in_features = kernel_torch.shape[1]
                        
                        linear_layer.weight.data = torch.from_numpy(kernel_torch).float()
                    
                    bias_path = f"{layer_name}/{layer_name}/bias:0"
                    if bias_path in h5file and linear_layer.bias is not None:
                        bias_data = h5file[bias_path][:]
                        linear_layer.bias.data = torch.from_numpy(bias_data).float()
                    
                    pytorch_module_idx += 1


def convert_tf_model_to_pytorch(json_path, h5_path=None):
    converter = TFtoPyTorchConverter(json_path, h5_path)
    return converter.convert()


def get_input_shape_from_config(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    layers = config['config']['layers']
    
    if layers and 'batch_input_shape' in layers[0]['config']:
        shape = layers[0]['config']['batch_input_shape']
        # TensorFlow: [batch, height, width, channels]
        # PyTorch: [batch, channels, height, width]
        if len(shape) == 4:
            return (shape[0], shape[3], shape[1], shape[2])
    
    return None
