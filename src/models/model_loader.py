
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

# Trained model loading utilities
# 
# This module handles loading of trained PyTorch models and conversion
# from legacy TensorFlow/Keras models (.h5, .weights.h5) to PyTorch format.
# 
# For training data loading, see data_generator.py

import torch
import json
import os


def loadModel(nn_name, dirnn):
    """ Smart model loader that handles both PyTorch and legacy TensorFlow models.
        Loading priority:
        1. If .pth exists -> load directly (native PyTorch model)
        2. If .json + .h5 exist -> detect old TensorFlow model and convert
        3. If .json + .weights.h5 exist -> detect old TensorFlow model and convert
        4. Otherwise -> raise error
    """
    pth_path = os.path.join(dirnn, nn_name + '.pth')
    json_path = os.path.join(dirnn, nn_name + '.json')
    config_json_path = os.path.join(dirnn, nn_name + '_config.json')
    h5_path = os.path.join(dirnn, nn_name + '.h5')
    weights_h5_path = os.path.join(dirnn, nn_name + '.weights.h5')
    
    # Also accept .pt extension and stem_best.pt fallback
    pt_path = os.path.join(dirnn, nn_name + '.pt')
    best_pt_path = os.path.join(dirnn, nn_name + '_best.pt')
    if not os.path.isfile(pth_path):
        if os.path.isfile(pt_path):
            pth_path = pt_path
        elif os.path.isfile(best_pt_path):
            pth_path = best_pt_path

    # Priority 1: Load native PyTorch model
    if os.path.isfile(pth_path):
        print(f"Loading PyTorch model: {os.path.basename(pth_path)}")
        loaded = torch.load(pth_path, map_location='cpu', weights_only=False)
        
        if hasattr(loaded, 'eval'):
            loaded.eval()
            return loaded
        elif isinstance(loaded, dict):
            # Try _config.json first (new convention), then .json (old convention)
            config_file = config_json_path if os.path.isfile(config_json_path) else json_path
            if os.path.isfile(config_file):
                print(f"  Detected state_dict file, loading architecture from {os.path.basename(config_file)}")
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                model_type = config.get('model_type', 'CNN')
                
                if model_type == 'AST':
                    from src.models import architectures
                    import torch.nn as nn
                    num_classes = config.get('num_classes', 2)
                    input_size = config.get('input_size', None)
                    multilabel = config.get('multilabel', False)
                    dropout = config.get('dropout', 0.1)
                    model = architectures.AST(
                        num_classes=num_classes,
                        multilabel=multilabel,
                        input_size=input_size,
                        dropout=dropout
                    )
                    
                    # Handle position embeddings size mismatch for AST models
                    # (trained model may have interpolated position embeddings)
                    pos_emb_key = 'ast.embeddings.position_embeddings'
                    if pos_emb_key in loaded:
                        saved_pos_emb = loaded[pos_emb_key]
                        current_pos_emb = model.state_dict()[pos_emb_key]
                        
                        if saved_pos_emb.shape != current_pos_emb.shape:
                            print(f"  Position embeddings shape mismatch: "
                                  f"saved {saved_pos_emb.shape} vs current {current_pos_emb.shape}")
                            print(f"  Using saved (interpolated) position embeddings from trained model")
                            # Replace the position embeddings parameter with the saved one
                            model.ast.embeddings.position_embeddings = nn.Parameter(saved_pos_emb)
                            # Remove from loaded dict to avoid error
                            loaded.pop(pos_emb_key)
                    
                    # Load remaining state dict
                    missing_keys, unexpected_keys = model.load_state_dict(loaded, strict=False)
                    
                elif model_type == 'CNN':
                    from src.models import architectures
                    input_size = config.get('input_size', [128, 400])
                    num_classes = config.get('num_classes', 2)
                    model = architectures.CNNModel(input_size[0], input_size[1], num_classes)
                    model.load_state_dict(loaded)
                elif model_type == 'RegNet':
                    import timm
                    import torch.nn as nn
                    num_classes = config.get('num_classes', 2)
                    model_name_timm = config.get('model_name', 'regnety_008')
                    model = timm.create_model(model_name_timm, pretrained=False,
                                             in_chans=1, drop_rate=0.0, drop_path_rate=0.0)
                    backbone_out = model.head.fc.in_features
                    model.head.fc = nn.Identity()
                    # Attach pooling + classifier to match training architecture
                    model.pooling = nn.AdaptiveAvgPool2d(1)
                    model.classifier = nn.Linear(backbone_out, num_classes)
                    # Wrap in a small module that matches the forward pass used at training
                    class _RegNetWrapper(nn.Module):
                        def __init__(self, backbone, pooling, classifier):
                            super().__init__()
                            self.backbone = backbone
                            self.pooling = pooling
                            self.classifier = classifier
                        def forward(self, x):
                            features = self.backbone(x)
                            if isinstance(features, dict):
                                features = features['features']
                            if len(features.shape) == 4:
                                features = self.pooling(features)
                                features = features.view(features.size(0), -1)
                            return self.classifier(features)
                    wrapper = _RegNetWrapper(model, model.pooling, model.classifier)
                    missing_keys, unexpected_keys = wrapper.load_state_dict(loaded, strict=False)
                    if missing_keys:
                        print(f"  Missing keys (may be OK): {missing_keys[:3]}")
                    model = wrapper
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                model.eval()
                return model
            else:
                raise RuntimeError(
                    f"Loaded state_dict from {nn_name}.pth but no {nn_name}.json or {nn_name}_config.json found. "
                    f"Need config file to reconstruct model architecture."
                )
        else:
            raise RuntimeError(f"Unexpected object type in {nn_name}.pth: {type(loaded)}")
    
    # Priority 2: Convert legacy TensorFlow model
    # Check for both .h5 and .weights.h5, preferring .h5
    weights_path = None
    if os.path.isfile(json_path):
        if os.path.isfile(h5_path):
            weights_path = h5_path
            weights_ext = '.h5'
        elif os.path.isfile(weights_h5_path):
            weights_path = weights_h5_path
            weights_ext = '.weights.h5'
    
    if weights_path:
        # Determine format for user feedback
        if weights_ext == '.weights.h5':
            format_msg = "newer TensorFlow 2.x format"
        else:
            format_msg = "legacy TensorFlow format"
        
        print(f"⚠️  Detected {format_msg}: {nn_name}")
        print(f"    Converting to PyTorch format (using {weights_ext})...")
        
        from src.models.tf_to_torch_converter import convert_tf_model_to_pytorch
        
        try:
            model = convert_tf_model_to_pytorch(json_path, weights_path)
            model.eval()
            
            print(f"    ✓ Conversion successful!")
            print(f"    Saving converted model as {nn_name}.pth for future use...")
            
            # Save converted model for future use
            torch.save(model, pth_path)
            print(f"    ✓ Saved! Future loads will use the converted .pth file.")
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to convert TensorFlow model {nn_name}: {e}")
    
    # Priority 3: Only JSON exists (incomplete model)
    elif os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Found {nn_name}.json but missing weights file. "
            f"Need either {nn_name}.pth (PyTorch), {nn_name}.h5, or {nn_name}.weights.h5"
        )
    
    # Nothing found
    else:
        raise FileNotFoundError(
            f"No model files found for '{nn_name}' in {dirnn}. "
            f"Expected either {nn_name}.pth or {nn_name}.json + ({nn_name}.h5 or {nn_name}.weights.h5)"
        )


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# These are kept for training/testing code that creates models from scratch
# ============================================================================

def loadModelFromJson(jsonPath):
    """ Load NEW PyTorch model architecture from training config JSON.
        For loading existing models, use loadModel() instead.
    """
    with open(jsonPath, "r") as f:
        config = json.load(f)
    
    from src.models import architectures
    
    if config.get('model_type') == 'CNN':
        model = architectures.CNNModel(
            config['imageHeight'],
            config['imageWidth'],
            config['outputDim']
        )
        return model
    else:
        raise ValueError(f"Unknown model type: {config.get('model_type')}")


def loadWeights(model, weightsPath):
    """ Load weights (.pth) into an existing PyTorch model. """
    model.load_state_dict(torch.load(weightsPath, map_location='cpu', weights_only=True))
    model.eval()
    return model
