
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

# PyTorch model loading utilities

import torch
import json
import os


def loadModel(nn_name, dirnn):
    """
    Smart model loader that handles both PyTorch and legacy TensorFlow models.
    
    Args:
        nn_name: Base name of the model (without extension)
        dirnn: Directory containing model files
        
    Returns:
        Loaded PyTorch model in evaluation mode
        
    Loading priority:
        1. If .pth exists -> load directly (native PyTorch model)
        2. If .json + .h5 exist -> detect old TensorFlow model and convert
        3. Otherwise -> raise error
    """
    pth_path = os.path.join(dirnn, nn_name + '.pth')
    json_path = os.path.join(dirnn, nn_name + '.json')
    h5_path = os.path.join(dirnn, nn_name + '.h5')
    
    # Priority 1: Load native PyTorch model
    if os.path.isfile(pth_path):
        print(f"Loading PyTorch model: {nn_name}.pth")
        model = torch.load(pth_path, map_location='cpu', weights_only=False)
        model.eval()
        return model
    
    # Priority 2: Convert legacy TensorFlow model
    elif os.path.isfile(json_path) and os.path.isfile(h5_path):
        print(f"⚠️  Detected legacy TensorFlow model: {nn_name}")
        print(f"    Converting to PyTorch format...")
        
        from src.models.tf_to_torch_converter import convert_tf_model_to_pytorch
        
        try:
            model = convert_tf_model_to_pytorch(json_path, h5_path)
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
            f"Need either {nn_name}.pth (PyTorch) or {nn_name}.h5 (TensorFlow legacy)"
        )
    
    # Nothing found
    else:
        raise FileNotFoundError(
            f"No model files found for '{nn_name}' in {dirnn}. "
            f"Expected either {nn_name}.pth or {nn_name}.json + {nn_name}.h5"
        )


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# These are kept for training/testing code that creates models from scratch
# ============================================================================

def loadModelFromJson(jsonPath):
    """
    Load NEW PyTorch model architecture from training config JSON.
    
    Note: This is for NEWLY TRAINED models only, not for loading existing models!
    The JSON format here is different from TensorFlow's JSON format.
    
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
    """
    Load weights (.pth) into an existing PyTorch model.
    
    Used during training evaluation to load checkpoint weights.
    For loading complete models, use loadModel() instead.
    """
    model.load_state_dict(torch.load(weightsPath, map_location='cpu', weights_only=True))
    model.eval()
    return model
