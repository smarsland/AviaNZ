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


def loadModelFromJson(jsonPath):
    """Load model architecture from JSON and return uninitialized model."""
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


def loadModelFromH5(h5Path):
    """Load a PyTorch model from .pth or .pt file."""
    model = torch.load(h5Path, map_location='cpu')
    model.eval()
    return model


def loadWeights(model, weightsPath):
    """Load weights into an existing model.
    
    Args:
        model: PyTorch model instance
        weightsPath: Path to .pth weights file
        
    Returns:
        model with loaded weights
    """
    model.load_state_dict(torch.load(weightsPath, map_location='cpu'))
    model.eval()
    return model
