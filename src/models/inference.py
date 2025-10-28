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

# PyTorch inference utilities

import torch


def configure_gpu_memory():
    """Configure GPU memory settings for PyTorch."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_batch(model, features):
    """Run batch prediction on features using the given model.
    
    Args:
        model: PyTorch model
        features: NumPy array or Tensor with shape (batch, height, width, channels) or (batch, height, width)
        
    Returns:
        NumPy array of predictions
    """
    model.eval()
    with torch.no_grad():
        if isinstance(features, torch.Tensor):
            tensor_input = features
        else:
            tensor_input = torch.from_numpy(features).float()
        
        # Convert from (batch, height, width, channels) to (batch, channels, height, width)
        if len(tensor_input.shape) == 4 and tensor_input.shape[-1] == 1:
            tensor_input = tensor_input.permute(0, 3, 1, 2)
        elif len(tensor_input.shape) == 3:
            tensor_input = tensor_input.unsqueeze(1)
        
        device = next(model.parameters()).device
        tensor_input = tensor_input.to(device)
        
        predictions = model(tensor_input)
        return predictions.cpu().numpy()
