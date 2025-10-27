
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

# Holds the models for NN training, does some saving / loading stuff

import torch
import torch.nn as nn
import json
import numpy as np


class CNNModel(nn.Module):
    def __init__(self, imageHeight, imageWidth, outputDim):
        super(CNNModel, self).__init__()
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.outputDim = outputDim
        
        apply_same_padding = imageHeight < 120 or imageWidth < 120
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding='same' if apply_same_padding else 0)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding='same' if apply_same_padding else 0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding='same' if apply_same_padding else 0)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding='same' if apply_same_padding else 0)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout(0.2)
        
        dummy_input = torch.zeros(1, 1, imageHeight, imageWidth)
        dummy_output = self.forward_conv(dummy_input)
        flattened_size = dummy_output.view(-1).shape[0]
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, outputDim)
        
    def forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = torch.relu(self.conv4(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = torch.relu(self.conv5(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        return x
    
    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout5(x)
        x = torch.softmax(self.fc2(x), dim=1)
        return x


def loadModel(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = CNNModel(checkpoint['imageHeight'], checkpoint['imageWidth'], checkpoint['outputDim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def saveModel(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'imageHeight': model.imageHeight,
        'imageWidth': model.imageWidth,
        'outputDim': model.outputDim
    }, path)


