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

# PyTorch model training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json, os

from src.core import config_loader
from src.models import architectures
from src.utils.device import get_device


class ModelTrainer:
    """ This class implements model training in AviaNZ. """

    def __init__(self, configdir, species, calltypes, fs, length, windowwidth, inc, imageheight, imagewidth, modelArchitecture):
        self.species = species
        self.length = length
        self.windowwidth = windowwidth
        self.inc = inc
        self.imageheight = imageheight
        self.imagewidth = imagewidth
        self.calltypes = calltypes
        self.fs = fs
        self.modelArchitecture = modelArchitecture
        self.device = get_device()

        cl = config_loader.ConfigLoader()
        self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))

    def createArchitecture(self):
        """ Sets self.model """
        if self.modelArchitecture == 'CNN':
            self.model = architectures.CNNModel(self.imageheight, self.imagewidth, len(self.calltypes)+1)
            self.model = self.model.to(self.device)
        else:
            raise ValueError("Model architecture not supported")

    def train(self, modelsavepath, training_batch_generator, validation_batch_generator):
        """ Train the model using image generators. """

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        epochs = self.LearningDict['epochs']
        monitor = self.LearningDict['monitor']
        patience = self.LearningDict['patience']
        
        best_val_metric = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for i in range(len(training_batch_generator)):
                batch_x, batch_y = training_batch_generator[i]
                batch_x = torch.from_numpy(batch_x).float().permute(0, 3, 1, 2).to(self.device)
                batch_y = torch.from_numpy(batch_y).float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels_idx = torch.max(batch_y, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == labels_idx).sum().item()
            
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(len(validation_batch_generator)):
                    batch_x, batch_y = validation_batch_generator[i]
                    batch_x = torch.from_numpy(batch_x).float().permute(0, 3, 1, 2).to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, labels_idx = torch.max(batch_y, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == labels_idx).sum().item()
            
            val_acc = val_correct / val_total
            
            if 'accuracy' in monitor and val_acc > best_val_metric:
                best_val_metric = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(modelsavepath, f"{epoch:02d}-{val_loss/len(validation_batch_generator):.2f}-{val_acc:.2f}.pth"))
                patience_counter = 0
            elif 'loss' in monitor and (best_val_metric == 0 or val_loss < best_val_metric):
                best_val_metric = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(modelsavepath, f"{epoch:02d}-{val_loss/len(validation_batch_generator):.2f}-{val_acc:.2f}.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch+1}/{epochs}: Val Loss: {val_loss/len(validation_batch_generator):.4f}, Val Acc: {val_acc:.4f}")
        
        model_config = {
            'model_type': 'CNN',
            'imageHeight': self.imageheight,
            'imageWidth': self.imagewidth,
            'outputDim': len(self.calltypes) + 1
        }
        with open(os.path.join(modelsavepath, "model.json"), "w") as json_file:
            json.dump(model_config, json_file)
        print("Saved model to ", modelsavepath)
