
import numpy as np
import pandas as pd
import pylab as pl
import os, time, copy

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import make_grid
from torchvision import transforms, models
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

# Data loader
class BatDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(labels_file)
        #print(np.shape(self.labels))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.labels.iloc[index, 0])
        image = np.load(img_path)
        image = torch.from_numpy(image).float()
        # Change 2D into 3D, all channels the same
        image = image.unsqueeze((0)).repeat(3,1,1)
        label = self.labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    # Specify the transformations to be applied to the data
    # Normalisation is the specified one for the pre-trained nets
    # Resizing is for the same reason
    dataTrans = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            transforms.Resize(224),
            transforms.RandomAutocontrast(),
            #transforms.RandomEqualize(),
            #transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def prepData(dirt,validation_split=0.2, batchSize=32):

    # Split into training and validation and set up loaders datasets
    dataset = BatDataset(os.path.join(dirt,'labels.txt'),dirt)

    # Creating data indices for training and validation splits
    indices = np.arange(dataset.__len__())
    np.random.shuffle(indices)
    split = int(np.floor(validation_split * dataset.__len__()))
    inds = {}
    inds['train'] = indices[split:]
    inds['val'] = indices[:split]
    
    # Load samples
    sampler = {d: SubsetRandomSampler(inds[d]) for d in ['train','val']}
    dataset = {d: BatDataset(os.path.join(dirt,'labels.txt'),dirt, transform=BatDataset.dataTrans[d]) for d in ['train','val']}
    dataloaders = {d: DataLoader(dataset[d], batch_size=batchSize, sampler=sampler[d]) for d in ['train','val']}
    
    dataset_sizes = {d: len(inds[d]) for d in ['train', 'val']}
    print(dataset_sizes)
    class_names = ['LT','ST','Noise']

    return dataloaders, class_names, dataset_sizes

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def resnet(trainAll,dataloaders,class_names,dataset_sizes,ext,num_epochs):
    # Train all
    model_rn = models.resnet18(pretrained=True)
    if not trainAll:
        for param in model_rn.parameters():
            param.requires_grad = False
    model_rn.fc = nn.Linear(model_rn.fc.in_features, len(class_names))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_rn = model_rn.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_rn = optim.SGD(model_rn.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_rn, step_size=7, gamma=0.1)
    model_rn = train_model(model_rn, dataloaders, dataset_sizes, criterion, optimizer_rn, exp_lr_scheduler, device, num_epochs=num_epochs)
    if trainAll:
    	torch.save(model_rn.state_dict(), 'bat-res-full'+str(ext)+'.pth')
    else:
    	torch.save(model_rn.state_dict(), 'bat-res'+str(ext)+'.pth')
    
def vgg(trainAll,dataloaders, class_names,dataset_sizes,ext,num_epochs):
    # VGG train all
    model_vgg = models.vgg16(pretrained=True)
    # The VGG fully connected layer is different. Took a while to find that...
    model_vgg.classifier[6].out_features = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_vgg = model_vgg.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_vgg = optim.SGD(model_vgg.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_vgg, step_size=7, gamma=0.1)
    model_vgg = train_model(model_vgg,dataloaders, dataset_sizes, criterion, optimizer_vgg, exp_lr_scheduler, device, num_epochs=num_epochs)
    torch.save(model_vgg.state_dict(), 'bat-vgg'+str(ext)+'.pth')

class BatNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.2),
            nn.Conv2d(64,64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(64,64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(64*17*17, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, data):
        return self.network(data)

def npnet(trainAll,dataloaders,class_names,dataset_sizes,ext,num_epochs):
    model = BatNetwork()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # NP actually used ADAM
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, device, num_epochs=num_epochs)
    torch.save(model.state_dict(), 'bat-np'+str(ext)+'.pth')
    #history = fit(model, dataloaders['train'], dataloaders['val'])

def run():
   num_epochs=20
   dataloaders, class_names, dataset_sizes = prepData('Full/img224_64_16/', validation_split=0.2, batchSize=32)
   #print("Resnet Part, 224")
   #resnet(False,dataloaders,class_names,dataset_sizes,224,num_epochs)
   #print("Resnet All, 224")
   #resnet(True,dataloaders,class_names,dataset_sizes,224,num_epochs)
   print("VGG, 224")
   vgg(False,dataloaders,class_names,dataset_sizes,224,num_epochs)
   print("NP, 224")
   npnet(False,dataloaders,class_names,dataset_sizes,224,num_epochs)

run()
