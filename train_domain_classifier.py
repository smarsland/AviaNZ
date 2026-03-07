"""
Domain Classifier for Diagnosing Dataset Generalization Issues

This script trains a binary classifier to distinguish between two datasets.
High accuracy indicates significant domain shift (datasets are very different).
Low accuracy (~50%) indicates similar domains (good generalization expected).

Usage:
    python train_domain_classifier.py dataset1/ dataset2/ outputs/domain_classifier
    
    # With BirdClef pretrained backbone
    python train_domain_classifier.py dataset1/ dataset2/ outputs/domain --pretrained BirdClefModels/model_fold0.pth
    
    # Simple CNN (faster, no pretraining)
    python train_domain_classifier.py dataset1/ dataset2/ outputs/domain --architecture simple_cnn
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import timm
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import config
from data_utils import DataLoader, SpectrogramDataset


class GradCAM:
    """Grad-CAM for visualizing what the model focuses on."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def remove_hooks(self):
        pass


class SimpleCNN(nn.Module):
    """Simple CNN for domain classification."""
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        return self.classifier(features)


class DomainClassifierModel(nn.Module):
    """Domain classifier using BirdClef backbone or simple CNN."""
    
    def __init__(self, architecture='resnet18', pretrained_path=None):
        super().__init__()
        self.architecture = architecture
        
        if architecture == 'simple_cnn':
            self.model = SimpleCNN()
        else:
            self.backbone = timm.create_model(
                architecture,
                pretrained=False,
                in_chans=1,
                drop_rate=0.0,
                drop_path_rate=0.0
            )
            
            if 'efficientnet' in architecture:
                backbone_out = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            elif 'resnet' in architecture:
                backbone_out = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif 'regnet' in architecture:
                backbone_out = self.backbone.head.fc.in_features
                self.backbone.head.fc = nn.Identity()
            else:
                backbone_out = self.backbone.get_classifier().in_features
                self.backbone.reset_classifier(0, '')
            
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(backbone_out, 2)
            
            if pretrained_path:
                self.load_pretrained_backbone(pretrained_path)
    
    def load_pretrained_backbone(self, pretrained_path):
        print(f"Loading pretrained backbone from {pretrained_path}")
        
        import sys
        if '__main__' not in sys.modules:
            sys.modules['__main__'] = sys.modules[__name__]
        
        class CFG:
            pass
        
        sys.modules[__name__].CFG = CFG
        globals()['CFG'] = CFG
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Warning: Could not load full checkpoint: {e}")
            checkpoint = {'model_state_dict': torch.load(pretrained_path, map_location='cpu', weights_only=False)}
        
        backbone_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                if 'classifier' not in new_key and 'fc' not in new_key:
                    backbone_dict[new_key] = v
        
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print("  Loaded pretrained backbone weights")
    
    def forward(self, x):
        if self.architecture == 'simple_cnn':
            return self.model(x)
        else:
            features = self.backbone(x)
            if isinstance(features, dict):
                features = features['features']
            if len(features.shape) == 4:
                features = self.pooling(features)
                features = features.view(features.size(0), -1)
            return self.classifier(features)


class DomainClassifierTrainer:
    """Trains and evaluates domain classifier."""
    
    def __init__(self, dataset1_folder, dataset2_folder, output_folder,
                 architecture='resnet18', pretrained_path=None, 
                 epochs=20, batch_size=32, lr=1e-4, 
                 normalize=False, validation_split=0.2, device=None,
                 num_visualizations=10):
        
        self.dataset1_folder = dataset1_folder
        self.dataset2_folder = dataset2_folder
        self.output_folder = output_folder
        self.architecture = architecture
        self.pretrained_path = pretrained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.normalize = normalize
        self.validation_split = validation_split
        self.num_visualizations = num_visualizations
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Domain Classifier Training Setup")
        print(f"  Dataset 1: {dataset1_folder}")
        print(f"  Dataset 2: {dataset2_folder}")
        print(f"  Output: {output_folder}")
        print(f"  Architecture: {architecture}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
    
    def load_data(self):
        print("\nLoading datasets...")
        
        loader1 = DataLoader(self.dataset1_folder)
        data1 = loader1.load_data(use_multilabel=False, validation_share=0.0)
        filenames1 = data1['train_filenames']
        
        loader2 = DataLoader(self.dataset2_folder)
        data2 = loader2.load_data(use_multilabel=False, validation_share=0.0)
        filenames2 = data2['train_filenames']
        
        print(f"  Dataset 1 samples: {len(filenames1)}")
        print(f"  Dataset 2 samples: {len(filenames2)}")
        
        all_filenames = filenames1 + filenames2
        domain_labels = [0] * len(filenames1) + [1] * len(filenames2)
        
        indices = np.random.permutation(len(all_filenames))
        all_filenames = [all_filenames[i] for i in indices]
        domain_labels = [domain_labels[i] for i in indices]
        
        domain_labels_onehot = []
        for label in domain_labels:
            onehot = [0.0, 0.0]
            onehot[label] = 1.0
            domain_labels_onehot.append(onehot)
        
        if self.validation_split > 0:
            split_idx = int(len(all_filenames) * (1 - self.validation_split))
            train_filenames = all_filenames[:split_idx]
            train_labels = domain_labels_onehot[:split_idx]
            train_labels_int = domain_labels[:split_idx]
            val_filenames = all_filenames[split_idx:]
            val_labels = domain_labels_onehot[split_idx:]
        else:
            train_filenames = all_filenames
            train_labels = domain_labels_onehot
            train_labels_int = domain_labels
            val_filenames = []
            val_labels = []
        
        img_height = config.DEFAULT_FREQ_BINS
        img_width = config.DEFAULT_TIME_BINS
        
        self.train_dataset = SpectrogramDataset(
            train_filenames,
            train_labels,
            img_height,
            img_width,
            config.DEFAULT_CHANNELS,
            cropping_mode='random',
            noise_filenames=None,
            noise_ratio=0.0,
            spec_transform=None,
            training=True,
            width_downsizing=None,
            normalize=self.normalize,
            use_sparse_patches=False,
            num_sparse_patches=0,
            use_temporal_roll=True,
            remove_baseline=False
        )
        
        if val_filenames:
            self.val_dataset = SpectrogramDataset(
                val_filenames,
                val_labels,
                img_height,
                img_width,
                config.DEFAULT_CHANNELS,
                cropping_mode='center',
                noise_filenames=None,
                noise_ratio=0.0,
                spec_transform=None,
                training=False,
                width_downsizing=None,
                normalize=self.normalize,
                use_sparse_patches=False,
                num_sparse_patches=0,
                use_temporal_roll=False,
                remove_baseline=False
            )
        else:
            self.val_dataset = None
        
        num_workers = 4 if torch.cuda.is_available() else 2
        self.train_loader = TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        if self.val_dataset:
            self.val_loader = TorchDataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        print(f"  Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"  Val samples: {len(self.val_dataset)}")
        
        domain1_train = sum(1 for l in train_labels_int if l == 0)
        domain2_train = sum(1 for l in train_labels_int if l == 1)
        print(f"  Train balance: Dataset1={domain1_train}, Dataset2={domain2_train}")
    
    def create_model(self):
        print("\nCreating model...")
        
        self.model = DomainClassifierModel(
            architecture=self.architecture,
            pretrained_path=self.pretrained_path
        )
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            if target.dim() == 2:
                target_labels = target.argmax(dim=1)
            else:
                target_labels = target.long()
            
            loss = self.criterion(output, target_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target_labels).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target_labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total, all_preds, all_labels
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if target.dim() == 2:
                    target_labels = target.argmax(dim=1)
                else:
                    target_labels = target.long()
                
                loss = self.criterion(output, target_labels)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target_labels).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target_labels.cpu().numpy())
        
        return total_loss / len(self.val_loader), 100. * correct / total, all_preds, all_labels
    
    def plot_confusion_matrix(self, labels, preds, title, filename):
        cm = confusion_matrix(labels, preds)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=[0, 1],
               yticks=[0, 1],
               xticklabels=['Dataset 1', 'Dataset 2'],
               yticklabels=['Dataset 1', 'Dataset 2'],
               title=title,
               ylabel='True Domain',
               xlabel='Predicted Domain')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        plt.savefig(os.path.join(self.output_folder, filename))
        plt.close()
    
    def train(self):
        print("\nStarting training...")
        
        best_val_acc = -1.0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.epochs):
            train_loss, train_acc, train_preds, train_labels = self.train_epoch(epoch)
            
            if self.val_loader:
                val_loss, val_acc, val_preds, val_labels = self.validate()
            else:
                val_loss, val_acc = None, None
            
            self.scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('domain_classifier_best.pt')
                print(f"  Saved best model (val_acc: {val_acc:.2f}%)")
            elif val_acc is None:
                self.save_model('domain_classifier_best.pt')
        
        self.save_model('domain_classifier_final.pt')
        
        history_path = os.path.join(self.output_folder, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if self.val_loader:
            self.model.load_state_dict(torch.load(os.path.join(self.output_folder, 'domain_classifier_best.pt')))
            _, final_val_acc, val_preds, val_labels = self.validate()
            
            self.plot_confusion_matrix(
                val_labels, val_preds, 
                'Validation Confusion Matrix', 
                'confusion_matrix_val.png'
            )
            
            print(f"\n{'='*60}")
            print(f"DOMAIN SHIFT ANALYSIS")
            print(f"{'='*60}")
            print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"\nInterpretation:")
            if best_val_acc >= 90:
                print("  SEVERE domain shift - datasets are very different!")
                print("  Models trained on one dataset will likely fail on the other.")
                print("  Consider: domain adaptation, data augmentation, or combined training.")
            elif best_val_acc >= 75:
                print("  MODERATE domain shift - datasets have noticeable differences.")
                print("  Some generalization issues expected.")
                print("  Consider: fine-tuning or data augmentation.")
            elif best_val_acc >= 60:
                print("  MILD domain shift - datasets somewhat similar.")
                print("  Reasonable generalization expected with proper training.")
            else:
                print("  MINIMAL domain shift - datasets are very similar!")
                print("  Good generalization expected across datasets.")
            
            print(f"\nClassification Report:")
            print(classification_report(
                val_labels, val_preds, 
                target_names=['Dataset 1', 'Dataset 2'],
                digits=4
            ))
        else:
            print(f"\nFinal Training Accuracy: {train_acc:.2f}%")
            print("(No validation set - accuracy may be overfitted)")
        
        self.plot_training_curves(history)
        
        print(f"\nResults saved to: {self.output_folder}")
        
        if self.num_visualizations > 0:
            self.generate_gradcam_visualizations()
    
    def get_target_layer(self):
        if self.architecture == 'simple_cnn':
            return self.model.model.conv_layers[-3]
        elif 'resnet' in self.architecture:
            return self.model.backbone.layer4[-1].conv2
        elif 'efficientnet' in self.architecture:
            return self.model.backbone.conv_head
        elif 'regnet' in self.architecture:
            return self.model.backbone.s4.b1.f.c3
        else:
            layers = list(self.model.backbone.children())
            return layers[-2]
    
    def generate_gradcam_visualizations(self):
        print(f"\n{'='*60}")
        print("Generating Grad-CAM Visualizations")
        print(f"{'='*60}")
        
        viz_folder = os.path.join(self.output_folder, 'gradcam_visualizations')
        os.makedirs(viz_folder, exist_ok=True)
        
        target_layer = self.get_target_layer()
        gradcam = GradCAM(self.model, target_layer)
        
        dataset = self.val_dataset if self.val_dataset else self.train_dataset
        
        domain0_indices = []
        domain1_indices = []
        
        for idx in range(len(dataset)):
            label = dataset.labels[idx]
            if isinstance(label, (list, tuple)):
                domain = 0 if label[0] > label[1] else 1
            elif torch.is_tensor(label):
                if label.numel() > 1:
                    domain = 0 if label[0] > label[1] else 1
                else:
                    domain = int(label.item())
            else:
                domain = int(label)
            
            if domain == 0 and len(domain0_indices) < self.num_visualizations:
                domain0_indices.append(idx)
            elif domain == 1 and len(domain1_indices) < self.num_visualizations:
                domain1_indices.append(idx)
            
            if len(domain0_indices) >= self.num_visualizations and len(domain1_indices) >= self.num_visualizations:
                break
        
        print(f"Generating {len(domain0_indices)} visualizations for Dataset 1")
        print(f"Generating {len(domain1_indices)} visualizations for Dataset 2")
        
        for idx in tqdm(domain0_indices, desc="Dataset 1"):
            self.visualize_sample(dataset, idx, gradcam, viz_folder, 'dataset1')
        
        for idx in tqdm(domain1_indices, desc="Dataset 2"):
            self.visualize_sample(dataset, idx, gradcam, viz_folder, 'dataset2')
        
        print(f"Visualizations saved to: {viz_folder}")
    
    def visualize_sample(self, dataset, idx, gradcam, output_folder, dataset_name):
        img, label = dataset[idx]
        img_tensor = img.unsqueeze(0).to(self.device)
        
        spec_filename = dataset.filenames[idx]
        
        if spec_filename.endswith('.npz'):
            spec_data = np.load(spec_filename)
            if 'spec' in spec_data:
                raw_spec = spec_data['spec']
            else:
                raw_spec = spec_data[spec_data.files[0]]
        else:
            raw_spec = np.load(spec_filename)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_class = output.argmax(dim=1).item()
            probs = torch.softmax(output, dim=1)[0]
        
        cam = gradcam.generate_cam(img_tensor, target_class=pred_class)
        
        img_np = img.squeeze().cpu().numpy()
        
        zoom_factors = (raw_spec.shape[0] / cam.shape[0], raw_spec.shape[1] / cam.shape[1])
        cam_resized = zoom(cam, zoom_factors, order=1)
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        
        im0 = axes[0].imshow(raw_spec, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
        axes[0].set_title(f'RAW Spectrogram from .npz file\nShape: {raw_spec.shape}, Range: [{raw_spec.min():.2f}, {raw_spec.max():.2f}]', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency Bin', fontsize=10)
        axes[0].set_xlabel('Time Bin', fontsize=10)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(img_np, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
        axes[1].set_title(f'What MODEL sees (after dataset transforms)\nShape: {img_np.shape}, Range: [{img_np.min():.2f}, {img_np.max():.2f}]', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency Bin', fontsize=10)
        axes[1].set_xlabel('Time Bin', fontsize=10)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(cam_resized, aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=1, interpolation='none')
        axes[2].set_title('Grad-CAM Heatmap (Red = High Attention)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Frequency Bin', fontsize=10)
        axes[2].set_xlabel('Time Bin', fontsize=10)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        axes[3].imshow(raw_spec, aspect='auto', origin='lower', cmap='gray', interpolation='none')
        im3 = axes[3].imshow(cam_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.6, vmin=0, vmax=1, interpolation='none')
        
        pred_label = 'Dataset 1' if pred_class == 0 else 'Dataset 2'
        
        if isinstance(label, (list, tuple)):
            true_class = 0 if label[0] > label[1] else 1
        elif torch.is_tensor(label):
            if label.numel() > 1:
                true_class = 0 if label[0] > label[1] else 1
            else:
                true_class = int(label.item())
        else:
            true_class = int(label)
        
        true_label = 'Dataset 1' if true_class == 0 else 'Dataset 2'
        axes[3].set_title(f'Overlay on RAW spectrogram\nTrue: {true_label} | Pred: {pred_label} | Conf: {probs[pred_class]:.2%}', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Frequency Bin', fontsize=10)
        axes[3].set_xlabel('Time Bin', fontsize=10)
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        output_path = os.path.join(output_folder, f'{dataset_name}_{idx:03d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history['train_loss'], label='Train Loss')
        if history['val_loss'][0] is not None:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history['train_acc'], label='Train Acc')
        if history['val_acc'][0] is not None:
            axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random Guess')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'training_curves.png'), dpi=150)
        plt.close()
    
    def save_model(self, filename):
        model_path = os.path.join(self.output_folder, filename)
        torch.save(self.model.state_dict(), model_path)
        
        config_path = model_path.replace('.pt', '_config.json')
        config_dict = {
            'model_type': 'domain_classifier',
            'architecture': self.architecture,
            'dataset1_folder': self.dataset1_folder,
            'dataset2_folder': self.dataset2_folder,
            'normalize': self.normalize,
            'freq_bins': config.DEFAULT_FREQ_BINS,
            'time_bins': config.DEFAULT_TIME_BINS
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Train domain classifier to diagnose dataset generalization issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - compare two datasets
  python train_domain_classifier.py data/dataset1 data/dataset2 outputs/domain
  
  # With BirdClef pretrained backbone (better features)
  python train_domain_classifier.py data/dataset1 data/dataset2 outputs/domain \\
      --pretrained BirdClefModels/model_fold0.pth --architecture regnety_008
  
  # Simple CNN (faster, no external dependencies)
  python train_domain_classifier.py data/dataset1 data/dataset2 outputs/domain \\
      --architecture simple_cnn --epochs 30
  
  # With normalization (if datasets use different recording equipment)
  python train_domain_classifier.py data/dataset1 data/dataset2 outputs/domain \\
      --normalize --epochs 25

Interpretation:
  - High accuracy (>90%): Severe domain shift, poor generalization expected
  - Medium accuracy (70-90%): Moderate shift, some generalization issues
  - Low accuracy (~50%): Minimal shift, good generalization expected
        """
    )
    
    parser.add_argument('dataset1', type=str,
                       help="Path to first dataset folder (with labels.json)")
    parser.add_argument('dataset2', type=str,
                       help="Path to second dataset folder (with labels.json)")
    parser.add_argument('output_folder', type=str,
                       help="Path to output folder for results")
    parser.add_argument('--architecture', default='resnet18',
                       choices=['simple_cnn', 'resnet18', 'resnet34', 'regnety_008', 'efficientnet_b0'],
                       help="Model architecture (default: resnet18)")
    parser.add_argument('--pretrained', default=None,
                       help="Path to pretrained checkpoint (optional)")
    parser.add_argument('--epochs', type=int, default=20,
                       help="Number of training epochs (default: 20)")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument('--normalize', action='store_true',
                       help="Apply background normalization")
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    parser.add_argument('--device', default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument('--num-visualizations', type=int, default=10,
                       help="Number of Grad-CAM visualizations to generate per dataset (default: 10)")
    
    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.dataset1, 'labels.json')):
        print(f"ERROR: labels.json not found in {args.dataset1}")
        return
    
    if not os.path.exists(os.path.join(args.dataset2, 'labels.json')):
        print(f"ERROR: labels.json not found in {args.dataset2}")
        return
    
    trainer = DomainClassifierTrainer(
        dataset1_folder=args.dataset1,
        dataset2_folder=args.dataset2,
        output_folder=args.output_folder,
        architecture=args.architecture,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        normalize=args.normalize,
        validation_split=args.validation_split,
        device=torch.device(args.device) if args.device else None,
        num_visualizations=args.num_visualizations
    )
    
    trainer.load_data()
    trainer.create_model()
    trainer.train()


if __name__ == '__main__':
    main()
