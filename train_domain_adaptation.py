#!/usr/bin/env python3
"""
Domain Adaptive Neural Network (DANN) training for cross-dataset bird classification.

This script implements DANN to align feature distributions between source
and target domains while learning to classify bird species.

Usage:
    python train_domain_adaptation.py source_data target_data output \\
        --pretrained BirdClefModels/model_fold0.pth \\
        --epochs 50 \\
        --lambda-domain 0.1
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from pathlib import Path

import config
from data_utils import DataLoader, create_data_loaders
from models import DANNModel
from evaluation_utils import EvaluationManager


class DANNTrainer:
    def __init__(self, source_folder, target_folder, output_folder, 
                 architecture='regnety_008', pretrained_path=None, 
                 epochs=50, batch_size=32, lr=1e-4, lambda_domain=0.1,
                 freeze_backbone=False, test_folder=None, test_folder2=None,
                 target_is_noise=False):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.output_folder = output_folder
        self.architecture = architecture
        self.pretrained_path = pretrained_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_domain = lambda_domain
        self.freeze_backbone = freeze_backbone
        self.test_folder = test_folder
        self.test_folder2 = test_folder2
        self.target_is_noise = target_is_noise
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"DANN Training Configuration:")
        print(f"  Source: {source_folder}")
        print(f"  Target: {target_folder} {'(NOISE - unlabeled)' if target_is_noise else ''}")
        print(f"  Output: {output_folder}")
        print(f"  Architecture: {architecture}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Lambda domain: {lambda_domain}")
        print(f"  Freeze backbone: {freeze_backbone}")
        print(f"  Device: {self.device}")
    
    def load_data(self):
        print("\nLoading datasets...")
        
        loader_source = DataLoader(self.source_folder)
        self.source_data = loader_source.load_data(use_multilabel=False, validation_share=0.0)
        
        if self.target_is_noise:
            loader_target = DataLoader(self.target_folder)
            target_raw = loader_target.load_data(use_multilabel=False, validation_share=0.0)
            
            num_noise_samples = len(target_raw['train_filenames'])
            num_classes = self.source_data['nclasses']
            dummy_labels = np.zeros((num_noise_samples, num_classes), dtype=np.float32)
            
            self.target_data = {
                'nclasses': self.source_data['nclasses'],
                'categories': self.source_data['categories'],
                'train_filenames': target_raw['train_filenames'],
                'train_labels': dummy_labels,
                'train_primary_species': ['noise'] * num_noise_samples,
                'train_noise_filenames': target_raw.get('train_noise_filenames', []),
                'test_filenames': [],
                'test_labels': np.zeros((0, num_classes), dtype=np.float32),
                'test_primary_species': [],
                'test_noise_filenames': []
            }
            print(f"  Target is NOISE (unlabeled data for domain adaptation)")
            print(f"  Loaded {num_noise_samples} noise files from {self.target_folder}")
        else:
            loader_target = DataLoader(self.target_folder)
            self.target_data = loader_target.load_data(use_multilabel=False, validation_share=0.15)
            if self.source_data['categories'] != self.target_data['categories']:
                raise ValueError("Source and target datasets must have same species")
        
        self.num_classes = self.source_data['nclasses']
        self.categories = self.source_data['categories']
        
        print(f"  Source samples: {len(self.source_data['train_filenames'])}")
        print(f"  Target train: {len(self.target_data['train_filenames'])}")
        print(f"  Target val: {len(self.target_data['test_filenames'])}")
        print(f"  Classes: {self.num_classes}")
        
        img_height = config.DEFAULT_FREQ_BINS
        img_width = config.DEFAULT_TIME_BINS
        num_workers = 4 if torch.cuda.is_available() else 2
        
        self.source_loader, _ = create_data_loaders(
            self.source_data, self.batch_size, img_height, img_width,
            config.DEFAULT_CHANNELS, cropping_mode='random',
            num_workers=num_workers, normalize='imagenet'
        )
        
        self.target_train_loader, self.target_val_loader = create_data_loaders(
            self.target_data, self.batch_size, img_height, img_width,
            config.DEFAULT_CHANNELS, cropping_mode='random',
            num_workers=num_workers, normalize='imagenet'
        )
        
        self.test_datasets = []
        if self.test_folder:
            test_loader = DataLoader(self.test_folder)
            test_data = test_loader.load_data(use_multilabel=False, validation_share=0.0)
            test_name = f"{Path(self.test_folder).parent.name}/{Path(self.test_folder).name}"
            self.test_datasets.append({
                'name': test_name,
                'filenames': test_data['train_filenames'],
                'labels': test_data['train_labels'],
                'primary_species': test_data['train_primary_species']
            })
        
        if self.test_folder2:
            test_loader2 = DataLoader(self.test_folder2)
            test_data2 = test_loader2.load_data(use_multilabel=False, validation_share=0.0)
            test_name2 = f"{Path(self.test_folder2).parent.name}/{Path(self.test_folder2).name}"
            self.test_datasets.append({
                'name': test_name2,
                'filenames': test_data2['train_filenames'],
                'labels': test_data2['train_labels'],
                'primary_species': test_data2['train_primary_species']
            })
    
    def create_model(self):
        print("\nCreating DANN model...")
        
        pretrained = False
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            pretrained = True
        
        self.model = DANNModel(
            num_classes=self.num_classes,
            backbone_name=self.architecture,
            pretrained=pretrained,
            freeze_backbone=self.freeze_backbone
        )
        
        if pretrained and self.pretrained_path:
            print(f"  Loading pretrained weights from {self.pretrained_path}")
            checkpoint = torch.load(self.pretrained_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            backbone_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_key = k.replace('backbone.', '')
                    if 'classifier' not in new_key and 'fc' not in new_key:
                        backbone_dict[new_key] = v
            
            self.model.feature_extractor.load_state_dict(backbone_dict, strict=False)
            print("  ✓ Loaded pretrained backbone")
        
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()
        
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
    
    def train_epoch(self, epoch):
        self.model.train()
        
        p = epoch / self.epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        alpha = alpha * self.lambda_domain
        
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_train_loader)
        
        n_batches = min(len(self.source_loader), len(self.target_train_loader))
        
        total_class_loss = 0
        total_domain_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for _ in pbar:
            try:
                source_data, source_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(self.source_loader)
                source_data, source_labels = next(source_iter)
            
            try:
                target_data, target_labels = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_train_loader)
                target_data, target_labels = next(target_iter)
            
            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            target_data = target_data.to(self.device)
            target_labels = target_labels.to(self.device)
            
            batch_size_s = source_data.size(0)
            batch_size_t = target_data.size(0)
            
            combined_data = torch.cat([source_data, target_data], dim=0)
            
            class_output, domain_output, _ = self.model(combined_data, alpha)
            
            source_class_output = class_output[:batch_size_s]
            class_loss = self.class_criterion(source_class_output, source_labels)
            
            domain_labels_source = torch.zeros(batch_size_s, 1).to(self.device)
            domain_labels_target = torch.ones(batch_size_t, 1).to(self.device)
            domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
            
            domain_loss = self.domain_criterion(domain_output, domain_labels)
            
            loss = class_loss + domain_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            
            _, predicted = torch.max(source_class_output, 1)
            total += source_labels.size(0)
            correct += (predicted == source_labels).sum().item()
            
            pbar.set_postfix({
                'cls_loss': f'{class_loss.item():.4f}',
                'dom_loss': f'{domain_loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'alpha': f'{alpha:.4f}'
            })
        
        avg_class_loss = total_class_loss / n_batches
        avg_domain_loss = total_domain_loss / n_batches
        train_acc = 100.0 * correct / total
        
        return avg_class_loss, avg_domain_loss, train_acc
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        # When target is noise (unlabeled), validate on source validation set instead
        val_loader = self.source_val_loader if self.target_is_noise else self.target_val_loader
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                class_output, _, _ = self.model(data, alpha=0.0)
                
                _, predicted = torch.max(class_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100.0 * correct / total
        return val_acc
    
    def test_on_dataset(self, test_dataset):
        self.model.eval()
        
        img_height = config.DEFAULT_FREQ_BINS
        img_width = config.DEFAULT_TIME_BINS
        
        test_data_dict = {
            'nclasses': self.num_classes,
            'categories': self.categories,
            'train_filenames': test_dataset['filenames'],
            'train_labels': test_dataset['labels'],
            'train_primary_species': test_dataset['primary_species'],
            'val_filenames': [],
            'val_labels': [],
            'val_primary_species': []
        }
        
        test_loader, _ = create_data_loaders(
            test_data_dict, self.batch_size, img_height, img_width,
            config.DEFAULT_CHANNELS, cropping_mode='center',
            num_workers=2, normalize='imagenet'
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                class_output, _, _ = self.model(data, alpha=0.0)
                
                _, predicted = torch.max(class_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100.0 * correct / total
        return test_acc
    
    def train(self):
        print("\nStarting DANN training...")
        
        history = {
            'train_class_loss': [],
            'train_domain_loss': [],
            'train_class_acc': [],
            'target_val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.epochs):
            class_loss, domain_loss, train_acc = self.train_epoch(epoch)
            val_acc = self.validate()
            
            history['train_class_loss'].append(class_loss)
            history['train_domain_loss'].append(domain_loss)
            history['train_class_acc'].append(train_acc)
            history['target_val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1}/{self.epochs}:")
            print(f"  Train class loss: {class_loss:.4f}")
            print(f"  Train domain loss: {domain_loss:.4f}")
            print(f"  Train class acc: {train_acc:.2f}%")
            val_label = "Source val acc" if self.target_is_noise else "Target val acc"
            print(f"  {val_label}: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(self.output_folder, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'categories': self.categories
                }, model_path)
                print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
            
            self.scheduler.step()
        
        print(f"\nTraining complete!")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
        
        if self.test_datasets:
            print(f"\nEvaluating on test sets...")
            for test_dataset in self.test_datasets:
                test_acc = self.test_on_dataset(test_dataset)
                print(f"  {test_dataset['name']} Accuracy: {test_acc:.2f}%")
        
        history_path = os.path.join(self.output_folder, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved training history to {history_path}")
    
    def run(self):
        self.load_data()
        self.create_model()
        self.train()


def main():
    parser = argparse.ArgumentParser(
        description='Train DANN for domain adaptation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('source', help='Source domain training folder')
    parser.add_argument('target', help='Target domain training folder')
    parser.add_argument('output', help='Output folder')
    parser.add_argument('--architecture', default='regnety_008',
                       help='Backbone architecture (default: regnety_008)')
    parser.add_argument('--pretrained', default=None,
                       help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--lambda-domain', type=float, default=0.1,
                       help='Domain loss weight (default: 0.1)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze feature extractor backbone')
    parser.add_argument('--target-is-noise', action='store_true',
                       help='Target domain is noise (unlabeled) data')
    parser.add_argument('--test-folder', default=None,
                       help='Test folder 1')
    parser.add_argument('--test-folder2', default=None,
                       help='Test folder 2')
    
    args = parser.parse_args()
    
    trainer = DANNTrainer(
        source_folder=args.source,
        target_folder=args.target,
        output_folder=args.output,
        architecture=args.architecture,
        pretrained_path=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_domain=args.lambda_domain,
        freeze_backbone=args.freeze_backbone,
        test_folder=args.test_folder,
        test_folder2=args.test_folder2,
        target_is_noise=args.target_is_noise
    )
    
    trainer.run()


if __name__ == '__main__':
    main()
