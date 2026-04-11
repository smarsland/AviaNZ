"""
Base trainer class with shared training infrastructure.

Consolidates common training logic from model_trainer.py and finetune_birdclef.py.
Eliminates ~1000 lines of duplication.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from . import config
from .trainer_config import TrainerConfig
from .models import GradientReversalLayer


class BaseTrainer(ABC):
    """Shared training infrastructure for AST and BirdClef trainers."""
    
    def __init__(self, cfg: TrainerConfig):
        """Initialize with structured config."""
        self.cfg = cfg
        
        # Unpack commonly used values
        self.output_folder = cfg.training.output_folder
        self.max_epochs = cfg.training.max_epochs
        self.batch_size = cfg.training.batch_size
        self.learning_rate = cfg.training.learning_rate
        self.weight_decay = cfg.training.weight_decay
        self.patience = cfg.training.patience
        self.use_amp = cfg.training.use_amp
        self.multilabel = cfg.model.multilabel
        self.use_dann = cfg.domain_adaptation.use_dann
        self.lambda_domain = cfg.domain_adaptation.lambda_domain
        self.use_cleaner = cfg.domain_adaptation.use_cleaner
        
        self._setup_device()
        self._setup_seed()
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize scaler for AMP
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Will be set by subclasses
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.num_classes = None
        
        # DANN components (if used)
        self.grl = None
        self.domain_classifier = None
        self.domain_criterion = None
        self.target_train_loader = None
        
        # Cleaner component (if used)
        self.cleaner = None
    
    def _setup_device(self):
        """Setup compute device."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def _setup_seed(self):
        """Set random seed for reproducibility."""
        if self.cfg.training.seed is not None:
            import random
            random.seed(self.cfg.training.seed)
            np.random.seed(self.cfg.training.seed)
            torch.manual_seed(self.cfg.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.training.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def create_model(self):
        """Create and return the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _compute_multilabel_metrics(self, outputs, targets):
        """Compute metrics for multilabel classification. Must be implemented by subclasses."""
        pass
    
    def setup_dann(self, feature_dim):
        """Setup DANN domain adaptation components."""
        if not self.use_dann:
            return
        
        print(f"  Setting up DANN (lambda={self.lambda_domain})...")
        
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.grl.to(self.device)
        self.domain_classifier.to(self.device)
        self.domain_criterion = nn.BCEWithLogitsLoss()
    
    def _domain_adaptation_step(self, features, batch_size_source, batch_size_target):
        """Perform DANN domain adaptation step."""
        if not self.use_dann:
            return torch.tensor(0.0), 0, 0
        
        # Normalize features
        norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
        reversed_features = self.grl(norm_features)
        domain_output = self.domain_classifier(reversed_features)
        
        # Create domain labels
        domain_labels_source = torch.zeros(batch_size_source).to(self.device)
        domain_labels_target = torch.ones(batch_size_target).to(self.device)
        domain_labels = torch.cat([domain_labels_source, domain_labels_target], dim=0)
        
        domain_loss = self.domain_criterion(domain_output.squeeze(), domain_labels)
        
        # Calculate domain accuracy
        domain_pred = (torch.sigmoid(domain_output.squeeze()) > 0.5).float()
        domain_correct = (domain_pred == domain_labels).sum().item()
        domain_total = domain_labels.size(0)
        
        return domain_loss, domain_correct, domain_total
    
    def train_epoch_dann(self, epoch):
        """Train one epoch with DANN domain adaptation."""
        self.model.train()
        if self.use_dann:
            self.grl.train()
            self.domain_classifier.train()
        if self.use_cleaner:
            self.cleaner.train()
        
        total_loss = 0
        total_class_loss = 0
        total_domain_loss = 0
        domain_correct = 0
        domain_total = 0
        
        # Set DANN lambda for this epoch
        p = float(epoch) / float(self.max_epochs)
        alpha = 2.0 / (1.0 + np.exp(-5 * p)) - 1.0
        self.grl.lambda_param = alpha * self.lambda_domain
        
        source_iter = iter(self.train_loader)
        target_iter = iter(self.target_train_loader)
        n_batches = min(len(self.train_loader), len(self.target_train_loader))
        
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{self.max_epochs}")
        
        for batch_idx in pbar:
            try:
                source_data, source_target = next(source_iter)
            except StopIteration:
                source_iter = iter(self.train_loader)
                source_data, source_target = next(source_iter)
            
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_train_loader)
                target_data, _ = next(target_iter)
            
            source_data = source_data.to(self.device)
            source_target = source_target.to(self.device)
            target_data = target_data.to(self.device)
            
            batch_size_s = source_data.size(0)
            batch_size_t = target_data.size(0)
            
            combined_data = torch.cat([source_data, target_data], dim=0)
            
            self.optimizer.zero_grad()
            
            # Get features from model
            features = self.model.backbone(combined_data)
            if isinstance(features, dict):
                features = features['features']
            if len(features.shape) == 4:
                features = self.model.pooling(features)
                features = features.view(features.size(0), -1)
            
            # Classification on source
            source_features = features[:batch_size_s]
            source_class_output = self.model.classifier(source_features)
            
            # Compute classification loss
            if self.multilabel:
                class_loss = self.criterion(source_class_output, source_target.float())
            else:
                if source_target.dim() == 2 and source_target.shape[1] > 1:
                    # Soft labels
                    is_soft = not torch.all((source_target == 0) | (source_target == 1))
                    if is_soft:
                        log_probs = torch.log_softmax(source_class_output, dim=1)
                        class_loss = -(source_target * log_probs).sum(dim=1).mean()
                    else:
                        target_labels = source_target.argmax(dim=1)
                        class_loss = self.criterion(source_class_output, target_labels)
                else:
                    target_labels = source_target.long()
                    class_loss = self.criterion(source_class_output, target_labels)
            
            # Domain adaptation loss
            domain_loss, dc, dt = self._domain_adaptation_step(features, batch_size_s, batch_size_t)
            domain_correct += dc
            domain_total += dt
            
            loss = class_loss + self.grl.lambda_param * domain_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.use_dann:
                torch.nn.utils.clip_grad_norm_(self.domain_classifier.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += loss.item()
            
            # Update progress bar
            dacc = 100.0 * domain_correct / domain_total if domain_total > 0 else 0
            pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}', 'dacc': f'{dacc:.1f}%'})
        
        avg_loss = total_loss / n_batches
        avg_dacc = 100.0 * domain_correct / domain_total if domain_total > 0 else 0
        
        return avg_loss, {'domain_acc': avg_dacc}
    
    def train_epoch_standard(self, epoch):
        """Train one epoch without DANN."""
        self.model.train()
        if self.use_cleaner:
            self.cleaner.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            # Compute loss
            if self.multilabel:
                loss = self.criterion(output, target.float())
            else:
                if target.dim() == 2 and target.shape[1] > 1:
                    # Soft labels
                    is_soft = not torch.all((target == 0) | (target == 1))
                    if is_soft:
                        log_probs = torch.log_softmax(output, dim=1)
                        loss = -(target * log_probs).sum(dim=1).mean()
                    else:
                        target_labels = target.argmax(dim=1)
                        loss = self.criterion(output, target_labels)
                else:
                    target_labels = target.long()
                    loss = self.criterion(output, target_labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, {}
    
    def validate(self):
        """Evaluate on validation set."""
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                
                # Compute loss
                if self.multilabel:
                    loss = self.criterion(output, target.float())
                else:
                    if target.dim() == 2:
                        target_labels = target.argmax(dim=1)
                        loss = self.criterion(output, target_labels)
                    else:
                        loss = self.criterion(output, target.long())
                
                total_loss += loss.item()
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        if self.multilabel:
            metrics = self._compute_multilabel_metrics(all_preds, all_targets)
            return avg_loss, metrics
        else:
            pred_labels = all_preds.argmax(dim=1)
            if all_targets.dim() == 2:
                target_labels = all_targets.argmax(dim=1)
            else:
                target_labels = all_targets
            accuracy = (pred_labels == target_labels).float().mean().item() * 100
            return avg_loss, accuracy
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.output_folder, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        
        if self.use_dann:
            checkpoint['domain_classifier_state_dict'] = self.domain_classifier.state_dict()
        
        if self.use_cleaner and self.cleaner:
            checkpoint['cleaner_state_dict'] = self.cleaner.state_dict()
        
        torch.save(checkpoint, path)
    
    def run_training(self):
        """Main training loop with early stopping."""
        print(f"\nStarting training for {self.max_epochs} epochs...")
        
        best_val_metric = -1.0
        epochs_without_improvement = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }
        
        for epoch in range(self.max_epochs):
            # Train
            if self.use_dann and self.target_train_loader:
                train_loss, train_info = self.train_epoch_dann(epoch)
            else:
                train_loss, train_info = self.train_epoch_standard(epoch)
            
            # Validate
            val_loss, val_metric = self.validate()
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss if val_loss is not None else None)
            
            # Print results
            print(f"\nEpoch {epoch+1}/{self.max_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
            if val_metric is not None:
                if isinstance(val_metric, dict):
                    print(f"  Val Metrics: {val_metric}")
                else:
                    print(f"  Val Metric: {val_metric:.2f}")
            
            # Early stopping based on validation
            if val_metric is not None:
                current_metric = val_metric['exact_match'] if isinstance(val_metric, dict) else val_metric
                
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    epochs_without_improvement = 0
                    self.save_checkpoint('model_best.pt')
                    print(f"  ✓ Saved best model (metric: {current_metric:.4f})")
                else:
                    epochs_without_improvement += 1
                    if self.patience > 0 and epochs_without_improvement >= self.patience:
                        print(f"\nEarly stopping: no improvement for {self.patience} epochs")
                        print(f"Best metric: {best_val_metric:.4f}")
                        break
            else:
                # No validation - save every epoch
                self.save_checkpoint('model_best.pt')
        
        # Save final model
        self.save_checkpoint('model_final.pt')
        
        # Save history
        history_path = os.path.join(self.output_folder, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print("\n✓ Training complete!")
        if best_val_metric > 0:
            print(f"  Best validation metric: {best_val_metric:.4f}")
