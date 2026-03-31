"""
Evaluation utilities for model performance assessment.
This module handles confusion matrices, classification reports, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
import torch
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, multilabel_confusion_matrix


class EvaluationManager:
    """Handles model evaluation and results visualization."""
    
    def __init__(self, outputs_folder, class_names, is_multilabel):
        """
        Initialize EvaluationManager.
        
        Args:
            outputs_folder: Path to save evaluation results
            class_names: List of class names
            is_multilabel: Whether this is multi-label classification
        """
        self.outputs_folder = outputs_folder
        self.class_names = class_names
        self.is_multilabel = is_multilabel
    
    def evaluate_model(self, model, test_loader, name, data, device='cpu'):
        """
        Generate comprehensive evaluation metrics for the model.
        
        Args:
            model: Trained PyTorch model
            test_loader: PyTorch DataLoader for test data
            name: Model name for saving files
            data: Dictionary containing all data
            device: Device to run evaluation on
        """
        print(f"Generating evaluation metrics for {name}...")
        
        model.eval()
        y_pred_probs = []
        y_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle both sparse and standard formats
                if isinstance(batch, dict):
                    # Sparse patches mode
                    patches = batch['patches'].to(device)
                    positions = batch['positions'].to(device)
                    mask = batch['mask'].to(device)
                    batch_labels = batch['label']
                    outputs = model(patches, sparse_mode=True, positions=positions, mask=mask)
                else:
                    # Standard mode
                    batch_data, batch_labels = batch
                    batch_data = batch_data.to(device)
                    outputs = model(batch_data)
                
                # Handle reconstruction models that return (logits, reconstruction)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert logits to probabilities
                if self.is_multilabel:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                y_pred_probs.extend(probs)
                y_true.extend(batch_labels.numpy())
        
        y_pred_probs = np.array(y_pred_probs)
        y_true = np.array(y_true)
        
        if self.is_multilabel:
            self._evaluate_multilabel(y_true, y_pred_probs, name, data)
        else:
            self._evaluate_singlelabel(y_true, y_pred_probs, name)
    
    def _evaluate_multilabel(self, y_true, y_pred_probs, name, data):
        """Evaluate multi-label classification model."""
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # Calculate per-class accuracy (fraction of correct predictions for each class)
        per_class_accuracy = []
        for i in range(y_true.shape[1]):
            class_correct = np.sum(y_true[:, i] == y_pred[:, i])
            class_total = y_true.shape[0]
            per_class_accuracy.append(class_correct / class_total if class_total > 0 else 0.0)
        per_class_accuracy = np.array(per_class_accuracy)
        
        # Create classification report
        class_report = {}
        for i, class_name in enumerate(self.class_names):
            class_report[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1-score': float(f1[i]),
                'accuracy': float(per_class_accuracy[i]),
                'support': int(support[i])
            }
        
        # Add macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        accuracy_macro = float(np.mean(per_class_accuracy))
        
        class_report['macro avg'] = {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1-score': float(f1_macro),
            'accuracy': accuracy_macro,
            'support': int(np.sum(support))
        }
        class_report['micro avg'] = {
            'precision': float(precision_micro),
            'recall': float(recall_micro),
            'f1-score': float(f1_micro),
            'accuracy': float(np.mean(y_true == y_pred)),  # Overall bit-wise accuracy
            'support': int(np.sum(support))
        }
        
        # Get per-class confusion matrices
        cm_multi = multilabel_confusion_matrix(y_true, y_pred)
        
        # Save multi-label specific metrics
        self._save_multilabel_metrics(y_true, y_pred, y_pred_probs, class_report, name, cm_multi)
        
        # Create visualizations
        self._plot_multilabel_confusion_matrices(cm_multi, name)
    
    def _evaluate_singlelabel(self, y_true, y_pred_probs, name):
        """Evaluate single-label classification model."""
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true_single = np.argmax(y_true, axis=1)
        
        # Generate list of all possible class labels (0 to num_classes-1)
        labels = list(range(len(self.class_names)))
        
        # Generate confusion matrix with all labels
        cm = confusion_matrix(y_true_single, y_pred, labels=labels)
        
        # Generate classification report with all labels
        class_report = classification_report(y_true_single, y_pred, labels=labels, target_names=self.class_names, output_dict=True, zero_division=0)
        
        # Save and visualize results
        self._save_confusion_matrix(cm, class_report, name)
        self._plot_confusion_matrix(cm, name)
    
    def _save_multilabel_metrics(self, y_true, y_pred, y_pred_probs, class_report, name, cm_multi):
        """Save multi-label specific metrics."""
        os.makedirs(self.outputs_folder, exist_ok=True)
        
        # Save classification report as JSON
        with open(os.path.join(self.outputs_folder, f"{name}_multilabel_report.json"), "w") as f:
            json.dump(class_report, f, indent=2)
        
        # Save per-class metrics in CSV format
        metrics_csv_path = os.path.join(self.outputs_folder, f"{name}_multilabel_metrics.csv")
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Support'])
            
            for class_name in self.class_names:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    writer.writerow([
                        class_name,
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}",
                        f"{metrics['f1-score']:.4f}",
                        f"{metrics['accuracy']:.4f}",
                        metrics['support']
                    ])
            
            # Add summary rows
            for avg_type in ['macro avg', 'micro avg']:
                if avg_type in class_report:
                    metrics = class_report[avg_type]
                    writer.writerow([
                        avg_type,
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}",
                        f"{metrics['f1-score']:.4f}",
                        f"{metrics['accuracy']:.4f}",
                        metrics['support']
                    ])
        
        # Calculate multi-label specific metrics
        from sklearn.metrics import hamming_loss, jaccard_score
        
        hamming = hamming_loss(y_true, y_pred)
        jaccard = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Exact match ratio (all labels must match exactly)
        exact_match = np.mean(np.all(y_true == y_pred, axis=1))
        
        print(f"Multi-label metrics - Hamming Loss: {hamming:.4f}, Jaccard Score: {jaccard:.4f}, Exact Match: {exact_match:.4f}")
    
    def _save_confusion_matrix(self, cm, class_report, name):
        """Save confusion matrix and classification report as files."""
        os.makedirs(self.outputs_folder, exist_ok=True)
        
        # Save confusion matrix as CSV
        np.savetxt(os.path.join(self.outputs_folder, f"{name}_confusion_matrix.csv"), 
                   cm, delimiter=',', fmt='%d')
        
        # Save classification report as JSON (contains all metrics)
        with open(os.path.join(self.outputs_folder, f"{name}_classification_report.json"), "w") as f:
            json.dump(class_report, f, indent=2)
    
    def _plot_confusion_matrix(self, cm, name):
        """Plot and save confusion matrix as an image."""
        # Calculate figure size based on number of classes and label lengths
        max_label_length = max(len(name) for name in self.class_names)
        base_size = max(8, len(self.class_names) * 0.8)
        fig_width = base_size + max(3, max_label_length * 0.15)
        fig_height = base_size + 2
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create the heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title(f'Confusion Matrix - {name}', fontsize=14, pad=20)
        
        # Set tick marks and labels
        tick_marks = np.arange(len(self.class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        
        # Truncate long names for display
        max_display_length = 30
        display_names = [name[:max_display_length] + '...' if len(name) > max_display_length else name for name in self.class_names]
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_yticklabels(display_names)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=max(8, 12 - len(self.class_names) // 5))
        
        plt.subplots_adjust(bottom=0.2, left=0.2, right=0.9, top=0.9)
        output_path = os.path.join(self.outputs_folder, f"{name}_confusion_matrix.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        print(f"Saved confusion matrix plot to {output_path}")
    
    def _plot_multilabel_confusion_matrices(self, cm_multi, name):
        """Plot per-class binary confusion matrices for multi-label classification."""
        n_classes = len(self.class_names)
        
        # Calculate grid dimensions
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # Handle case where we only have one subplot
        if n_classes == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_classes > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            cm_class = cm_multi[i]
            ax = axes[i]
            
            # Plot binary confusion matrix for this class
            im = ax.imshow(cm_class, interpolation='nearest', cmap='Blues')
            ax.set_title(f'{class_name}\n(Binary Classification)', fontsize=10)
            
            # Set binary labels
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Absent', 'Present'])
            ax.set_yticklabels(['Absent', 'Present'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Add text annotations
            thresh = cm_class.max() / 2
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(cm_class[row, col]),
                           ha="center", va="center",
                           color="white" if cm_class[row, col] > thresh else "black",
                           fontsize=12)
        
        # Hide unused subplots
        for j in range(n_classes, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(self.outputs_folder, f"{name}_multilabel_per_class_cm.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved per-class confusion matrices to {output_path}")
