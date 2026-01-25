"""
Train Shape Transformer Model

This script trains a transformer model on pre-extracted shape sequences.
Requires sequences extracted by extract_shape_sequences.py.

The trained model and artifacts can be used with shape_transformer_predict.py to generate
predictions in the same format as other models for comparison.
"""

import argparse
import os
import json
import math
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ShapeSequenceDataset(Dataset):
    """Dataset that loads pre-extracted shape sequences from JSON files"""
    
    def __init__(self, filenames, label_mapping, mlb, sequence_dir, max_length=100, shape_dim=1024, use_intensity=True, augment=False):
        self.filenames = filenames
        self.label_mapping = label_mapping
        self.mlb = mlb
        self.sequence_dir = sequence_dir
        self.max_length = max_length
        self.shape_dim = shape_dim
        self.use_intensity = use_intensity
        self.n_continuous = 6 if use_intensity else 4
        self.augment = augment
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        json_path = os.path.join(self.sequence_dir, filename.replace('.npy', '.json'))
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        shapes = data['shapes']
        
        seq_len = min(len(shapes), self.max_length)
        
        shape_vectors = torch.zeros(self.max_length, self.shape_dim, dtype=torch.float32)
        continuous_features = torch.zeros(self.max_length, self.n_continuous, dtype=torch.float32)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        
        for i in range(seq_len):
            shape = shapes[i]
            shape_vectors[i] = torch.FloatTensor(shape['shape_vector'])
            continuous_features[i, 0] = shape['time_pos']
            continuous_features[i, 1] = shape['freq_pos']
            continuous_features[i, 2] = shape['duration']
            continuous_features[i, 3] = shape['bandwidth']
            mask[i] = True
        
        if self.augment:
            shape_vectors, mask = self._apply_augmentation(shape_vectors, mask, seq_len)
        
        class_names = self.label_mapping.get(filename, [])
        labels = torch.FloatTensor(self.mlb.transform([class_names])[0])
        
        return shape_vectors, continuous_features, mask, labels
    
    def _apply_augmentation(self, shape_vectors, mask, seq_len):
        if seq_len > 0 and np.random.rand() < 0.5:
            n_mask = max(1, int(seq_len * 0.15))
            mask_indices = np.random.choice(seq_len, size=n_mask, replace=False)
            mask[mask_indices] = False
        
        if np.random.rand() < 0.3:
            shape_vectors = shape_vectors + torch.randn_like(shape_vectors) * 0.05
        
        return shape_vectors, mask


class ShapeTransformer(nn.Module):
    def __init__(
        self,
        shape_dim=1024,
        cont_dim=6,
        d_model=96,
        nhead=4,
        num_layers=2,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(shape_dim + cont_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, shape_vectors, continuous_features, mask):
        x = torch.cat([shape_vectors, continuous_features], dim=-1)
        x = self.input_proj(x)

        x = self.transformer(
            x,
            src_key_padding_mask=~mask,
        )

        mask_f = mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        logits = self.classifier(pooled)
        token_logits = self.classifier(x)

        return logits, token_logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for shape_vectors, continuous_features, masks, labels in dataloader:
        shape_vectors = shape_vectors.to(device)
        continuous_features = continuous_features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(shape_vectors, continuous_features, masks)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for shape_vectors, continuous_features, masks, labels in dataloader:
            shape_vectors = shape_vectors.to(device)
            continuous_features = continuous_features.to(device)
            masks = masks.to(device)
            
            logits, _ = model(shape_vectors, continuous_features, masks)
            logits = torch.clamp(logits, min=-10, max=10)
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    
    return all_probs, all_preds, all_labels


def compute_detailed_metrics(probs, labels, class_names, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    
    probs = np.clip(probs, 0.0, 1.0)
    probs = np.nan_to_num(probs, nan=0.0)
    
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    
    valid_cols = labels.sum(axis=0) > 0
    if valid_cols.sum() > 0:
        try:
            ap_micro = average_precision_score(labels[:, valid_cols], probs[:, valid_cols], average='micro')
        except:
            ap_micro = 0.0
        try:
            ap_macro = average_precision_score(labels[:, valid_cols], probs[:, valid_cols], average='macro')
        except:
            ap_macro = 0.0
    else:
        ap_micro = 0.0
        ap_macro = 0.0
    
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_ap = np.zeros(labels.shape[1])
    for i in range(labels.shape[1]):
        if labels[:, i].sum() > 0:
            try:
                per_class_ap[i] = average_precision_score(labels[:, i], probs[:, i])
            except:
                per_class_ap[i] = 0.0
    
    try:
        auc_macro = roc_auc_score(labels[:, valid_cols], probs[:, valid_cols], average='macro')
    except:
        auc_macro = 0.0
    
    metrics = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'ap_micro': ap_micro,
        'ap_macro': ap_macro,
        'auc_macro': auc_macro,
        'per_class': {}
    }
    
    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'f1': float(per_class_f1[i]),
            'ap': float(per_class_ap[i]),
            'support': int(labels[:, i].sum())
        }
    
    return metrics


def optimize_threshold(probs, labels, metric='f1_micro'):
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probs >= threshold).astype(int)
        if metric == 'f1_micro':
            score = f1_score(labels, preds, average='micro', zero_division=0)
        elif metric == 'f1_macro':
            score = f1_score(labels, preds, average='macro', zero_division=0)
        else:
            score = f1_score(labels, preds, average='micro', zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def train_shape_transformer(data_folder, output_model, d_model=128, nhead=4, num_layers=3,
                            dropout=0.1, batch_size=32, lr=0.001,
                            epochs=50, train_split=0.8, seed=42, max_files=None,
                            weight_decay=0.01, patience=10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training config: d_model={d_model}, nhead={nhead}, layers={num_layers}, dropout={dropout}")
    print(f"Weight decay: {weight_decay}, Early stopping patience: {patience}")
    
    sequence_dir = os.path.join(data_folder, 'sequences')
    if not os.path.exists(sequence_dir):
        raise FileNotFoundError(f"Missing sequences directory: {sequence_dir}")
    labels_file = os.path.join(data_folder, 'labels.json')
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing labels file: {labels_file}")
    
    with open(labels_file, 'r') as f:
        labels_data = json.load(f)
    
    files = labels_data['files']
    filename_to_class = {item['filename']: item['class_names'] for item in files}
    
    all_files = [f['filename'] for f in files]
    all_labels = [filename_to_class[f] for f in all_files]
    
    if max_files is not None and max_files < len(all_files):
        indices = np.random.permutation(len(all_files))[:max_files]
        all_files = [all_files[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]
    
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    
    indices = np.random.permutation(len(all_files))
    split_idx = int(len(all_files) * train_split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_files = [all_files[i] for i in train_idx]
    test_files = [all_files[i] for i in test_idx]
    
    train_labels_binarized = mlb.transform([filename_to_class[f] for f in train_files])
    test_labels_binarized = mlb.transform([filename_to_class[f] for f in test_files])
    
    label_density = train_labels_binarized.mean()
    raw_pos_weight = (1 - label_density) / (label_density + 1e-5)
    pos_weight = torch.FloatTensor([min(10.0, raw_pos_weight)]).to(device)
    
    print(f"\nDataset: {len(train_files)} train, {len(test_files)} test")
    print(f"Classes: {len(mlb.classes_)}, Label density: {label_density:.4f}")
    print(f"Using pos_weight: {pos_weight.item():.2f} (raw: {raw_pos_weight:.2f})")
    
    train_dataset = ShapeSequenceDataset(train_files, filename_to_class, mlb, sequence_dir, augment=True)
    test_dataset = ShapeSequenceDataset(test_files, filename_to_class, mlb, sequence_dir, augment=False)
    
    sample_vecs, sample_feats, sample_mask, sample_labels = train_dataset[0]
    cont_dim = sample_feats.shape[1]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ShapeTransformer(
        shape_dim=1024,
        cont_dim=cont_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=len(mlb.classes_),
        dropout=dropout,
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    print("\nTraining...")
    best_val_f1 = 0.0
    best_epoch = -1
    patience_counter = 0
    history = {'train_loss': [], 'val_f1_micro': [], 'val_f1_macro': [], 'val_ap_micro': [], 'val_ap_macro': []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_probs, test_preds, test_labels = evaluate(model, test_loader, device, threshold=0.5)
        
        metrics = compute_detailed_metrics(test_probs, test_labels, mlb.classes_.tolist(), threshold=0.5)
        
        f1_micro = metrics['f1_micro']
        f1_macro = metrics['f1_macro']
        ap_micro = metrics['ap_micro']
        ap_macro = metrics['ap_macro']
        
        n_pred = test_preds.sum()
        n_true = test_labels.sum()
        pred_density = test_preds.mean()
        
        history['train_loss'].append(train_loss)
        history['val_f1_micro'].append(f1_micro)
        history['val_f1_macro'].append(f1_macro)
        history['val_ap_micro'].append(ap_micro)
        history['val_ap_macro'].append(ap_macro)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        prob_min = np.nanmin(test_probs) if not np.all(np.isnan(test_probs)) else 0.0
        prob_max = np.nanmax(test_probs) if not np.all(np.isnan(test_probs)) else 1.0
        prob_mean = np.nanmean(test_probs)
        
        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | F1-micro: {f1_micro:.4f} | F1-macro: {f1_macro:.4f} | AP: {ap_micro:.4f} | LR: {current_lr:.6f}")
        print(f"         | Preds: {n_pred:.0f}/{test_preds.size} ({pred_density:.4f}) | True: {n_true:.0f} | Prob: mean={prob_mean:.3f} [{prob_min:.3f}, {prob_max:.3f}]")
        
        if f1_macro > best_val_f1:
            best_val_f1 = f1_macro
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_model)
            print(f"  ✓ New best model saved (F1-macro: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered! No improvement for {patience} epochs (best: {best_val_f1:.4f} at epoch {best_epoch})")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Top-3 classes by F1:")
            per_class_metrics = [(name, metrics['per_class'][name]['f1']) for name in mlb.classes_]
            per_class_metrics.sort(key=lambda x: x[1], reverse=True)
            for name, f1 in per_class_metrics[:3]:
                print(f"    {name}: {f1:.4f}")
    
    print(f"\nBest validation F1-macro: {best_val_f1:.4f} at epoch {best_epoch}")
    
    model.load_state_dict(torch.load(output_model, map_location=device))
    print("\nOptimizing classification threshold...")
    test_probs, _, test_labels = evaluate(model, test_loader, device, threshold=0.5)
    optimal_threshold, optimal_f1 = optimize_threshold(test_probs, test_labels, metric='f1_macro')
    print(f"Optimal threshold: {optimal_threshold:.3f} (F1-macro: {optimal_f1:.4f})")
    
    final_metrics = compute_detailed_metrics(test_probs, test_labels, mlb.classes_.tolist(), threshold=optimal_threshold)
    print(f"\nFinal metrics with optimized threshold:")
    print(f"  F1-micro: {final_metrics['f1_micro']:.4f}")
    print(f"  F1-macro: {final_metrics['f1_macro']:.4f}")
    print(f"  AP-micro: {final_metrics['ap_micro']:.4f}")
    print(f"  AP-macro: {final_metrics['ap_macro']:.4f}")
    print(f"  AUC-macro: {final_metrics['auc_macro']:.4f}")
    
    print(f"\nPer-class performance:")
    per_class_list = [(name, final_metrics['per_class'][name]['f1'], final_metrics['per_class'][name]['support']) 
                      for name in mlb.classes_]
    per_class_list.sort(key=lambda x: x[1], reverse=True)
    for name, f1, support in per_class_list:
        print(f"  {name:30s} F1: {f1:.4f}  (n={support})")
    
    config = {
        'class_names': mlb.classes_.tolist(),
        'shape_dim': 1024,
        'cont_dim': cont_dim,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dropout': dropout,
        'num_classes': len(mlb.classes_),
        'optimal_threshold': float(optimal_threshold),
        'best_val_f1_macro': float(best_val_f1),
        'final_metrics': final_metrics,
    }
    
    with open(output_model.replace('.pt', '_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    with open(output_model.replace('.pt', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nSaved: {output_model}")
    print(f"Config: {output_model.replace('.pt', '_config.json')}")
    print(f"History: {output_model.replace('.pt', '_history.json')}")


def main():
    parser = argparse.ArgumentParser(description="Train Shape Transformer Model")
    
    parser.add_argument('data_folder', type=str,
                       help="Path to folder with sequences/ subdirectory and labels.json")
    parser.add_argument('output_model', type=str,
                       help="Path to save trained model (.pt file)")
    parser.add_argument('--d-model', type=int, default=128,
                       help="Transformer model dimension (default: 128)")
    parser.add_argument('--nhead', type=int, default=4,
                       help="Number of attention heads (default: 4)")
    parser.add_argument('--num-layers', type=int, default=3,
                       help="Number of transformer layers (default: 3)")
    parser.add_argument('--dropout', type=float, default=0.1,
                       help="Dropout rate (default: 0.1)")
    parser.add_argument('--batch-size', type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument('--epochs', type=int, default=100,
                       help="Epochs (default: 100)")
    parser.add_argument('--train-split', type=float, default=0.8,
                       help="Train split (default: 0.8)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument('--max-files', type=int, default=None,
                       help="Maximum number of files to use (default: None = use all)")
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help="Weight decay for regularization (default: 0.01)")
    parser.add_argument('--patience', type=int, default=15,
                       help="Early stopping patience (default: 15)")
    
    args = parser.parse_args()
    
    train_shape_transformer(
        data_folder=args.data_folder,
        output_model=args.output_model,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        train_split=args.train_split,
        seed=args.seed,
        max_files=args.max_files,
        weight_decay=args.weight_decay,
        patience=args.patience
    )


if __name__ == '__main__':
    main()
