#!/usr/bin/env python3
"""Re-evaluate AST test sets to fix missing reports."""
import torch
import json
from pathlib import Path
from data_utils import DataLoader, SpectrogramDataset
from evaluation_utils import EvaluationManager
from models import AST
import config
import sys

if len(sys.argv) < 2:
    print("Usage: python reeval_ast.py <exp_folder> <test1> <test2>")
    sys.exit(1)

exp = Path(sys.argv[1])
test1 = sys.argv[2]
test2 = sys.argv[3]

print(f"Evaluating: {exp.name}")

# Load config and model
cfg = json.load(open(exp/'ast_model_config.json'))
model = AST(cfg['num_classes'], True, (cfg.get('freq_bins',128), cfg.get('time_bins',1024)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(exp/'ast_model_best.pt', map_location=device))
model.to(device).eval()

# Evaluate test1
print(f"\nTest 1: {test1}")
d = DataLoader(test1, None).load_data(True, 0.0)
ds = SpectrogramDataset(d['train_filenames'], d['train_labels'], cfg.get('freq_bins',128), cfg.get('time_bins',1024), 1, 'center', None, 0.0, None, False, None, False, False, 20, False)
dl = torch.utils.data.DataLoader(ds, 16, False, num_workers=2)
EvaluationManager(str(exp), d['class_names'], True).evaluate_model(model, dl, f'ast_test_{Path(test1).parent.name}', d, device)
print(f'✓ {Path(test1).parent.name}')

# Evaluate test2
print(f"\nTest 2: {test2}")
d = DataLoader(test2, None).load_data(True, 0.0)
ds = SpectrogramDataset(d['train_filenames'], d['train_labels'], cfg.get('freq_bins',128), cfg.get('time_bins',1024), 1, 'center', None, 0.0, None, False, None, False, False, 20, False)
dl = torch.utils.data.DataLoader(ds, 16, False, num_workers=2)
EvaluationManager(str(exp), d['class_names'], True).evaluate_model(model, dl, f'ast_test_{Path(test2).parent.name}', d, device)
print(f'✓ {Path(test2).parent.name}')

print('\nDone!')
