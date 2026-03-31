#!/usr/bin/env python3
"""Fix result.json files by reading test reports."""
import json
from pathlib import Path
import sys

exp = Path(sys.argv[1])
result_file = exp / 'result.json'

print(f"Fixing: {exp.name}")

with open(result_file) as f:
    result = json.load(f)

# Add training history metrics
history_file = exp / 'training_history.json'
if history_file.exists():
    with open(history_file) as f:
        history = json.load(f)
        if 'val_acc' in history and history['val_acc']:
            val_accs = [v for v in history['val_acc'] if v is not None]
            if val_accs:
                best_epoch = val_accs.index(max(val_accs))
                result['best_val_acc'] = max(val_accs) * 100
                if 'train_acc' in history:
                    train_accs = [v for v in history['train_acc'] if v is not None]
                    if len(train_accs) > best_epoch:
                        result['best_train_acc'] = train_accs[best_epoch] * 100

# Determine test names
name = result.get('name', exp.name)
if name.startswith('avianz_'):
    result['test1_name'] = 'avianz_split'
    result['test2_name'] = 'doc_split'
else:
    result['test1_name'] = 'doc_split'
    result['test2_name'] = 'avianz_split'

# Load test reports
for key, test_name in [('test1_acc', result['test1_name']), ('test2_acc', result['test2_name'])]:
    report_file = exp / f'ast_test_{test_name}_multilabel_report.json'
    if report_file.exists():
        with open(report_file) as f:
            report = json.load(f)
            if 'macro avg' in report and 'accuracy' in report['macro avg']:
                result[key] = report['macro avg']['accuracy'] * 100
                print(f"  {key}: {result[key]:.1f}%")

# Save
with open(result_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"✓ Updated")
