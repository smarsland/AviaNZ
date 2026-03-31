#!/usr/bin/env python3
"""Re-run test evaluation using the ACTUAL training code."""
import sys
import subprocess

experiments = [
    {
        'folder': '/local/scratch/freangi/experiments_matched/avianz_ast_baseline_seed573',
        'train': '/local/scratch/freangi/data_matched/avianz_split/train',
        'test1': '/local/scratch/freangi/data_matched/avianz_split/test',
        'test2': '/local/scratch/freangi/data_matched/doc_split/test',
    },
    {
        'folder': '/local/scratch/freangi/experiments_matched/doc_ast_baseline_seed573',
        'train': '/local/scratch/freangi/data_matched/doc_split/train',
        'test1': '/local/scratch/freangi/data_matched/doc_split/test',
        'test2': '/local/scratch/freangi/data_matched/avianz_split/test',
    }
]

for exp in experiments:
    print(f"\n{'='*70}")
    print(f"Re-evaluating: {exp['folder'].split('/')[-1]}")
    print(f"{'='*70}")
    
    # Just run the evaluation part by calling train_models.py
    # Since the model exists, it should skip training
    cmd = [
        'python3', 'train_models.py',
        exp['train'],
        exp['folder'],
        '--model', 'ast',
        '--multilabel',
        '--test-folder', exp['test1'],
        '--test-folder2', exp['test2'],
        '--epochs', '1',  # Doesn't matter
        '--batch-size', '16',
    ]
    
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)

print("\n✓ Done!")
