#!/usr/bin/env python3
"""
Re-evaluate AST models on test sets to recover missing metrics.
Only runs evaluation (no training) - very fast.
"""

import json
import sys
import subprocess
from pathlib import Path

def needs_reevaluation(exp_folder):
    """Check if experiment needs test re-evaluation"""
    result_file = exp_folder / 'result.json'
    
    if not result_file.exists():
        return False, "No result.json"
    
    with open(result_file) as f:
        result = json.load(f)
    
    # Check if model exists
    model_file = exp_folder / 'ast_model_best.pt'
    if not model_file.exists():
        return False, "No model file"
    
    # Check what's missing
    missing = []
    if 'test1_acc' not in result:
        missing.append('test1')
    if 'test2_acc' not in result:
        missing.append('test2')
    
    if missing:
        return True, f"Missing: {', '.join(missing)}"
    else:
        return False, "Complete"

def get_test_folders(exp_name, base_folders):
    """Determine test folder paths based on experiment name"""
    if exp_name.startswith('avianz_'):
        return base_folders['avianz_test'], base_folders['doc_test']
    elif exp_name.startswith('doc_'):
        return base_folders['doc_test'], base_folders['avianz_test']
    else:
        return None, None

def main():
    if len(sys.argv) < 3:
        print("Usage: python reeval_ast_tests.py <experiments_folder> <avianz_test> <doc_test>")
        print("Example: python reeval_ast_tests.py /local/scratch/freangi/experiments_matched \\")
        print("         /local/scratch/freangi/data_matched/avianz_split/test \\")
        print("         /local/scratch/freangi/data_matched/doc_split/test")
        print()
        print("NOTE: Run this from the AviaNZ code directory!")
        sys.exit(1)
    
    exp_folder = Path(sys.argv[1])
    avianz_test = sys.argv[2]
    doc_test = sys.argv[3]
    
    base_folders = {
        'avianz_test': avianz_test,
        'doc_test': doc_test
    }
    
    # Find AST experiments that need re-evaluation
    ast_folders = sorted([
        f for f in exp_folder.iterdir() 
        if f.is_dir() and 'ast_baseline' in f.name
    ])
    
    if not ast_folders:
        print(f"No AST experiment folders found in {exp_folder}")
        sys.exit(1)
    
    print(f"Checking {len(ast_folders)} AST experiments...")
    print("="*70)
    
    to_reeval = []
    for folder in ast_folders:
        needs, reason = needs_reevaluation(folder)
        status = "⚠ NEEDS REEVAL" if needs else "✓"
        print(f"{status} {folder.name}: {reason}")
        if needs:
            to_reeval.append(folder)
    
    if not to_reeval:
        print("\n✓ All experiments have complete results!")
        return
    
    print(f"\n{'='*70}")
    print(f"Re-evaluating {len(to_reeval)} experiments...")
    print(f"{'='*70}\n")
    
    for i, folder in enumerate(to_reeval, 1):
        print(f"\n[{i}/{len(to_reeval)}] {folder.name}")
        
        # Load result to get experiment details
        with open(folder / 'result.json') as f:
            result = json.load(f)
        
        exp_name = result.get('name', folder.name)
        test1_folder, test2_folder = get_test_folders(exp_name, base_folders)
        
        if not test1_folder or not test2_folder:
            print(f"  ⚠ Cannot determine test folders for {exp_name}")
            continue
        
        # Get the training data folder from the original experiment
        # We need to infer it from the experiment name
        if exp_name.startswith('avianz_'):
            train_folder = avianz_test.replace('/test', '/train')
        elif exp_name.startswith('doc_'):
            train_folder = doc_test.replace('/test', '/train')
        else:
            print(f"  ⚠ Cannot determine train folder for {exp_name}")
            continue
        
        # Run Python code directly to evaluate without retraining
        eval_script = f"""
import torch
import json
from pathlib import Path
from data_utils import DataLoader, SpectrogramDataset
from evaluation_utils import EvaluationManager
from models import ASTModel
import config

# Load model config
config_path = Path('{folder}') / 'ast_model_config.json'
with open(config_path) as f:
    model_config = json.load(f)

# Create model
num_classes = model_config['num_classes']
model = ASTModel(
    label_dim=num_classes,
    fstride=10, tstride=10,
    input_fdim=128, input_tdim=1024,
    imagenet_pretrain=False,
    audioset_pretrain=False,
    model_size='base384'
)

# Load weights
model_path = Path('{folder}') / 'ast_model_best.pt'
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Load test data and evaluate
for test_num, test_folder in enumerate(['{test1_folder}', '{test2_folder}'], 1):
    print(f"\\nEvaluating test{{test_num}}: {{test_folder}}")
    
    loader = DataLoader(test_folder, noise_folder=None)
    test_data = loader.load_data(use_multilabel=True, validation_share=0.0)
    
    dataset = SpectrogramDataset(
        test_data['train_filenames'], test_data['train_labels'],
        128, 1024, 1, 'center',
        noise_filenames=None, noise_ratio=0.0, spec_transform=None,
        training=False, width_downsizing=None, normalize=False,
        use_sparse_patches=False, num_sparse_patches=20,
        use_temporal_roll=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Evaluate
    test_name = Path(test_folder).parent.name
    evaluator = EvaluationManager('{folder}', test_data['class_names'], multilabel=True)
    evaluator.evaluate_model(model, test_loader, f'ast_test_{{test_name}}', test_data, device=device)
    
    print(f"Saved: ast_test_{{test_name}}_multilabel_report.json")
"""
        
        # Write temp script and run it FROM THE AVIANZ DIRECTORY
        temp_script = Path('temp_eval_ast.py')
        with open(temp_script, 'w') as f:
            f.write(eval_script)
        
        print(f"  Running evaluation...")
        result = subprocess.run(['python3', str(temp_script)])
        
        if result.returncode == 0:
            print(f"  ✓ Evaluation complete")
            temp_script.unlink()  # Clean up
        else:
            print(f"  ✗ Evaluation failed")
            print(f"  Debug script saved at: {temp_script}")

if __name__ == '__main__':
    main()
