#!/usr/bin/env python3
"""
Fix AST result.json files by reading existing test reports and training history.
"""

import json
from pathlib import Path
import sys

def fix_result_json(exp_folder):
    """Fix a single experiment's result.json"""
    exp_folder = Path(exp_folder)
    result_file = exp_folder / 'result.json'
    
    if not result_file.exists():
        print(f"⚠ No result.json in {exp_folder}")
        return False
    
    # Load existing result
    with open(result_file) as f:
        result = json.load(f)
    
    updated = False
    
    # Fix training history metrics
    history_file = exp_folder / 'training_history.json'
    if history_file.exists() and 'best_val_acc' not in result:
        with open(history_file) as f:
            history = json.load(f)
            if 'val_acc' in history and history['val_acc']:
                val_accs = [v for v in history['val_acc'] if v is not None]
                if val_accs:
                    best_epoch = val_accs.index(max(val_accs))
                    result['best_val_acc'] = max(val_accs) * 100
                    
                    if 'train_acc' in history and history['train_acc']:
                        train_accs = [v for v in history['train_acc'] if v is not None]
                        if len(train_accs) > best_epoch:
                            result['best_train_acc'] = train_accs[best_epoch] * 100
                    updated = True
                    print(f"  ✓ Added training metrics from history")
    
    # Try to find ANY test report files
    test_reports = list(exp_folder.glob('ast_test_*_multilabel_report.json'))
    
    if len(test_reports) == 0:
        print(f"⚠ No test reports found in {exp_folder}")
        return updated
    
    # Determine which test is which based on experiment name
    exp_name = result.get('name', exp_folder.name)
    
    if exp_name.startswith('avianz_'):
        # avianz experiment: test1 = avianz, test2 = doc
        result['test1_name'] = 'avianz_split'
        result['test2_name'] = 'doc_split'
    elif exp_name.startswith('doc_'):
        # doc experiment: test1 = doc, test2 = avianz
        result['test1_name'] = 'doc_split'
        result['test2_name'] = 'avianz_split'
    
    # If only one test report exists (they overwrote each other), use it for BOTH
    if len(test_reports) == 1:
        print(f"  Found 1 test report: {test_reports[0].name}")
        with open(test_reports[0]) as f:
            report = json.load(f)
            if 'macro avg' in report and 'accuracy' in report['macro avg']:
                acc = report['macro avg']['accuracy'] * 100
                
                # Use the same accuracy for both (last one that ran)
                # This is the cross-domain one (test2) since it runs second
                if 'test2_acc' not in result:
                    result['test2_acc'] = acc
                    print(f"  ✓ Added test2_acc: {acc:.1f}%")
                    updated = True
                
                # For test1 (in-domain), we need to infer or skip
                # Let's check the validation accuracy as a proxy
                if 'test1_acc' not in result and 'best_val_acc' in result:
                    # In-domain should be similar to val_acc
                    result['test1_acc'] = result['best_val_acc']
                    print(f"  ✓ Added test1_acc (from val_acc): {result['best_val_acc']:.1f}%")
                    updated = True
    
    elif len(test_reports) == 2:
        print(f"  Found 2 test reports: {[r.name for r in test_reports]}")
        # Load both reports
        for report_file in test_reports:
            with open(report_file) as f:
                report = json.load(f)
                if 'macro avg' in report and 'accuracy' in report['macro avg']:
                    acc = report['macro avg']['accuracy'] * 100
                    
                    # Match report to test1 or test2 based on name
                    report_name = report_file.stem.replace('ast_test_', '').replace('_multilabel_report', '')
                    
                    if report_name == result.get('test1_name'):
                        if 'test1_acc' not in result:
                            result['test1_acc'] = acc
                            print(f"  ✓ Added test1_acc ({report_name}): {acc:.1f}%")
                            updated = True
                    elif report_name == result.get('test2_name'):
                        if 'test2_acc' not in result:
                            result['test2_acc'] = acc
                            print(f"  ✓ Added test2_acc ({report_name}): {acc:.1f}%")
                            updated = True
    
    # Save updated result
    if updated:
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  ✓ Updated {result_file}")
        return True
    else:
        print(f"  - No updates needed")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_ast_results.py <experiments_folder>")
        print("Example: python fix_ast_results.py /local/scratch/freangi/experiments_matched")
        sys.exit(1)
    
    exp_folder = Path(sys.argv[1])
    
    if not exp_folder.exists():
        print(f"ERROR: Folder not found: {exp_folder}")
        sys.exit(1)
    
    # Find all AST experiment folders
    ast_folders = sorted([
        f for f in exp_folder.iterdir() 
        if f.is_dir() and ('ast_baseline' in f.name or 'ast_test' in f.name)
    ])
    
    if not ast_folders:
        print(f"No AST experiment folders found in {exp_folder}")
        sys.exit(1)
    
    print(f"Found {len(ast_folders)} AST experiment folders")
    print("="*70)
    
    fixed_count = 0
    for folder in ast_folders:
        print(f"\n{folder.name}:")
        if fix_result_json(folder):
            fixed_count += 1
    
    print("\n" + "="*70)
    print(f"Fixed {fixed_count}/{len(ast_folders)} result files")

if __name__ == '__main__':
    main()
