#!/usr/bin/env python3
"""
Validate that train/test splits have consistent classes across datasets.

This ensures cross-dataset experiments can be run without class mismatch issues.

Usage:
    python validate_splits.py <path_to_split1> <path_to_split2>
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_split_classes(split_folder):
    """Load class information from a split folder."""
    labels_path = os.path.join(split_folder, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found in {split_folder}")
    
    with open(labels_path, 'r') as f:
        metadata = json.load(f)
    
    files = metadata.get('files', [])
    categories = set(metadata.get('categories', []))
    
    # Collect ALL classes (multilabel setting)
    file_classes = set()
    class_counts = defaultdict(int)
    for entry in files:
        for cls in entry.get('class_names', []):
            file_classes.add(cls)
            class_counts[cls] += 1
    
    return {
        'folder': split_folder,
        'categories': categories,
        'file_classes': file_classes,
        'class_counts': dict(class_counts),
        'num_files': len(files)
    }


def validate_splits(avianz_train, avianz_test, doc_train, doc_test):
    """Validate that splits have consistent classes."""
    
    print("="*60)
    print("Validating split consistency")
    print("="*60)
    
    # Load all splits
    av_train = load_split_classes(avianz_train)
    av_test = load_split_classes(avianz_test)
    doc_train_data = load_split_classes(doc_train)
    doc_test_data = load_split_classes(doc_test)
    
    print("\nDataset sizes:")
    print(f"  AviaNZ train: {av_train['num_files']} files, {len(av_train['file_classes'])} classes")
    print(f"  AviaNZ test:  {av_test['num_files']} files, {len(av_test['file_classes'])} classes")
    print(f"  DOC train:    {doc_train_data['num_files']} files, {len(doc_train_data['file_classes'])} classes")
    print(f"  DOC test:     {doc_test_data['num_files']} files, {len(doc_test_data['file_classes'])} classes")
    
    # Check for class consistency issues
    issues = []
    warnings = []
    
    # Issue 1: Test sets should have same classes
    av_test_classes = av_test['file_classes']
    doc_test_classes = doc_test_data['file_classes']
    
    if av_test_classes != doc_test_classes:
        only_avianz = av_test_classes - doc_test_classes
        only_doc = doc_test_classes - av_test_classes
        
        issues.append(f"Test sets have different classes!")
        if only_avianz:
            issues.append(f"  Only in AviaNZ test: {only_avianz}")
        if only_doc:
            issues.append(f"  Only in DOC test: {only_doc}")
    
    # Issue 2: Test classes should appear in training
    av_test_not_in_train = av_test_classes - av_train['file_classes']
    doc_test_not_in_train = doc_test_classes - doc_train_data['file_classes']
    
    if av_test_not_in_train:
        issues.append(f"AviaNZ test has classes not in AviaNZ train: {av_test_not_in_train}")
    
    if doc_test_not_in_train:
        issues.append(f"DOC test has classes not in DOC train: {doc_test_not_in_train}")
    
    # Warning: Classes only in training
    av_train_only = av_train['file_classes'] - av_test_classes
    doc_train_only = doc_train_data['file_classes'] - doc_test_classes
    
    if av_train_only:
        warnings.append(f"Classes only in AviaNZ train (not tested): {av_train_only}")
    
    if doc_train_only:
        warnings.append(f"Classes only in DOC train (not tested): {doc_train_only}")
    
    # Print results
    print("\n" + "="*60)
    if issues:
        print("❌ VALIDATION FAILED")
        print("="*60)
        for issue in issues:
            print(f"  • {issue}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        print("\n" + "="*60)
        return False
    else:
        print("✓ VALIDATION PASSED")
        print("="*60)
        print("  All test sets have consistent classes")
        print(f"  Shared test classes: {sorted(av_test_classes)}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        print("\n" + "="*60)
        return True


def main():
    parser = argparse.ArgumentParser(description='Validate split consistency')
    parser.add_argument('avianz_train', help='Path to AviaNZ train folder')
    parser.add_argument('avianz_test', help='Path to AviaNZ test folder')
    parser.add_argument('doc_train', help='Path to DOC train folder')
    parser.add_argument('doc_test', help='Path to DOC test folder')
    args = parser.parse_args()
    
    success = validate_splits(args.avianz_train, args.avianz_test, args.doc_train, args.doc_test)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
