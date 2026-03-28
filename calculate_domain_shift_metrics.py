"""
Calculate domain shift metrics that control for intrinsic dataset difficulty.

For each test domain, we compare:
- In-domain model (trained on same domain): baseline performance
- Cross-domain model (trained on different domain): degraded performance
- Percentage reduction: quantifies domain shift severity

This controls for difficulty by comparing performance on the SAME test set.
"""

import json
from pathlib import Path


def load_results(results_file='all_results.json'):
    """Load experiment results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def find_experiment(results, train_dataset_key, spec_transform, suffix=''):
    """
    Find experiment by training dataset and spec transform.
    
    Args:
        train_dataset_key: 'avianz' or 'doc'
        spec_transform: e.g., 'Log', 'PCEN', 'Log+normalize', 'dann_Log+normalize'
        suffix: optional suffix like '_intensity0.5' or '_variety100'
    """
    # Handle DANN specially
    if spec_transform.startswith('dann_'):
        base_transform = spec_transform.replace('dann_', '')
        search_name = f"{train_dataset_key}_dann_{base_transform}{suffix}"
    else:
        search_name = f"{train_dataset_key}_baseline_{spec_transform}{suffix}"
    
    for exp in results:
        if exp['name'] == search_name:
            return exp
    
    # Handle special cases
    if '+' in spec_transform:
        # Try with different formatting
        alt_name = search_name.replace('+', '')
        for exp in results:
            if exp['name'] == alt_name:
                return exp
    
    return None


def calculate_domain_shift(results, spec_transform='Log', suffix=''):
    """
    Calculate domain shift metrics for a given spec transform.
    
    Returns dict with metrics for both test domains.
    """
    # Find the four relevant experiments
    avianz_model = find_experiment(results, 'avianz', spec_transform, suffix)
    doc_model = find_experiment(results, 'doc', spec_transform, suffix)
    
    if not avianz_model or not doc_model:
        return None
    
    # Extract accuracies
    # AviaNZ model sees: avianz train, tests on both
    avianz_on_avianz = avianz_model['test1_acc']  # test1 is avianz_split_test
    avianz_on_doc = avianz_model['test2_acc']     # test2 is doc_split_test
    
    # DOC model sees: doc train, tests on both  
    doc_on_doc = doc_model['test1_acc']           # test1 is doc_split_test
    doc_on_avianz = doc_model['test2_acc']        # test2 is avianz_split_test
    
    # Calculate percentage reduction for each test domain
    # DOC test set: compare doc_on_doc (in-domain) vs avianz_on_doc (cross-domain)
    doc_test_reduction = ((doc_on_doc - avianz_on_doc) / doc_on_doc) * 100
    
    # AviaNZ test set: compare avianz_on_avianz (in-domain) vs doc_on_avianz (cross-domain)
    avianz_test_reduction = ((avianz_on_avianz - doc_on_avianz) / avianz_on_avianz) * 100
    
    return {
        'spec_transform': spec_transform + suffix,
        'doc_test': {
            'in_domain_acc': doc_on_doc,
            'cross_domain_acc': avianz_on_doc,
            'reduction_pct': doc_test_reduction,
            'in_domain_model': 'DOC',
            'cross_domain_model': 'AviaNZ'
        },
        'avianz_test': {
            'in_domain_acc': avianz_on_avianz,
            'cross_domain_acc': doc_on_avianz,
            'reduction_pct': avianz_test_reduction,
            'in_domain_model': 'AviaNZ',
            'cross_domain_model': 'DOC'
        }
    }


def print_metrics(metrics):
    """Pretty print domain shift metrics."""
    if not metrics:
        print("No metrics available")
        return
    
    print(f"\n{'='*70}")
    print(f"Domain Shift Analysis: {metrics['spec_transform']}")
    print(f"{'='*70}")
    
    print(f"\n📊 DOC Test Set:")
    print(f"  In-domain (DOC→DOC):     {metrics['doc_test']['in_domain_acc']:.1f}%")
    print(f"  Cross-domain (AviaNZ→DOC): {metrics['doc_test']['cross_domain_acc']:.1f}%")
    print(f"  ⚠️  Reduction from lack of domain exposure: {metrics['doc_test']['reduction_pct']:.1f}%")
    
    print(f"\n📊 AviaNZ Test Set:")
    print(f"  In-domain (AviaNZ→AviaNZ): {metrics['avianz_test']['in_domain_acc']:.1f}%")
    print(f"  Cross-domain (DOC→AviaNZ):   {metrics['avianz_test']['cross_domain_acc']:.1f}%")
    print(f"  ⚠️  Reduction from lack of domain exposure: {metrics['avianz_test']['reduction_pct']:.1f}%")
    
    print(f"\n🔍 Interpretation:")
    ratio = metrics['avianz_test']['reduction_pct'] / metrics['doc_test']['reduction_pct']
    print(f"  Domain shift severity ratio: {ratio:.2f}×")
    print(f"  (Deploying to AviaNZ is {ratio:.1f}× harder than deploying to DOC)")


def main():
    results = load_results()
    
    # Calculate for main normalization methods
    transforms = [
        'Log',
        'Log+normalize',
        'Log+normalize-no-median', 
        'Log+median-only',
        'PCEN',
        'Box-Cox',
        'dann_Log+normalize'  # Add DANN
    ]
    
    print("DOMAIN SHIFT METRICS")
    print("="*70)
    print("Comparing models WITH vs WITHOUT domain exposure on same test set")
    print("Controls for intrinsic dataset difficulty")
    
    for transform in transforms:
        metrics = calculate_domain_shift(results, transform)
        print_metrics(metrics)
    
    # Summary table - Reduction metrics
    print("\n" + "="*70)
    print("SUMMARY: PERCENTAGE REDUCTION FROM LACK OF DOMAIN EXPOSURE")
    print("="*70)
    print(f"\n{'Method':<25} {'DOC Test Reduction':<20} {'AviaNZ Test Reduction':<20} {'Asymmetry'}")
    print("-"*85)
    
    for transform in transforms:
        metrics = calculate_domain_shift(results, transform)
        if metrics:
            ratio = metrics['avianz_test']['reduction_pct'] / metrics['doc_test']['reduction_pct']
            print(f"{transform:<25} {metrics['doc_test']['reduction_pct']:>6.1f}% "
                  f"{metrics['avianz_test']['reduction_pct']:>18.1f}% "
                  f"{ratio:>18.2f}×")
    
    # Absolute cross-domain performance table
    print("\n" + "="*70)
    print("SUMMARY: ABSOLUTE CROSS-DOMAIN PERFORMANCE (UNSEEN DATA)")
    print("="*70)
    print("Which model performs best when deployed to unseen domains?")
    print(f"\n{'Method':<25} {'AviaNZ→DOC':<15} {'DOC→AviaNZ':<15} {'Best Model→DOC':<20} {'Best Model→AviaNZ'}")
    print("-"*100)
    
    best_on_doc = None
    best_on_avianz = None
    
    for transform in transforms:
        metrics = calculate_domain_shift(results, transform)
        if metrics:
            avianz_to_doc = metrics['doc_test']['cross_domain_acc']
            doc_to_avianz = metrics['avianz_test']['cross_domain_acc']
            
            # Track best
            if best_on_doc is None or avianz_to_doc > best_on_doc[1]:
                best_on_doc = (transform, avianz_to_doc)
            if best_on_avianz is None or doc_to_avianz > best_on_avianz[1]:
                best_on_avianz = (transform, doc_to_avianz)
            
            # Determine which model is better for each test domain
            better_for_doc = "AviaNZ" if avianz_to_doc > metrics['doc_test']['in_domain_acc'] else "DOC (in-domain)"
            better_for_avianz = "DOC" if doc_to_avianz > metrics['avianz_test']['in_domain_acc'] else "AviaNZ (in-domain)"
            
            print(f"{transform:<25} {avianz_to_doc:>6.1f}% {doc_to_avianz:>18.1f}% "
                  f"{better_for_doc:<26} {better_for_avianz}")
    
    print("\n" + "="*70)
    print("BEST CROSS-DOMAIN PERFORMANCE")
    print("="*70)
    if best_on_doc:
        print(f"Deploy to DOC (unseen):    Use {best_on_doc[0]:<20} → {best_on_doc[1]:.1f}%")
    if best_on_avianz:
        print(f"Deploy to AviaNZ (unseen): Use {best_on_avianz[0]:<20} → {best_on_avianz[1]:.1f}%")


if __name__ == '__main__':
    main()
