import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize hyperparameter search results")
    parser.add_argument('results_file', type=str, help="Path to search_results.json")
    parser.add_argument('--top-k', type=int, default=10, help="Show top K trials (default: 10)")
    args = parser.parse_args()
    
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    trials_df = pd.DataFrame(results['all_trials'])
    completed_trials = trials_df[trials_df['state'] == 'TrialState.COMPLETE'].copy()
    
    if len(completed_trials) == 0:
        print("No completed trials found!")
        return
    
    completed_trials = completed_trials.sort_values('value')
    
    print("="*80)
    print(f"Hyperparameter Search Results")
    print("="*80)
    print(f"Total trials: {len(trials_df)}")
    print(f"Completed: {len(completed_trials)}")
    print(f"Pruned: {len(trials_df[trials_df['state'] == 'TrialState.PRUNED'])}")
    print(f"Failed: {len(trials_df[trials_df['state'] == 'TrialState.FAIL'])}")
    print()
    
    print(f"Best Trial: #{results['best_trial']}")
    print(f"Best Validation Loss: {results['best_value']:.4f}")
    print()
    print("Best Parameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")
    print()
    
    print(f"\nTop {args.top_k} Trials:")
    print("-"*80)
    for idx, (_, row) in enumerate(completed_trials.head(args.top_k).iterrows(), 1):
        print(f"\n{idx}. Trial #{row['number']} - Val Loss: {row['value']:.4f}")
        params = row['params']
        print(f"   mixup_alpha={params['mixup_alpha']:.3f}, dropout={params['dropout']:.3f}, "
              f"weight_decay={params['weight_decay']:.6f}")
        print(f"   bce_smoothing={params['bce_smoothing']:.3f}, num_sparse_patches={params['num_sparse_patches']}, "
              f"lr={params['learning_rate']:.6f}")
    
    param_names = list(results['best_params'].keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        x = [trial['params'][param_name] for trial in results['all_trials'] 
             if trial['state'] == 'TrialState.COMPLETE']
        y = [trial['value'] for trial in results['all_trials'] 
             if trial['state'] == 'TrialState.COMPLETE']
        
        ax.scatter(x, y, alpha=0.6)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Validation Loss')
        ax.set_title(f'{param_name} vs Val Loss')
        ax.grid(True, alpha=0.3)
        
        if param_name in ['weight_decay', 'learning_rate']:
            ax.set_xscale('log')
    
    plt.tight_layout()
    
    output_dir = os.path.dirname(args.results_file)
    plot_path = os.path.join(output_dir, 'search_visualization.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {plot_path}")
    
    csv_path = os.path.join(output_dir, 'search_results.csv')
    completed_trials.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")

if __name__ == "__main__":
    main()
