#!/usr/bin/env python3
"""
Analyze Optuna hyperparameter search results.
"""
import argparse
import optuna
import pandas as pd
import json

def analyze_study(storage_path, study_name='ast_sparse_search'):
    """Load and analyze an Optuna study."""
    
    storage = f"sqlite:///{storage_path}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"Study '{study_name}' not found in {storage_path}")
        print("\nAvailable studies:")
        studies = optuna.study.get_all_study_summaries(storage=storage)
        for s in studies:
            print(f"  - {s.study_name}")
        return
    
    print("="*80)
    print(f"Study: {study_name}")
    print("="*80)
    
    trials = study.trials
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
    running_trials = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    
    # Identify "pseudo-completed" trials - pruned but finished all epochs
    finished_pruned = []
    early_pruned = []
    for t in pruned_trials:
        if hasattr(t, 'intermediate_values') and t.intermediate_values:
            max_epoch = max(t.intermediate_values.keys())
            if max_epoch >= 29:  # Finished 30 epochs (0-indexed)
                finished_pruned.append(t)
            else:
                early_pruned.append(t)
        else:
            early_pruned.append(t)
    
    # Treat finished_pruned as completed for analysis
    all_finished = completed_trials + finished_pruned
    
    print(f"\nTotal trials: {len(trials)}")
    print(f"  Completed: {len(completed_trials)}")
    print(f"  Finished (marked as pruned): {len(finished_pruned)}")
    print(f"  Pruned early: {len(early_pruned)}")
    print(f"  Failed: {len(failed_trials)}")
    print(f"  Running: {len(running_trials)}")
    print(f"\nTrials with full results: {len(all_finished)}")
    
    if all_finished:
        # Get best val loss from intermediate values for finished_pruned trials
        def get_final_loss(t):
            if t.state == optuna.trial.TrialState.COMPLETE:
                return t.value
            elif hasattr(t, 'intermediate_values') and t.intermediate_values:
                return t.intermediate_values[max(t.intermediate_values.keys())]
            return float('inf')
        
        all_finished_sorted = sorted(all_finished, key=get_final_loss)
        best_trial = all_finished_sorted[0]
        
        print(f"\n{'='*80}")
        print("BEST TRIAL")
        print("="*80)
        print(f"Trial #: {best_trial.number}")
        print(f"Final Validation Loss: {get_final_loss(best_trial):.6f}")
        print("\nBest hyperparameters:")
        for key, value in sorted(best_trial.params.items()):
            print(f"  {key:25s}: {value}")
        
        print(f"\n{'='*80}")
        print("TOP 10 TRIALS")
        print("="*80)
        for i, trial in enumerate(all_finished_sorted[:10], 1):
            final_loss = get_final_loss(trial)
            print(f"\n{i}. Trial {trial.number}: val_loss = {final_loss:.6f}")
            key_params = ['learning_rate', 'mixup_alpha', 'dropout', 'weight_decay', 'use_focal_loss', 'normalize', 'scheduler_type']
            for key in key_params:
                if key in trial.params:
                    print(f"     {key:23s}: {trial.params[key]}")
        
        print(f"\n{'='*80}")
        print("PARAMETER ANALYSIS")
        print("="*80)
        
        df = pd.DataFrame([t.params for t in all_finished])
        df['val_loss'] = [get_final_loss(t) for t in all_finished]
        df['trial_num'] = [t.number for t in all_finished]
        
        for param in sorted(df.columns):
            if param in ['val_loss', 'trial_num']:
                continue
            print(f"\n{param}:")
            if df[param].dtype in ['float64', 'int64']:
                print(f"  Range: {df[param].min():.6f} to {df[param].max():.6f}")
                print(f"  Mean: {df[param].mean():.6f}")
                corr = df[param].corr(df['val_loss'])
                print(f"  Correlation with val_loss: {corr:+.3f} {'(lower is better)' if corr < 0 else '(higher is worse)' if corr > 0.1 else '(minimal effect)'}")
                # Show best values
                best_5 = df.nsmallest(5, 'val_loss')[param]
                print(f"  Best 5 trials used: {list(best_5.round(6))}")
            else:
                value_counts = df[param].value_counts()
                print(f"  Distribution: {dict(value_counts)}")
                print(f"  Avg val_loss by value:")
                for val in sorted(df[param].unique()):
                    subset = df[df[param] == val]
                    avg_loss = subset['val_loss'].mean()
                    min_loss = subset['val_loss'].min()
                    count = len(subset)
                    print(f"    {str(val):20s}: avg={avg_loss:.6f}, best={min_loss:.6f}, n={count}")
        
        print(f"\n{'='*80}")
        print("KEY INSIGHTS")
        print("="*80)
        
        # Find patterns in top trials
        top_5 = all_finished_sorted[:5]
        print(f"\nCommon settings in top 5 trials:")
        for param in ['normalize', 'use_focal_loss', 'use_class_weights', 'scheduler_type', 'use_multiscale']:
            if param in df.columns:
                values = [t.params.get(param) for t in top_5]
                from collections import Counter
                counts = Counter(values)
                most_common = counts.most_common(1)[0]
                print(f"  {param}: {most_common[0]} ({most_common[1]}/5 trials)")
        
        # Numeric parameter ranges in top 5
        print(f"\nParameter ranges in top 5 trials:")
        for param in ['learning_rate', 'mixup_alpha', 'dropout', 'weight_decay']:
            if param in df.columns:
                values = [t.params.get(param) for t in top_5]
                print(f"  {param}: {min(values):.6f} to {max(values):.6f}")
    
    else:
        print("\nNo finished trials yet to analyze.")
    
    if early_pruned:
        print(f"\n{'='*80}")
        print(f"EARLY PRUNED TRIALS ({len(early_pruned)} trials stopped before completion)")
        print("="*80)
        print("(Not shown - use --show-pruned flag to see these)")
    
    if failed_trials:
        print(f"\n{'='*80}")
        print("FAILED TRIALS")
        print("="*80)
        for trial in failed_trials:
            print(f"\nTrial {trial.number}:")
            print(f"  Params: {trial.params}")
            if trial.user_attrs:
                print(f"  User attrs: {trial.user_attrs}")
    
    if completed_trials:
        print(f"\n{'='*80}")
        print("BEST TRIAL")
        print("="*80)
        print(f"Trial #: {study.best_trial.number}")
        print(f"Validation Loss: {study.best_trial.value:.6f}")
        print("\nBest hyperparameters:")
        for key, value in sorted(study.best_trial.params.items()):
            print(f"  {key:25s}: {value}")
        
        print(f"\n{'='*80}")
        print("TOP 5 TRIALS")
        print("="*80)
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        for i, trial in enumerate(sorted_trials[:5], 1):
            print(f"\n{i}. Trial {trial.number}: loss = {trial.value:.6f}")
            for key, value in sorted(trial.params.items()):
                print(f"     {key:23s}: {value}")
        
        print(f"\n{'='*80}")
        print("PARAMETER STATISTICS")
        print("="*80)
        
        df = pd.DataFrame([t.params for t in completed_trials])
        df['val_loss'] = [t.value for t in completed_trials]
        
        for param in sorted(df.columns):
            if param == 'val_loss':
                continue
            print(f"\n{param}:")
            if df[param].dtype in ['float64', 'int64']:
                print(f"  Range: {df[param].min():.6f} to {df[param].max():.6f}")
                print(f"  Mean: {df[param].mean():.6f}")
                print(f"  Median: {df[param].median():.6f}")
                # Correlation with validation loss
                corr = df[param].corr(df['val_loss'])
                print(f"  Correlation with val_loss: {corr:.3f}")
            else:
                value_counts = df[param].value_counts()
                print(f"  Values: {dict(value_counts)}")
                # Show average loss for each value
                print(f"  Avg loss by value:")
                for val in df[param].unique():
                    avg_loss = df[df[param] == val]['val_loss'].mean()
                    count = len(df[df[param] == val])
                    print(f"    {val}: {avg_loss:.6f} (n={count})")
        
        print(f"\n{'='*80}")
        print("PROGRESS OVER TIME")
        print("="*80)
        print(f"Trial   Val Loss    Best So Far")
        print("-" * 40)
        best_so_far = float('inf')
        for trial in sorted(completed_trials, key=lambda t: t.number):
            if trial.value < best_so_far:
                best_so_far = trial.value
                marker = " *NEW BEST*"
            else:
                marker = ""
            print(f"{trial.number:5d}   {trial.value:.6f}    {best_so_far:.6f}{marker}")
    
    else:
        print("\nNo completed trials yet.")
    
    return study

def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna hyperparameter search results")
    parser.add_argument('database', type=str, help="Path to optuna_study.db file")
    parser.add_argument('--study-name', type=str, default='ast_sparse_search', help="Name of the study (default: ast_sparse_search)")
    parser.add_argument('--export-csv', type=str, default=None, help="Export results to CSV file")
    
    args = parser.parse_args()
    
    study = analyze_study(args.database, args.study_name)
    
    if args.export_csv and study:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            df = pd.DataFrame([t.params for t in completed])
            df['trial_number'] = [t.number for t in completed]
            df['val_loss'] = [t.value for t in completed]
            df = df[['trial_number', 'val_loss'] + [c for c in df.columns if c not in ['trial_number', 'val_loss']]]
            df.to_csv(args.export_csv, index=False)
            print(f"\nResults exported to: {args.export_csv}")

if __name__ == "__main__":
    main()
