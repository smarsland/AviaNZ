#!/usr/bin/env python3
"""
Visualize cross-domain experiment results as heatmaps
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def parse_experiment_name(name):
    """Extract training source, method, and parameters from experiment name"""
    parts = name.split('_')
    
    # Training source
    train_source = parts[0]  # 'avianz' or 'doc'
    
    # Method type
    if 'dann' in name:
        method = 'DANN'
        norm = parts[-1]  # usually 'Log+normalize'
    elif 'baseline' in name:
        method_parts = parts[2:]
        if 'intensity' in name:
            # Noise augmentation
            intensity = name.split('intensity')[1]
            method = f'Log+noise_{intensity}'
            norm = 'noise_aug'
        elif 'variety' in name:
            variety = name.split('variety')[1]
            method = f'Log+variety_{variety}'
            norm = 'noise_aug'
        else:
            # Normalization method
            norm = '_'.join(method_parts)
            method = norm
    else:
        method = name
        norm = name
    
    return train_source, method, norm

def create_main_results_heatmap(results):
    """Create heatmap for main normalization methods"""
    
    # Filter main experiments (baseline normalizations + DANN)
    main_methods = [
        'Log', 'Log+median-only', 'Log+normalize-no-median', 
        'Log+normalize', 'PCEN', 'Box-Cox'
    ]
    
    # Organize data
    avianz_data = {}
    doc_data = {}
    
    for result in results:
        name = result['name']
        train_source, method, norm = parse_experiment_name(name)
        
        # Include baseline normalizations and DANN
        if any(m in norm for m in main_methods) or 'dann' in name:
            
            if 'dann' in name:
                display_name = 'DANN+Log+norm'
            else:
                display_name = norm
            
            if train_source == 'avianz':
                # test1 is avianz (within-domain), test2 is doc (cross-domain)
                avianz_data[display_name] = {
                    'within': result['test1_acc'],
                    'cross': result['test2_acc'],
                    'shift': result['test2_acc'] - result['test1_acc']
                }
            else:  # doc
                # test1 is doc (within-domain), test2 is avianz (cross-domain)
                doc_data[display_name] = {
                    'within': result['test1_acc'],
                    'cross': result['test2_acc'],
                    'shift': result['test2_acc'] - result['test1_acc']
                }
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cross-Domain Performance Analysis', fontsize=16, fontweight='bold')
    
    # Method ordering for consistency
    method_order = ['Log', 'Log+median-only', 'Log+normalize-no-median', 
                    'Log+normalize', 'PCEN', 'Box-Cox', 'DANN+Log+norm']
    
    # Filter to existing methods
    avianz_methods = [m for m in method_order if m in avianz_data]
    doc_methods = [m for m in method_order if m in doc_data]
    
    # Prepare data matrices
    avianz_within = [avianz_data[m]['within'] for m in avianz_methods]
    avianz_cross = [avianz_data[m]['cross'] for m in avianz_methods]
    avianz_shift = [avianz_data[m]['shift'] for m in avianz_methods]
    
    doc_within = [doc_data[m]['within'] for m in doc_methods]
    doc_cross = [doc_data[m]['cross'] for m in doc_methods]
    doc_shift = [doc_data[m]['shift'] for m in doc_methods]
    
    # Plot 1: AviaNZ training - Within-domain accuracy
    ax = axes[0, 0]
    sns.heatmap(np.array(avianz_within).reshape(-1, 1), 
                annot=True, fmt='.1f', cmap='YlGn', 
                yticklabels=avianz_methods, xticklabels=['Accuracy'],
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('AviaNZ Training\nWithin-Domain (AviaNZ test)', fontweight='bold')
    ax.set_ylabel('')
    
    # Plot 2: AviaNZ training - Cross-domain accuracy
    ax = axes[0, 1]
    sns.heatmap(np.array(avianz_cross).reshape(-1, 1), 
                annot=True, fmt='.1f', cmap='YlOrRd', 
                yticklabels=avianz_methods, xticklabels=['Accuracy'],
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('AviaNZ Training\nCross-Domain (DOC test)', fontweight='bold')
    ax.set_ylabel('')
    
    # Plot 3: AviaNZ training - Domain shift
    ax = axes[0, 2]
    sns.heatmap(np.array(avianz_shift).reshape(-1, 1), 
                annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                yticklabels=avianz_methods, xticklabels=['Shift'],
                vmin=-20, vmax=5, ax=ax, cbar_kws={'label': 'Shift (pp)'})
    ax.set_title('AviaNZ Training\nDomain Shift (Cross - Within)', fontweight='bold')
    ax.set_ylabel('')
    
    # Plot 4: DOC training - Within-domain accuracy
    ax = axes[1, 0]
    sns.heatmap(np.array(doc_within).reshape(-1, 1), 
                annot=True, fmt='.1f', cmap='YlGn', 
                yticklabels=doc_methods, xticklabels=['Accuracy'],
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('DOC Training\nWithin-Domain (DOC test)', fontweight='bold')
    ax.set_ylabel('')
    
    # Plot 5: DOC training - Cross-domain accuracy
    ax = axes[1, 1]
    sns.heatmap(np.array(doc_cross).reshape(-1, 1), 
                annot=True, fmt='.1f', cmap='YlOrRd', 
                yticklabels=doc_methods, xticklabels=['Accuracy'],
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('DOC Training\nCross-Domain (AviaNZ test)', fontweight='bold')
    ax.set_ylabel('')
    
    # Plot 6: DOC training - Domain shift
    ax = axes[1, 2]
    sns.heatmap(np.array(doc_shift).reshape(-1, 1), 
                annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                yticklabels=doc_methods, xticklabels=['Shift'],
                vmin=-45, vmax=5, ax=ax, cbar_kws={'label': 'Shift (pp)'})
    ax.set_title('DOC Training\nDomain Shift (Cross - Within)', fontweight='bold')
    ax.set_ylabel('')
    
    plt.tight_layout()
    return fig

def create_noise_augmentation_heatmap(results):
    """Create heatmap for noise augmentation experiments"""
    
    # Filter noise augmentation experiments
    intensity_data = {'avianz': {}, 'doc': {}}
    variety_data = {'avianz': {}, 'doc': {}}
    
    for result in results:
        name = result['name']
        
        if 'intensity' in name:
            train_source = name.split('_')[0]
            intensity = name.split('intensity')[1]
            
            if train_source == 'avianz':
                within = result['test1_acc']
                cross = result['test2_acc']
            else:
                within = result['test1_acc']
                cross = result['test2_acc']
            
            intensity_data[train_source][intensity] = {
                'within': within,
                'cross': cross,
                'shift': cross - within
            }
        
        elif 'variety' in name and 'intensity' not in name:
            train_source = name.split('_')[0]
            variety = name.split('variety')[1]
            
            if train_source == 'avianz':
                within = result['test1_acc']
                cross = result['test2_acc']
            else:
                within = result['test1_acc']
                cross = result['test2_acc']
            
            variety_data[train_source][variety] = {
                'within': within,
                'cross': cross,
                'shift': cross - within
            }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Noise Augmentation Analysis', fontsize=16, fontweight='bold')
    
    # Intensity experiments
    if intensity_data['avianz'] or intensity_data['doc']:
        intensities = sorted(list(set(list(intensity_data['avianz'].keys()) + 
                                     list(intensity_data['doc'].keys()))))
        
        # AviaNZ intensity
        ax = axes[0, 0]
        avianz_int_within = [intensity_data['avianz'].get(i, {}).get('within', 0) for i in intensities]
        avianz_int_cross = [intensity_data['avianz'].get(i, {}).get('cross', 0) for i in intensities]
        
        data_matrix = np.array([avianz_int_within, avianz_int_cross])
        sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=intensities, yticklabels=['Within-Domain', 'Cross-Domain'],
                    vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
        ax.set_title('AviaNZ Training\nNoise Intensity Experiments', fontweight='bold')
        ax.set_xlabel('Noise Intensity')
        
        # DOC intensity
        ax = axes[0, 1]
        doc_int_within = [intensity_data['doc'].get(i, {}).get('within', 0) for i in intensities]
        doc_int_cross = [intensity_data['doc'].get(i, {}).get('cross', 0) for i in intensities]
        
        data_matrix = np.array([doc_int_within, doc_int_cross])
        sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=intensities, yticklabels=['Within-Domain', 'Cross-Domain'],
                    vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
        ax.set_title('DOC Training\nNoise Intensity Experiments', fontweight='bold')
        ax.set_xlabel('Noise Intensity')
    
    # Variety experiments
    if variety_data['avianz'] or variety_data['doc']:
        varieties = sorted(list(set(list(variety_data['avianz'].keys()) + 
                                   list(variety_data['doc'].keys()))))
        
        # AviaNZ variety
        ax = axes[1, 0]
        avianz_var_within = [variety_data['avianz'].get(v, {}).get('within', 0) for v in varieties]
        avianz_var_cross = [variety_data['avianz'].get(v, {}).get('cross', 0) for v in varieties]
        
        data_matrix = np.array([avianz_var_within, avianz_var_cross])
        sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=varieties, yticklabels=['Within-Domain', 'Cross-Domain'],
                    vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
        ax.set_title('AviaNZ Training\nNoise Variety Experiments', fontweight='bold')
        ax.set_xlabel('Number of Noise Sources')
        
        # DOC variety
        ax = axes[1, 1]
        doc_var_within = [variety_data['doc'].get(v, {}).get('within', 0) for v in varieties]
        doc_var_cross = [variety_data['doc'].get(v, {}).get('cross', 0) for v in varieties]
        
        data_matrix = np.array([doc_var_within, doc_var_cross])
        sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                    xticklabels=varieties, yticklabels=['Within-Domain', 'Cross-Domain'],
                    vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
        ax.set_title('DOC Training\nNoise Variety Experiments', fontweight='bold')
        ax.set_xlabel('Number of Noise Sources')
    
    plt.tight_layout()
    return fig

def create_comparison_matrix_heatmap(results):
    """Create a matrix comparing all methods across both training sources"""
    
    # Organize data
    methods = {}
    
    for result in results:
        name = result['name']
        train_source, method, norm = parse_experiment_name(name)
        
        # Skip noise augmentation for this view
        if 'intensity' in name or 'variety' in name:
            continue
        
        if 'dann' in name:
            display_name = 'DANN+Log+norm'
        else:
            display_name = norm
        
        if display_name not in methods:
            methods[display_name] = {
                'avianz_within': None,
                'avianz_cross': None,
                'doc_within': None,
                'doc_cross': None
            }
        
        if train_source == 'avianz':
            methods[display_name]['avianz_within'] = result['test1_acc']
            methods[display_name]['avianz_cross'] = result['test2_acc']
        else:  # doc
            methods[display_name]['doc_within'] = result['test1_acc']
            methods[display_name]['doc_cross'] = result['test2_acc']
    
    # Filter to complete methods
    complete_methods = {k: v for k, v in methods.items() 
                       if all(v.values())}
    
    if not complete_methods:
        print("No complete methods found for comparison matrix")
        return None
    
    # Create matrix
    method_order = ['Log', 'Log+median-only', 'Log+normalize-no-median', 
                    'Log+normalize', 'PCEN', 'Box-Cox', 'DANN+Log+norm']
    method_names = [m for m in method_order if m in complete_methods]
    
    column_labels = ['AviaNZ→AviaNZ\n(within)', 'AviaNZ→DOC\n(cross)',
                     'DOC→DOC\n(within)', 'DOC→AviaNZ\n(cross)']
    
    data_matrix = []
    for method in method_names:
        row = [
            complete_methods[method]['avianz_within'],
            complete_methods[method]['avianz_cross'],
            complete_methods[method]['doc_within'],
            complete_methods[method]['doc_cross']
        ]
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=column_labels, yticklabels=method_names,
                vmin=0, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('Complete Method Comparison Across Domains', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('Method', fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    # Load results
    json_path = Path(__file__).parent / 'experiments_matched' / 'all_results.json'
    results = load_results(json_path)
    
    print(f"Loaded {len(results)} experiment results")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Generate heatmaps
    print("\nGenerating main results heatmap...")
    fig1 = create_main_results_heatmap(results)
    fig1.savefig(output_dir / 'results_heatmap_main.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / 'results_heatmap_main.png', dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_dir / 'results_heatmap_main.pdf'}")
    
    print("\nGenerating noise augmentation heatmap...")
    fig2 = create_noise_augmentation_heatmap(results)
    fig2.savefig(output_dir / 'results_heatmap_noise.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / 'results_heatmap_noise.png', dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_dir / 'results_heatmap_noise.pdf'}")
    
    print("\nGenerating comparison matrix heatmap...")
    fig3 = create_comparison_matrix_heatmap(results)
    if fig3:
        fig3.savefig(output_dir / 'results_heatmap_comparison.pdf', dpi=300, bbox_inches='tight')
        fig3.savefig(output_dir / 'results_heatmap_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved to {output_dir / 'results_heatmap_comparison.pdf'}")
    
    print("\nAll heatmaps generated successfully!")
    print(f"Output directory: {output_dir}")
    
    # Show plots
    plt.show()

if __name__ == '__main__':
    main()
