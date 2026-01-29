"""
Dataset Distribution Analysis Script
Analyzes ground truth labels to understand class imbalance and dataset composition
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
import os


def load_bird_naming_map(csv_path):
    """Load the DOC bird naming map CSV"""
    df = pd.read_csv(csv_path)
    name_to_common = {}
    
    for _, row in df.iterrows():
        common_name = row['CommonName']
        if pd.notna(row['eBird']):
            name_to_common[row['eBird']] = common_name
        if pd.notna(row['ExtraName']):
            name_to_common[row['ExtraName']] = common_name
        if pd.notna(row['ListDOCBirds']):
            name_to_common[row['ListDOCBirds']] = common_name
        if pd.notna(row['ScientificName']):
            name_to_common[row['ScientificName']] = common_name
        name_to_common[common_name] = common_name
    
    name_to_common['Ruru'] = 'Morepork'
    name_to_common['Bellbird/Tui'] = 'Bellbird'
    
    return name_to_common


def normalize_to_common_name(name, name_mapping):
    """Normalize any species name to CommonName"""
    if name in ['Empty Sample', 'Tree Weta', 'Spy Bird', None, '']:
        return None
    
    if name in name_mapping:
        return name_mapping[name]
    
    name_lower = name.lower()
    for key, value in name_mapping.items():
        if key.lower() == name_lower:
            return value
    
    if '(' in name:
        base_name = name.split('(')[0].strip()
        if base_name in name_mapping:
            return name_mapping[base_name]
    
    return None


def analyze_dataset(labels_path, naming_map_path):
    """Comprehensive dataset analysis"""
    
    print("="*80)
    print("DATASET DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load naming map
    print("\nLoading bird naming map...")
    name_mapping = load_bird_naming_map(naming_map_path)
    
    # Load ground truth
    print("Loading ground truth labels...")
    with open(labels_path, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data['files'])
    print(f"\nTotal samples: {total_samples:,}")
    
    # Analyze sample composition
    samples_with_birds = 0
    samples_without_birds = 0
    all_species_raw = []
    all_species_normalized = []
    labels_per_sample = []
    
    species_counter_raw = Counter()
    species_counter_normalized = Counter()
    
    for file_info in data['files']:
        class_names = file_info.get('class_names', [])
        
        # Filter out non-bird labels
        bird_species = [s for s in class_names if s not in ['Empty Sample', 'Tree Weta', 'Spy Bird']]
        
        if bird_species:
            samples_with_birds += 1
            labels_per_sample.append(len(bird_species))
            
            # Count raw species
            for species in bird_species:
                species_counter_raw[species] += 1
                all_species_raw.append(species)
                
                # Normalize and count
                normalized = normalize_to_common_name(species, name_mapping)
                if normalized:
                    species_counter_normalized[normalized] += 1
                    all_species_normalized.append(normalized)
        else:
            samples_without_birds += 1
    
    # Basic statistics
    print("\n" + "="*80)
    print("SAMPLE COMPOSITION")
    print("="*80)
    print(f"Samples WITH birds:    {samples_with_birds:>8,} ({samples_with_birds/total_samples*100:>6.2f}%)")
    print(f"Samples WITHOUT birds: {samples_without_birds:>8,} ({samples_without_birds/total_samples*100:>6.2f}%)")
    print(f"\nClass imbalance ratio: {samples_without_birds/samples_with_birds:.2f}:1 (empty:bird)")
    
    # Multi-label statistics
    print("\n" + "="*80)
    print("MULTI-LABEL STATISTICS (for samples with birds)")
    print("="*80)
    if labels_per_sample:
        print(f"Average labels per sample: {np.mean(labels_per_sample):.2f}")
        print(f"Median labels per sample:  {np.median(labels_per_sample):.2f}")
        print(f"Min labels per sample:     {np.min(labels_per_sample)}")
        print(f"Max labels per sample:     {np.max(labels_per_sample)}")
        print(f"\nLabel distribution:")
        label_dist = Counter(labels_per_sample)
        for num_labels in sorted(label_dist.keys()):
            count = label_dist[num_labels]
            print(f"  {num_labels} label(s): {count:>6,} samples ({count/samples_with_birds*100:>5.2f}%)")
    
    # Species statistics (raw names)
    print("\n" + "="*80)
    print("SPECIES DISTRIBUTION (Raw Names)")
    print("="*80)
    print(f"Unique species: {len(species_counter_raw)}")
    print(f"Total bird occurrences: {len(all_species_raw):,}")
    print(f"\nTop 20 most common species:")
    for i, (species, count) in enumerate(species_counter_raw.most_common(20), 1):
        pct = count / len(all_species_raw) * 100
        print(f"{i:>2}. {species:<40} {count:>6,} ({pct:>5.2f}%)")
    
    # Species statistics (normalized names)
    print("\n" + "="*80)
    print("SPECIES DISTRIBUTION (Normalized to CommonName)")
    print("="*80)
    print(f"Unique species: {len(species_counter_normalized)}")
    print(f"Total bird occurrences: {len(all_species_normalized):,}")
    print(f"\nTop 20 most common species:")
    for i, (species, count) in enumerate(species_counter_normalized.most_common(20), 1):
        pct = count / len(all_species_normalized) * 100
        samples_pct = count / total_samples * 100
        print(f"{i:>2}. {species:<40} {count:>6,} ({pct:>5.2f}% of occurrences, {samples_pct:>5.2f}% of samples)")
    
    # Rare species analysis
    print("\n" + "="*80)
    print("RARE SPECIES ANALYSIS (Normalized)")
    print("="*80)
    rare_threshold = 10
    rare_species = [s for s, c in species_counter_normalized.items() if c <= rare_threshold]
    print(f"Species with ≤{rare_threshold} occurrences: {len(rare_species)}")
    if rare_species:
        print(f"Rare species list:")
        for species in sorted(rare_species):
            count = species_counter_normalized[species]
            print(f"  - {species:<40} ({count} occurrences)")
    
    # Calculate metrics for perfect "always predict no birds" baseline
    print("\n" + "="*80)
    print("BASELINE PREDICTOR ANALYSIS")
    print("="*80)
    print("If a model predicts 'No Birds' for every sample:")
    baseline_accuracy = samples_without_birds / total_samples
    print(f"  Accuracy:  {baseline_accuracy*100:.2f}%")
    print(f"  Precision: 0.00% (no positive predictions)")
    print(f"  Recall:    0.00% (no birds detected)")
    print(f"  F1 Score:  0.00%")
    print(f"\nThis explains why 'Ast Test 72' has ~72.5% accuracy with all-zero predictions!")
    
    # Save summary statistics
    output_dir = os.path.dirname(labels_path)
    summary_path = os.path.join(output_dir, 'dataset_distribution_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("DATASET DISTRIBUTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples: {total_samples:,}\n")
        f.write(f"Samples WITH birds:    {samples_with_birds:,} ({samples_with_birds/total_samples*100:.2f}%)\n")
        f.write(f"Samples WITHOUT birds: {samples_without_birds:,} ({samples_without_birds/total_samples*100:.2f}%)\n")
        f.write(f"Class imbalance ratio: {samples_without_birds/samples_with_birds:.2f}:1\n\n")
        f.write(f"Unique species (normalized): {len(species_counter_normalized)}\n")
        f.write(f"Total bird occurrences: {len(all_species_normalized):,}\n\n")
        f.write("Top species:\n")
        for species, count in species_counter_normalized.most_common(20):
            pct = count / len(all_species_normalized) * 100
            f.write(f"  {species:<40} {count:>6,} ({pct:>5.2f}%)\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Create visualizations
    create_visualizations(
        samples_with_birds, samples_without_birds,
        species_counter_normalized, labels_per_sample,
        output_dir
    )
    
    return {
        'total_samples': total_samples,
        'samples_with_birds': samples_with_birds,
        'samples_without_birds': samples_without_birds,
        'species_counter': species_counter_normalized,
        'labels_per_sample': labels_per_sample
    }


def create_visualizations(samples_with_birds, samples_without_birds, 
                         species_counter, labels_per_sample, output_dir):
    """Create visualization plots"""
    
    print("\nGenerating visualizations...")
    
    # 1. Sample composition pie chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Pie chart
    ax = axes[0, 0]
    sizes = [samples_with_birds, samples_without_birds]
    labels = [f'With Birds\n({samples_with_birds:,})', 
              f'No Birds\n({samples_without_birds:,})']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax.set_title('Sample Distribution: Birds vs No Birds', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 2. Top species bar chart
    ax = axes[0, 1]
    top_n = 15
    top_species = species_counter.most_common(top_n)
    species_names = [s[0] for s in top_species]
    counts = [s[1] for s in top_species]
    
    y_pos = np.arange(len(species_names))
    ax.barh(y_pos, counts, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(species_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Occurrences', fontweight='bold')
    ax.set_title(f'Top {top_n} Most Common Species', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Labels per sample distribution
    if labels_per_sample:
        ax = axes[1, 0]
        label_dist = Counter(labels_per_sample)
        nums = sorted(label_dist.keys())
        counts = [label_dist[n] for n in nums]
        
        ax.bar(nums, counts, color='coral', edgecolor='black')
        ax.set_xlabel('Number of Labels per Sample', fontweight='bold')
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Distribution of Labels per Sample (Bird Samples Only)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
    
    # 4. Species frequency distribution
    ax = axes[1, 1]
    all_counts = sorted(species_counter.values(), reverse=True)
    ax.plot(range(1, len(all_counts) + 1), all_counts, marker='o', 
            linewidth=2, markersize=4, color='darkgreen')
    ax.set_xlabel('Species Rank', fontweight='bold')
    ax.set_ylabel('Number of Occurrences', fontweight='bold')
    ax.set_title('Species Frequency Distribution (Long Tail)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'dataset_distribution.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {viz_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ground truth dataset distribution and class imbalance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python analyze_dataset_distribution.py MoreporkResults\n'
               '  python analyze_dataset_distribution.py Joe_MO_Results'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Folder containing labels.json'
    )
    
    args = parser.parse_args()
    
    base_dir = '/home/giotto/Desktop/AviaNZ'
    results_folder = os.path.join(base_dir, args.folder)
    labels_path = os.path.join(results_folder, 'labels.json')
    naming_map_path = os.path.join(base_dir, 'DOC_bird_naming_map.csv')
    
    if not os.path.exists(labels_path):
        print(f"Error: labels.json not found in {results_folder}")
        return
    
    if not os.path.exists(naming_map_path):
        print(f"Error: DOC_bird_naming_map.csv not found in {base_dir}")
        return
    
    analyze_dataset(labels_path, naming_map_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
