import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

def analyze_spectrograms(data_dir, name, num_samples=100):
    """Analyze actual spectrogram content."""
    data_path = Path(data_dir) / "data"
    labels_path = Path(data_dir) / "labels.json"
    
    with open(labels_path) as f:
        labels_data = json.load(f)
    
    # Group by species
    species_files = {}
    for item in labels_data['files']:
        species = item['primary_class']
        if species not in species_files:
            species_files[species] = []
        species_files[species].append(data_path / item['filename'])
    
    print(f"\n{name}:")
    print(f"  Species: {len(species_files)}")
    
    # Analyze spectrograms per species
    species_stats = {}
    all_specs = []
    
    for species, files in species_files.items():
        specs = []
        for f in files[:num_samples]:
            if f.exists():
                spec = np.load(f)
                specs.append(spec)
                all_specs.append(spec)
        
        if specs:
            species_stats[species] = {
                'count': len(specs),
                'mean_energy': [s.mean() for s in specs],
                'max_energy': [s.max() for s in specs],
                'std_energy': [s.std() for s in specs],
                'shapes': [s.shape for s in specs],
                'freq_profiles': [s.mean(axis=1) for s in specs],
                'time_profiles': [s.mean(axis=0) for s in specs],
            }
    
    # Compute aggregate statistics
    all_means = []
    all_maxs = []
    all_stds = []
    all_freq_ranges = []
    all_time_ranges = []
    
    for species, stats in species_stats.items():
        all_means.extend(stats['mean_energy'])
        all_maxs.extend(stats['max_energy'])
        all_stds.extend(stats['std_energy'])
        
        # Analyze frequency distribution
        for freq_profile in stats['freq_profiles']:
            # Find frequency bins with significant energy
            threshold = freq_profile.max() * 0.1
            active_bins = np.where(freq_profile > threshold)[0]
            if len(active_bins) > 0:
                all_freq_ranges.append((active_bins.min(), active_bins.max()))
        
        # Analyze temporal distribution
        for time_profile in stats['time_profiles']:
            threshold = time_profile.max() * 0.1
            active_bins = np.where(time_profile > threshold)[0]
            if len(active_bins) > 0:
                all_time_ranges.append((active_bins.min(), active_bins.max()))
    
    print(f"  Total spectrograms analyzed: {len(all_specs)}")
    print(f"  Mean energy: {np.mean(all_means):.2f} ± {np.std(all_means):.2f}")
    print(f"  Max energy: {np.mean(all_maxs):.2f} ± {np.std(all_maxs):.2f}")
    print(f"  Std energy: {np.mean(all_stds):.2f} ± {np.std(all_stds):.2f}")
    
    if all_freq_ranges:
        freq_mins = [r[0] for r in all_freq_ranges]
        freq_maxs = [r[1] for r in all_freq_ranges]
        print(f"  Frequency range (bins): {np.mean(freq_mins):.1f}-{np.mean(freq_maxs):.1f}")
        print(f"    (Hz @ 32kHz/224bins): {np.mean(freq_mins)*16000/224:.0f}-{np.mean(freq_maxs)*16000/224:.0f} Hz")
    
    if all_time_ranges:
        time_mins = [r[0] for r in all_time_ranges]
        time_maxs = [r[1] for r in all_time_ranges]
        print(f"  Temporal spread: {np.mean(time_mins):.1f}-{np.mean(time_maxs):.1f} bins")
        print(f"    Duration: {100*(np.mean(time_maxs)-np.mean(time_mins))/np.mean([s.shape[1] for s in all_specs]):.1f}% of clip")
    
    return species_stats, all_specs


def compare_per_species(doc_stats, joe_stats):
    """Compare same species across datasets."""
    print("\n" + "="*80)
    print("PER-SPECIES COMPARISON (same species, different datasets):")
    print("="*80)
    
    common_species = set(doc_stats.keys()) & set(joe_stats.keys())
    print(f"\nCommon species: {len(common_species)}")
    
    for species in sorted(common_species):
        doc = doc_stats[species]
        joe = joe_stats[species]
        
        doc_mean = np.mean(doc['mean_energy'])
        joe_mean = np.mean(joe['mean_energy'])
        
        doc_freq = np.mean([fp for fp in doc['freq_profiles']], axis=0)
        joe_freq = np.mean([fp for fp in joe['freq_profiles']], axis=0)
        
        # Correlation between frequency profiles
        freq_corr = np.corrcoef(doc_freq, joe_freq)[0, 1]
        
        print(f"\n{species}:")
        print(f"  DOC: {doc['count']} samples, mean={doc_mean:.2f}")
        print(f"  Joe: {joe['count']} samples, mean={joe_mean:.2f}")
        print(f"  Energy ratio (Joe/DOC): {joe_mean/doc_mean:.2f}x")
        print(f"  Frequency profile correlation: {freq_corr:.3f}")
        
        if freq_corr < 0.7:
            print(f"  ⚠️  LOW CORRELATION - different frequency content!")


def plot_comparison(doc_specs, joe_specs, output_path):
    """Plot sample spectrograms side by side."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Sample Spectrograms: DOC vs Joe_Mo (same indices)', fontsize=14)
    
    for i in range(4):
        # DOC
        ax = axes[i, 0]
        doc_spec = doc_specs[i]
        im = ax.imshow(doc_spec, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'DOC #{i}\nmean={doc_spec.mean():.2f}')
        plt.colorbar(im, ax=ax)
        
        # DOC frequency profile
        ax = axes[i, 1]
        freq_profile = doc_spec.mean(axis=1)
        ax.plot(freq_profile)
        ax.set_title('DOC Freq Profile')
        ax.set_xlabel('Frequency bin')
        
        # Joe_Mo
        ax = axes[i, 2]
        joe_spec = joe_specs[i]
        im = ax.imshow(joe_spec, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'Joe_Mo #{i}\nmean={joe_spec.mean():.2f}')
        plt.colorbar(im, ax=ax)
        
        # Joe_Mo frequency profile
        ax = axes[i, 3]
        freq_profile = joe_spec.mean(axis=1)
        ax.plot(freq_profile)
        ax.set_title('Joe_Mo Freq Profile')
        ax.set_xlabel('Frequency bin')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot: {output_path}")


def main():
    base_dir = Path("test")
    
    print("="*80)
    print("DETAILED SPECTROGRAM INVESTIGATION")
    print("="*80)
    
    # Analyze both datasets
    doc_stats, doc_specs = analyze_spectrograms(
        base_dir / "doc_split" / "train",
        "DOC Dataset"
    )
    
    joe_stats, joe_specs = analyze_spectrograms(
        base_dir / "joe_mo_split" / "train",
        "Joe_Mo Dataset"
    )
    
    # Compare per species
    compare_per_species(doc_stats, joe_stats)
    
    # Plot samples
    output_dir = base_dir / "diagnostics"
    output_dir.mkdir(exist_ok=True)
    plot_comparison(doc_specs[:4], joe_specs[:4], output_dir / "detailed_spectrogram_comparison.png")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
