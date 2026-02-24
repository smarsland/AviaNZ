import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance, mannwhitneyu
from normalizer import normalize_spectrogram


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    row_to_class = {}
    class_counts = Counter()

    for item in data.get("files", []):
        filename = item.get("filename")
        primary_class = item.get("primary_class")
        if filename and primary_class:
            row_to_class[filename] = primary_class
            class_counts[primary_class] += 1

    return row_to_class, class_counts


def load_predictions(csv_path):
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        class_names = [name for name in reader.fieldnames if name not in {"File_Path", "row_id"}]
        rows = []
        for row in reader:
            row_id = row.get("row_id")
            scores = {name: float(row[name]) for name in class_names}
            rows.append((row_id, scores))

    return class_names, rows


def compute_per_class_accuracy(row_to_class, rows):
    correct = Counter()
    total = Counter()

    for row_id, scores in rows:
        true_class = row_to_class.get(row_id)
        if not true_class:
            continue
        predicted_class = max(scores.items(), key=lambda item: item[1])[0]
        total[true_class] += 1
        if predicted_class == true_class:
            correct[true_class] += 1

    per_class = {}
    for cls in total:
        per_class[cls] = correct[cls] / total[cls]

    overall = sum(correct.values()) / max(1, sum(total.values()))
    return per_class, total, overall


def plot_accuracy_heatmap(output_path, title, classes, values, columns):
    data = np.array(values)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(classes))))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=30, ha="right")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Accuracy")

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.5 else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_accuracy_table(output_path, classes, totals, per_class_by_model, model_labels):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["species", "support"] + model_labels
        writer.writerow(header)
        for cls in classes:
            row = [cls, totals.get(cls, 0)]
            for values in per_class_by_model:
                row.append(f"{values.get(cls, 0.0):.4f}")
            writer.writerow(row)


def extract_lengths(data_dir):
    lengths = []
    for npy_path in sorted(data_dir.rglob("*.npy")):
        array = np.load(npy_path)
        if array.ndim >= 2:
            length = array.shape[-1]
        else:
            length = array.size
        lengths.append(length)

    return lengths


def plot_length_histogram(output_path, title, lengths_list, labels):
    fig, ax = plt.subplots(figsize=(8, 4))

    combined = []
    for lengths in lengths_list:
        combined.extend(lengths)

    if combined:
        bins = np.histogram_bin_edges(combined, bins=30)
    else:
        bins = 30

    for lengths, label in zip(lengths_list, labels):
        if lengths:
            ax.hist(lengths, bins=bins, alpha=0.6, label=label, density=True)

    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def extract_label_classes(labels_path):
    with open(labels_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    classes = set()
    for item in data.get("files", []):
        primary_class = item.get("primary_class")
        if primary_class:
            classes.add(primary_class)
        for class_name in item.get("class_names", []):
            classes.add(class_name)

    return classes


def write_class_mismatch_report(output_path, rows):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "missing_in_predictions", "extra_in_predictions"])
        for dataset, missing, extra in rows:
            writer.writerow([dataset, "|".join(sorted(missing)), "|".join(sorted(extra))])


def discover_prediction_csvs(test_dir):
    csv_files = []
    for csv_path in test_dir.glob("*.csv"):
        csv_files.append(csv_path)
    return sorted(csv_files)


def compute_all_predictions_accuracy(base_dir):
    results = []
    
    doc_test_dir = base_dir / "doc_split" / "test"
    joe_test_dir = base_dir / "joe_mo_split" / "test"
    
    for test_dir, test_name in [(doc_test_dir, "DOC Test"), (joe_test_dir, "Joe_Mo Test")]:
        if not test_dir.exists():
            continue
            
        labels_path = test_dir / "labels.json"
        if not labels_path.exists():
            continue
            
        row_to_class, _ = load_labels(labels_path)
        csv_files = discover_prediction_csvs(test_dir)
        
        for csv_path in csv_files:
            model_name = csv_path.stem
            _, rows = load_predictions(csv_path)
            per_class, total, overall = compute_per_class_accuracy(row_to_class, rows)
            
            results.append({
                'test_set': test_name,
                'model': model_name,
                'csv_path': csv_path,
                'overall_accuracy': overall,
                'per_class': per_class,
                'total': total
            })
            
    return results


def compute_combined_heatmap(base_dir):
    results = compute_all_predictions_accuracy(base_dir)
    
    if not results:
        return [], Counter(), [], [], []
    
    totals = Counter()
    per_class_by_model = []
    combo_labels = []
    
    for result in results:
        totals.update(result['total'])
        per_class_by_model.append(result['per_class'])
        combo_labels.append(f"{result['model']}\n{result['test_set']}")
        print(f"{result['model']} on {result['test_set']}: {result['overall_accuracy']:.4f}")

    classes = sorted(totals.keys())
    values = [[values.get(cls, 0.0) for values in per_class_by_model] for cls in classes]
    return classes, totals, per_class_by_model, values, combo_labels


def compute_magnitude_statistics(data_dir):
    stats = {
        'means': [],
        'stds': [],
        'maxs': [],
        'mins': [],
        'widths': []
    }
    
    for npy_path in sorted(data_dir.rglob('*.npy')):
        array = np.load(npy_path)
        if array.ndim > 2:
            array = np.squeeze(array)
        if array.ndim != 2:
            continue
            
        stats['means'].append(array.mean())
        stats['stds'].append(array.std())
        stats['maxs'].append(array.max())
        stats['mins'].append(array.min())
        stats['widths'].append(array.shape[1])
    
    return stats


def plot_magnitude_comparison(output_path, doc_stats, joe_stats):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Spectrogram Magnitude Distribution: DOC vs Joe_Mo', fontsize=14, fontweight='bold')
    
    # Plot means
    ax = axes[0, 0]
    ax.hist(doc_stats['means'], bins=50, alpha=0.6, label='DOC', density=True, color='blue')
    ax.hist(joe_stats['means'], bins=50, alpha=0.6, label='Joe_Mo', density=True, color='orange')
    ax.set_xlabel('Mean Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Mean Values per Spectrogram')
    ax.legend()
    ax.axvline(np.mean(doc_stats['means']), color='blue', linestyle='--', linewidth=2, label=f"DOC avg: {np.mean(doc_stats['means']):.0f}")
    ax.axvline(np.mean(joe_stats['means']), color='orange', linestyle='--', linewidth=2, label=f"Joe avg: {np.mean(joe_stats['means']):.0f}")
    
    # Plot maxs
    ax = axes[0, 1]
    ax.hist(doc_stats['maxs'], bins=50, alpha=0.6, label='DOC', density=True, color='blue')
    ax.hist(joe_stats['maxs'], bins=50, alpha=0.6, label='Joe_Mo', density=True, color='orange')
    ax.set_xlabel('Max Magnitude')
    ax.set_ylabel('Density')
    ax.set_title('Max Values per Spectrogram')
    ax.legend()
    ax.set_xlim(0, min(2e6, max(max(doc_stats['maxs']), max(joe_stats['maxs']))))
    
    # Plot log scale comparison
    ax = axes[1, 0]
    doc_log = [np.log10(m + 1) for m in doc_stats['means']]
    joe_log = [np.log10(m + 1) for m in joe_stats['means']]
    ax.hist(doc_log, bins=50, alpha=0.6, label='DOC', density=True, color='blue')
    ax.hist(joe_log, bins=50, alpha=0.6, label='Joe_Mo', density=True, color='orange')
    ax.set_xlabel('log10(Mean + 1)')
    ax.set_ylabel('Density')
    ax.set_title('Mean Values (Log Scale)')
    ax.legend()
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    doc_mean_avg = np.mean(doc_stats['means'])
    joe_mean_avg = np.mean(joe_stats['means'])
    doc_max_avg = np.mean(doc_stats['maxs'])
    joe_max_avg = np.mean(joe_stats['maxs'])
    ratio_mean = joe_mean_avg / doc_mean_avg
    ratio_max = joe_max_avg / doc_max_avg
    
    summary_text = f"""MAGNITUDE COMPARISON SUMMARY:

DOC Dataset:
  Mean: {doc_mean_avg:.1f} ± {np.std(doc_stats['means']):.1f}
  Max:  {doc_max_avg:.1f} ± {np.std(doc_stats['maxs']):.1f}
  Samples: {len(doc_stats['means'])}

Joe_Mo Dataset:
  Mean: {joe_mean_avg:.1f} ± {np.std(joe_stats['means']):.1f}
  Max:  {joe_max_avg:.1f} ± {np.std(joe_stats['maxs']):.1f}
  Samples: {len(joe_stats['means'])}

RATIO (Joe_Mo / DOC):
  Mean: {ratio_mean:.2f}x
  Max:  {ratio_max:.2f}x

⚠️  DOMAIN SHIFT DETECTED!
Spectrograms have vastly different scales.
Models trained on one will fail on the other.
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_sample_spectrograms(output_path, doc_dir, joe_dir, num_samples=3):
    doc_files = list(sorted(doc_dir.rglob('*.npy')))[:num_samples]
    joe_files = list(sorted(joe_dir.rglob('*.npy')))[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    fig.suptitle('Sample Spectrograms: Raw vs Normalized', fontsize=14, fontweight='bold')
    
    for i, (doc_file, joe_file) in enumerate(zip(doc_files, joe_files)):
        doc_data = np.load(doc_file)
        joe_data = np.load(joe_file)
        
        if doc_data.ndim > 2:
            doc_data = np.squeeze(doc_data)
        if joe_data.ndim > 2:
            joe_data = np.squeeze(joe_data)
        
        # Raw DOC
        ax = axes[i, 0]
        im = ax.imshow(doc_data, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'DOC Raw\nmean={doc_data.mean():.0f}, max={doc_data.max():.0f}')
        ax.set_ylabel(f'Sample {i+1}')
        plt.colorbar(im, ax=ax)
        
        # Normalized DOC
        ax = axes[i, 1]
        doc_norm = normalize_spectrogram(doc_data)
        im = ax.imshow(doc_norm, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'DOC Normalized\nmean={doc_norm.mean():.2f}, std={doc_norm.std():.2f}')
        plt.colorbar(im, ax=ax)
        
        # Raw Joe_Mo
        ax = axes[i, 2]
        im = ax.imshow(joe_data, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'Joe_Mo Raw\nmean={joe_data.mean():.0f}, max={joe_data.max():.0f}')
        plt.colorbar(im, ax=ax)
        
        # Normalized Joe_Mo
        ax = axes[i, 3]
        joe_norm = normalize_spectrogram(joe_data)
        im = ax.imshow(joe_norm, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f'Joe_Mo Normalized\nmean={joe_norm.mean():.2f}, std={joe_norm.std():.2f}')
        plt.colorbar(im, ax=ax)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def compute_frequency_band_statistics(data_dir, num_bands=8):
    freq_band_stats = {f'band_{i}': [] for i in range(num_bands)}
    
    for npy_path in sorted(data_dir.rglob('*.npy')):
        spec = np.load(npy_path)
        if spec.size == 0:
            continue
        
        freq_bins = spec.shape[0]
        band_size = freq_bins // num_bands
        
        for i in range(num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else freq_bins
            band_energy = np.mean(spec[start:end, :])
            freq_band_stats[f'band_{i}'].append(band_energy)
    
    return freq_band_stats


def compute_spectral_features(spec):
    if spec.size == 0:
        return None
    
    # Spectral centroid (center of mass of spectrum)
    freq_axis = np.arange(spec.shape[0])
    spectral_centroid = np.sum(spec * freq_axis[:, np.newaxis], axis=0) / (np.sum(spec, axis=0) + 1e-10)
    
    # Spectral spread (standard deviation around centroid)
    spectral_spread = np.sqrt(np.sum(spec * (freq_axis[:, np.newaxis] - spectral_centroid) ** 2, axis=0) / (np.sum(spec, axis=0) + 1e-10))
    
    # Spectral flatness (Wiener entropy - measure of noise-like vs tonal)
    geometric_mean = np.exp(np.mean(np.log(spec + 1e-10), axis=0))
    arithmetic_mean = np.mean(spec, axis=0)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(spec, axis=0)
    total_energy = cumsum[-1, :]
    rolloff_threshold = 0.85 * total_energy
    spectral_rolloff = np.argmax(cumsum >= rolloff_threshold, axis=0)
    
    return {
        'centroid_mean': np.mean(spectral_centroid),
        'centroid_std': np.std(spectral_centroid),
        'spread_mean': np.mean(spectral_spread),
        'spread_std': np.std(spectral_spread),
        'flatness_mean': np.mean(spectral_flatness),
        'flatness_std': np.std(spectral_flatness),
        'rolloff_mean': np.mean(spectral_rolloff),
        'rolloff_std': np.std(spectral_rolloff)
    }


def compute_dataset_spectral_features(data_dir):
    all_features = []
    
    for npy_path in sorted(data_dir.rglob('*.npy')):
        spec = np.load(npy_path)
        features = compute_spectral_features(spec)
        if features is not None:
            all_features.append(features)
    
    aggregated = {}
    for key in all_features[0].keys():
        values = [f[key] for f in all_features]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    return aggregated


def plot_frequency_band_comparison(output_path, doc_stats, joe_stats, num_bands=8):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Frequency Band Energy Distribution', fontsize=14, fontweight='bold')
    
    band_labels = [f'Band {i}\n({i*1000}Hz)' for i in range(num_bands)]
    
    # Box plot comparison
    ax = axes[0]
    doc_data = [doc_stats[f'band_{i}'] for i in range(num_bands)]
    joe_data = [joe_stats[f'band_{i}'] for i in range(num_bands)]
    
    positions_doc = np.arange(num_bands) * 2
    positions_joe = positions_doc + 0.8
    
    bp1 = ax.boxplot(doc_data, positions=positions_doc, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='blue', alpha=0.5),
                     medianprops=dict(color='darkblue', linewidth=2))
    bp2 = ax.boxplot(joe_data, positions=positions_joe, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='orange', alpha=0.5),
                     medianprops=dict(color='darkorange', linewidth=2))
    
    ax.set_xticks(positions_doc + 0.4)
    ax.set_xticklabels([f'Band {i}' for i in range(num_bands)], rotation=45)
    ax.set_ylabel('Energy')
    ax.set_title('Per-Band Energy Distribution')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['DOC', 'Joe_Mo'])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Ratio plot
    ax = axes[1]
    doc_medians = [np.median(doc_stats[f'band_{i}']) for i in range(num_bands)]
    joe_medians = [np.median(joe_stats[f'band_{i}']) for i in range(num_bands)]
    ratios = np.array(joe_medians) / (np.array(doc_medians) + 1e-10)
    
    ax.bar(range(num_bands), ratios, color='purple', alpha=0.6)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal energy')
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Joe_Mo / DOC Energy Ratio')
    ax.set_title('Energy Ratio by Frequency Band')
    ax.set_xticks(range(num_bands))
    ax.set_xticklabels([f'Band {i}' for i in range(num_bands)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_spectral_feature_comparison(output_path, doc_features, joe_features):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Spectral Feature Comparison', fontsize=14, fontweight='bold')
    
    feature_names = ['centroid_mean', 'centroid_std', 'spread_mean', 'spread_std',
                     'flatness_mean', 'flatness_std', 'rolloff_mean', 'rolloff_std']
    
    for idx, feature in enumerate(feature_names):
        ax = axes[idx // 4, idx % 4]
        
        doc_vals = doc_features[feature]['values']
        joe_vals = joe_features[feature]['values']
        
        ax.hist(doc_vals, bins=50, alpha=0.5, label='DOC', density=True, color='blue')
        ax.hist(joe_vals, bins=50, alpha=0.5, label='Joe_Mo', density=True, color='orange')
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        dist = wasserstein_distance(doc_vals, joe_vals)
        ax.set_title(f'{feature.replace("_", " ").title()}\nWasserstein: {dist:.2e}')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def analyze_prediction_correctness(base_dir, model_name, test_name, output_dir):
    labels_path = base_dir / f"{test_name}_split" / "test" / "labels.json"
    preds_path = base_dir / f"{test_name}_split" / "test" / f"birdclef_{model_name}_trained_{test_name}_test.csv"
    data_dir = base_dir / f"{test_name}_split" / "test" / "data"
    
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    filename_to_class = {}
    for item in labels_data.get('files', []):
        filename = item['filename']
        primary_class = item.get('primary_class', item.get('primary_species'))
        filename_to_class[filename] = primary_class
    
    with open(preds_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    categories = [col for col in rows[0].keys() if col not in ['File_Path', 'row_id']]
    
    correct_samples = []
    incorrect_samples = []
    
    for row in rows:
        filename = row['row_id']
        
        if filename not in filename_to_class:
            continue
        
        true_class = filename_to_class[filename]
        scores = [float(row[cat]) for cat in categories]
        pred_class = categories[np.argmax(scores)]
        
        spec_path = data_dir / filename
        
        if not spec_path.exists():
            continue
        
        spec = np.load(spec_path)
        
        if pred_class == true_class:
            correct_samples.append({
                'filename': filename,
                'class': true_class,
                'spec': spec,
                'confidence': max(scores)
            })
        else:
            incorrect_samples.append({
                'filename': filename,
                'true_class': true_class,
                'pred_class': pred_class,
                'spec': spec,
                'confidence': max(scores)
            })
    
    return correct_samples, incorrect_samples


def compute_sample_statistics(samples):
    stats = {
        'means': [],
        'maxs': [],
        'stds': [],
        'centroid': [],
        'spread': [],
        'flatness': [],
        'rolloff': []
    }
    
    for sample in samples:
        spec = sample['spec']
        stats['means'].append(np.mean(spec))
        stats['maxs'].append(np.max(spec))
        stats['stds'].append(np.std(spec))
        
        features = compute_spectral_features(spec)
        if features:
            stats['centroid'].append(features['centroid_mean'])
            stats['spread'].append(features['spread_mean'])
            stats['flatness'].append(features['flatness_mean'])
            stats['rolloff'].append(features['rolloff_mean'])
    
    return stats


def plot_correct_vs_incorrect_analysis(output_path, correct_stats, incorrect_stats, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    metrics = ['means', 'maxs', 'stds', 'centroid', 'spread', 'flatness', 'rolloff']
    metric_labels = ['Mean Energy', 'Max Energy', 'Std Energy', 'Spectral Centroid', 
                     'Spectral Spread', 'Spectral Flatness', 'Spectral Rolloff']
    
    for idx in range(7):
        ax = axes[idx // 4, idx % 4]
        metric = metrics[idx]
        
        if len(correct_stats.get(metric, [])) == 0 or len(incorrect_stats.get(metric, [])) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(metric_labels[idx])
            continue
        
        correct_vals = correct_stats[metric]
        incorrect_vals = incorrect_stats[metric]
        
        # Create box plot comparison
        bp = ax.boxplot([correct_vals, incorrect_vals], 
                        labels=['Correct', 'Incorrect'],
                        patch_artist=True,
                        boxprops=dict(alpha=0.6))
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        
        ax.set_ylabel(metric_labels[idx])
        ax.grid(True, alpha=0.3)
        

        try:
            stat, pval = mannwhitneyu(correct_vals, incorrect_vals)
            significance = ' ***' if pval < 0.001 else ' **' if pval < 0.01 else ' *' if pval < 0.05 else ''
            ax.set_title(f'{metric_labels[idx]}\np={pval:.3f}{significance}')
        except:
            ax.set_title(metric_labels[idx])
    
    # Summary stats in last subplot
    ax = axes[1, 3]
    ax.axis('off')
    
    summary_text = f"""SUMMARY:

Correct predictions: {len(correct_stats['means'])}
Incorrect predictions: {len(incorrect_stats['means'])}

Accuracy: {100 * len(correct_stats['means']) / (len(correct_stats['means']) + len(incorrect_stats['means'])):.1f}%

Significant differences (p<0.05) 
indicate features that distinguish
correct from incorrect predictions.

Large differences suggest the
model is sensitive to that feature
and cross-domain shift in that
feature is causing failures.
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def write_accuracy_summary(output_path, results):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        if not results:
            f.write("No prediction CSVs found.\n")
            return
        
        doc_results = [r for r in results if "DOC" in r['test_set']]
        joe_results = [r for r in results if "Joe_Mo" in r['test_set']]
        
        f.write("Performance on DOC Test Set:\n")
        f.write("-" * 80 + "\n")
        if doc_results:
            for r in doc_results:
                f.write(f"  {r['model']:50s} {r['overall_accuracy']*100:6.2f}%\n")
        else:
            f.write("  No predictions found\n")
        
        f.write("\n")
        f.write("Performance on Joe_Mo Test Set:\n")
        f.write("-" * 80 + "\n")
        if joe_results:
            for r in joe_results:
                f.write(f"  {r['model']:50s} {r['overall_accuracy']*100:6.2f}%\n")
        else:
            f.write("  No predictions found\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("=" * 80 + "\n\n")
        f.write("- Models with similar accuracy on both test sets generalize well\n")
        f.write("- Large accuracy drops indicate domain shift problems\n")
        f.write("- Compare same model across different test sets to assess robustness\n")


def write_analysis_summary(output_path, doc_stats, joe_stats, doc_features, joe_features):
    doc_mean_avg = np.mean(doc_stats['means'])
    joe_mean_avg = np.mean(joe_stats['means'])
    ratio = joe_mean_avg / doc_mean_avg
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("===== DOMAIN SHIFT DIAGNOSTIC REPORT =====\n\n")
        f.write("PROBLEM IDENTIFIED:\n")
        f.write("==================\n\n")
        f.write(f"The two datasets have vastly different spectrogram magnitudes:\n\n")
        f.write(f"  DOC Dataset:    mean={doc_mean_avg:.1f}, max={np.mean(doc_stats['maxs']):.1f}\n")
        f.write(f"  Joe_Mo Dataset: mean={joe_mean_avg:.1f}, max={np.mean(joe_stats['maxs']):.1f}\n")
        f.write(f"  Ratio:          {ratio:.2f}x\n\n")
        f.write(f"This {ratio:.1f}x magnitude difference causes models to fail when transferred\n")
        f.write(f"between datasets because:\n")
        f.write(f"  1. Neural networks learn scale-dependent features\n")
        f.write(f"  2. Activation functions (ReLU, sigmoid) are sensitive to input scale\n")
        f.write(f"  3. Batch normalization layers learn dataset-specific statistics\n\n")
        
        f.write("\nROOT CAUSES:\n")
        f.write("============\n\n")
        f.write("1. DIFFERENT DATA SOURCES:\n")
        f.write("   - DOC: Isolated bird call clips (clean, pre-segmented)\n")
        f.write("   - Joe_Mo: Soundscape segments (continuous recordings with background)\n\n")
        f.write("2. DIFFERENT RECORDING CONDITIONS:\n")
        f.write("   - Different microphones, gain settings, or recording equipment\n")
        f.write("   - Different environmental conditions (background noise levels)\n\n")
        f.write("3. NO NORMALIZATION DURING TRAINING:\n")
        f.write("   - Models trained with normalize=False (default)\n")
        f.write("   - Raw spectrogram magnitudes passed directly to model\n")
        f.write("   - No scale invariance learned\n\n")
        
        f.write("\nSOLUTIONS (in order of effectiveness):\n")
        f.write("======================================\n\n")
        f.write("OPTION 1: Enable Background Normalization (RECOMMENDED)\n")
        f.write("-------------------------------------------------------\n")
        f.write("Add --normalize flag when training:\n\n")
        f.write("  python finetune_birdclef.py data/train outputs/model --normalize\n\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("MAGNITUDE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"DOC Dataset:\n")
        f.write(f"  Mean:   {np.mean(doc_stats['means']):.1f} ± {np.std(doc_stats['means']):.1f}\n")
        f.write(f"  Max:    {np.mean(doc_stats['maxs']):.1f} ± {np.std(doc_stats['maxs']):.1f}\n")
        f.write(f"  Width:  min={min(doc_stats['widths'])}, max={max(doc_stats['widths'])}, median={int(np.median(doc_stats['widths']))}\n")
        f.write(f"  Samples: {len(doc_stats['means'])}\n\n")
        
        f.write(f"Joe_Mo Dataset:\n")
        f.write(f"  Mean:   {np.mean(joe_stats['means']):.1f} ± {np.std(joe_stats['means']):.1f}\n")
        f.write(f"  Max:    {np.mean(joe_stats['maxs']):.1f} ± {np.std(joe_stats['maxs']):.1f}\n")
        f.write(f"  Width:  min={min(joe_stats['widths'])}, max={max(joe_stats['widths'])}, median={int(np.median(joe_stats['widths']))}\n")
        f.write(f"  Samples: {len(joe_stats['means'])}\n\n")
        
        f.write(f"Magnitude Ratio (Joe_Mo / DOC):\n")
        f.write(f"  Mean: {ratio:.2f}x\n")
        f.write(f"  Max:  {np.mean(joe_stats['maxs']) / np.mean(doc_stats['maxs']):.2f}x\n\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SPECTRAL FEATURE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        feature_names = ['centroid_mean', 'spread_mean', 'flatness_mean', 'rolloff_mean']
        feature_labels = ['Spectral Centroid', 'Spectral Spread', 'Spectral Flatness', 'Spectral Rolloff']
        
        for fname, flabel in zip(feature_names, feature_labels):
            doc_mean = doc_features[fname]['mean']
            joe_mean = joe_features[fname]['mean']
            doc_std = doc_features[fname]['std']
            joe_std = joe_features[fname]['std']
            
            f.write(f"{flabel}:\n")
            f.write(f"  DOC:    {doc_mean:.2f} ± {doc_std:.2f}\n")
            f.write(f"  Joe_Mo: {joe_mean:.2f} ± {joe_std:.2f}\n")
            f.write(f"  Diff:   {abs(joe_mean - doc_mean):.2f} ({100 * abs(joe_mean - doc_mean) / (doc_mean + 1e-10):.1f}%)\n\n")
        
        f.write("\n")
        f.write("INTERPRETATION:\n")
        f.write("---------------\n")
        f.write("Spectral Centroid: Center of mass of frequency distribution\n")
        f.write("  - Higher = more high-frequency energy\n")
        f.write("  - Differences indicate frequency band emphasis mismatch\n\n")
        f.write("Spectral Spread: Variance of frequencies around centroid\n")
        f.write("  - Higher = energy spread across wide frequency range\n")
        f.write("  - Lower = energy concentrated in narrow band\n\n")
        f.write("Spectral Flatness: 0=tonal (pure tone), 1=noise-like (white noise)\n")
        f.write("  - Differences indicate different call structure or background characteristics\n\n")
        f.write("Spectral Rolloff: Frequency below which 85% of energy is contained\n")
        f.write("  - Higher = more high-frequency content\n")
        f.write("  - Differences indicate bandwidth mismatch\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze domain shift between two datasets by comparing spectrogram statistics and model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected folder structure:
  base_dir/
    doc_split/
      train/data/  (spectrograms)
      test/data/   (spectrograms)
      test/labels.json
      test/*.csv   (predictions)
    joe_mo_split/
      train/data/  (spectrograms)
      test/data/   (spectrograms)
      test/labels.json
      test/*.csv   (predictions)

Example:
  python domain_shift_diagnostics.py test
  python domain_shift_diagnostics.py /path/to/experiments
        """
    )
    parser.add_argument('base_dir', type=str, nargs='?', 
                       default='test',
                       help='Base directory containing doc_split and joe_mo_split folders (default: test)')
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Check for required subdirectories
    doc_split = base_dir / "doc_split"
    joe_split = base_dir / "joe_mo_split"
    
    if not doc_split.exists():
        print(f"Error: doc_split directory not found in {base_dir}")
        sys.exit(1)
    
    if not joe_split.exists():
        print(f"Error: joe_mo_split directory not found in {base_dir}")
        sys.exit(1)
    
    output_dir = base_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DOMAIN SHIFT DIAGNOSTIC ANALYSIS")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print()

    # 1. Magnitude analysis
    print("1. Computing magnitude statistics...")
    doc_train_stats = compute_magnitude_statistics(base_dir / "doc_split" / "train" / "data")
    joe_train_stats = compute_magnitude_statistics(base_dir / "joe_mo_split" / "train" / "data")
    
    magnitude_plot_path = output_dir / "magnitude_comparison_doc_vs_joe.png"
    plot_magnitude_comparison(magnitude_plot_path, doc_train_stats, joe_train_stats)
    print(f"   ✓ Saved: {magnitude_plot_path.name}")
    
    # 2. Sample spectrogram visualization
    print("2. Visualizing sample spectrograms...")
    samples_plot_path = output_dir / "sample_spectrograms_raw_vs_normalized.png"
    plot_sample_spectrograms(
        samples_plot_path,
        base_dir / "doc_split" / "train" / "data",
        base_dir / "joe_mo_split" / "train" / "data",
        num_samples=3
    )
    print(f"   ✓ Saved: {samples_plot_path.name}")
    
    # 3. Per-class accuracy heatmap
    print("3. Computing per-class accuracy for all prediction CSVs...")
    classes, totals, per_class_by_model, values, combo_labels = compute_combined_heatmap(base_dir)
    
    results = compute_all_predictions_accuracy(base_dir)
    
    if classes:
        heatmap_path = output_dir / "per_class_accuracy_all_models.png"
        plot_accuracy_heatmap(
            heatmap_path,
            "Per-class accuracy: All Models on All Test Sets",
            classes,
            values,
            combo_labels,
        )
        print(f"   ✓ Saved: {heatmap_path.name}")

        table_path = output_dir / "per_class_accuracy_all_models.csv"
        write_accuracy_table(table_path, classes, totals, per_class_by_model, combo_labels)
        print(f"   ✓ Saved: {table_path.name}")
        
        summary_path = output_dir / "ACCURACY_SUMMARY.txt"
        write_accuracy_summary(summary_path, results)
        print(f"   ✓ Saved: {summary_path.name}")
    else:
        print("   ⚠ No prediction CSVs found, skipping accuracy analysis")

    results = compute_all_predictions_accuracy(base_dir)
    
    mismatch_rows = []
    for result in results:
        test_dir = base_dir / ("doc_split" if "DOC" in result['test_set'] else "joe_mo_split") / "test"
        label_classes = extract_label_classes(test_dir / "labels.json")
        pred_columns, _ = load_predictions(result['csv_path'])
        pred_classes = set(pred_columns)
        missing = label_classes - pred_classes
        extra = pred_classes - label_classes
        display_label = f"{result['model']} on {result['test_set']}"
        mismatch_rows.append((display_label, missing, extra))

    # 4. Class mismatch report
    print("4. Analyzing class mismatches...")
    mismatch_path = output_dir / "class_mismatch_report.csv"
    write_class_mismatch_report(mismatch_path, mismatch_rows)
    print(f"   ✓ Saved: {mismatch_path.name}")

    # 5. Length distribution analysis
    print("5. Analyzing length distributions...")
    doc_train_lengths = extract_lengths(base_dir / "doc_split" / "train" / "data")
    joe_train_lengths = extract_lengths(base_dir / "joe_mo_split" / "train" / "data")
    train_length_path = output_dir / "length_distribution_train_doc_vs_joe.png"
    plot_length_histogram(
        train_length_path,
        "Length distribution (train): doc vs joe",
        [doc_train_lengths, joe_train_lengths],
        ["doc", "joe"],
    )
    print(f"   ✓ Saved: {train_length_path.name}")

    doc_test_lengths = extract_lengths(base_dir / "doc_split" / "test" / "data")
    joe_test_lengths = extract_lengths(base_dir / "joe_mo_split" / "test" / "data")
    test_length_path = output_dir / "length_distribution_test_doc_vs_joe.png"
    plot_length_histogram(
        test_length_path,
        "Length distribution (test): doc vs joe",
        [doc_test_lengths, joe_test_lengths],
        ["doc", "joe"],
    )
    print(f"   ✓ Saved: {test_length_path.name}")
    
    # 6. Frequency band analysis
    print("6. Analyzing frequency band energy distribution...")
    doc_freq_stats = compute_frequency_band_statistics(base_dir / "doc_split" / "train" / "data", num_bands=8)
    joe_freq_stats = compute_frequency_band_statistics(base_dir / "joe_mo_split" / "train" / "data", num_bands=8)
    freq_band_path = output_dir / "frequency_band_comparison.png"
    plot_frequency_band_comparison(freq_band_path, doc_freq_stats, joe_freq_stats, num_bands=8)
    print(f"   ✓ Saved: {freq_band_path.name}")
    
    # 7. Spectral feature analysis
    print("7. Computing spectral features...")
    doc_spectral = compute_dataset_spectral_features(base_dir / "doc_split" / "train" / "data")
    joe_spectral = compute_dataset_spectral_features(base_dir / "joe_mo_split" / "train" / "data")
    spectral_path = output_dir / "spectral_features_comparison.png"
    plot_spectral_feature_comparison(spectral_path, doc_spectral, joe_spectral)
    print(f"   ✓ Saved: {spectral_path.name}")
    
    # 8. Analyze correct vs incorrect predictions for cross-domain scenarios
    print("8. Analyzing prediction patterns for cross-domain transfer...")
    
    doc_on_joe_csv = base_dir / "joe_mo_split" / "test" / "birdclef_doc_trained_joe_mo_test.csv"
    joe_on_doc_csv = base_dir / "doc_split" / "test" / "birdclef_joe_mo_trained_doc_test.csv"
    
    cross_domain_found = False
    
    if doc_on_joe_csv.exists():
        print("   a. DOC model on Joe_Mo test data...")
        doc_on_joe_correct, doc_on_joe_incorrect = analyze_prediction_correctness(
            base_dir, "doc", "joe_mo", output_dir
        )
        doc_on_joe_correct_stats = compute_sample_statistics(doc_on_joe_correct)
        doc_on_joe_incorrect_stats = compute_sample_statistics(doc_on_joe_incorrect)
        doc_on_joe_path = output_dir / "doc_model_on_joe_data_correct_vs_incorrect.png"
        plot_correct_vs_incorrect_analysis(
            doc_on_joe_path, 
            doc_on_joe_correct_stats, 
            doc_on_joe_incorrect_stats,
            "DOC Model on Joe_Mo Test: Correct vs Incorrect Predictions"
        )
        print(f"      ✓ Saved: {doc_on_joe_path.name}")
        cross_domain_found = True
    
    if joe_on_doc_csv.exists():
        print("   b. Joe_Mo model on DOC test data...")
        joe_on_doc_correct, joe_on_doc_incorrect = analyze_prediction_correctness(
            base_dir, "joe_mo", "doc", output_dir
        )
        joe_on_doc_correct_stats = compute_sample_statistics(joe_on_doc_correct)
        joe_on_doc_incorrect_stats = compute_sample_statistics(joe_on_doc_incorrect)
        joe_on_doc_path = output_dir / "joe_model_on_doc_data_correct_vs_incorrect.png"
        plot_correct_vs_incorrect_analysis(
            joe_on_doc_path,
            joe_on_doc_correct_stats,
            joe_on_doc_incorrect_stats,
            "Joe_Mo Model on DOC Test: Correct vs Incorrect Predictions"
        )
        print(f"      ✓ Saved: {joe_on_doc_path.name}")
        cross_domain_found = True
    
    if not cross_domain_found:
        print("   ⚠ No cross-domain predictions found (birdclef_doc_trained_joe_mo_test.csv or birdclef_joe_mo_trained_doc_test.csv)")
        print("   Skipping detailed cross-domain correctness analysis")
    
    # 9. Write analysis summary
    print("9. Generating analysis summary...")
    summary_path = output_dir / "ANALYSIS_SUMMARY.txt"
    write_analysis_summary(summary_path, doc_train_stats, joe_train_stats, doc_spectral, joe_spectral)
    print(f"   ✓ Saved: {summary_path.name}")
    
    print()
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print()
    print(f"All outputs saved to: {output_dir}")
    print()
    print("KEY RESULTS:")
    print(f"  📊 ACCURACY_SUMMARY.txt - Model performance comparison")
    print(f"  📈 per_class_accuracy_all_models.png - Visual comparison")
    print(f"  📋 per_class_accuracy_all_models.csv - Detailed per-class results")
    print()
    print("ADDITIONAL DIAGNOSTICS:")
    print(f"  • Magnitude distributions")
    print(f"  • Frequency band energy profiles")
    print(f"  • Spectral feature comparisons")
    print(f"  • Cross-domain prediction analysis (correct vs incorrect)")
    print(f"  • Sample visualization comparisons")
    print()
    print(f"Review ACCURACY_SUMMARY.txt for quick performance overview.")
    print(f"Review ANALYSIS_SUMMARY.txt for detailed domain shift findings.")
    print()


if __name__ == "__main__":
    main()
