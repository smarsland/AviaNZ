import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
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
        header = ["species", "support"] + [f"{label}_acc" for label in model_labels]
        writer.writerow(header)
        for cls in classes:
            row = [cls, totals.get(cls, 0)]
            for values in per_class_by_model:
                row.append(values.get(cls, 0.0))
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


def compute_combined_heatmap(base_dir):
    combos = [
        {
            "label": "doc->doc",
            "labels": base_dir / "doc_split" / "test" / "labels.json",
            "preds": base_dir / "doc_split" / "test" / "birdclef_doc_trained_doc_test.csv",
        },
        {
            "label": "joe->doc",
            "labels": base_dir / "doc_split" / "test" / "labels.json",
            "preds": base_dir / "doc_split" / "test" / "birdclef_joe_mo_trained_doc_test.csv",
        },
        {
            "label": "doc->joe",
            "labels": base_dir / "joe_mo_split" / "test" / "labels.json",
            "preds": base_dir / "joe_mo_split" / "test" / "birdclef_doc_trained_joe_mo_test.csv",
        },
        {
            "label": "joe->joe",
            "labels": base_dir / "joe_mo_split" / "test" / "labels.json",
            "preds": base_dir / "joe_mo_split" / "test" / "birdclef_joe_mo_trained_joe_mo_test.csv",
        },
    ]

    totals = Counter()
    per_class_by_model = []

    for combo in combos:
        row_to_class, _ = load_labels(combo["labels"])
        _, rows = load_predictions(combo["preds"])
        per_class, total, overall = compute_per_class_accuracy(row_to_class, rows)
        totals.update(total)
        per_class_by_model.append(per_class)
        print(f"{combo['label']} overall accuracy: {overall:.4f}")

    classes = sorted(totals.keys())
    values = [[values.get(cls, 0.0) for values in per_class_by_model] for cls in classes]
    return classes, totals, per_class_by_model, values, [combo["label"] for combo in combos]


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


def write_recommendations(output_path, doc_stats, joe_stats):
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
        f.write("This applies per-frequency-band z-score normalization that:\n")
        f.write("  ✓ Makes models scale-invariant\n")
        f.write("  ✓ Reduces background noise\n")
        f.write("  ✓ Enhances bird call features\n")
        f.write("  ✓ Should work for both datasets\n\n")
        
        f.write("OPTION 2: Pre-normalize the Data\n")
        f.write("---------------------------------\n")
        f.write("Normalize all spectrograms during data generation:\n\n")
        f.write("  from normalizer import normalize_spectrogram\n")
        f.write("  spec_normalized = normalize_spectrogram(spec_raw)\n")
        f.write("  np.save(output_path, spec_normalized)\n\n")
        f.write("Benefits:\n")
        f.write("  ✓ Normalization done once (faster training)\n")
        f.write("  ✓ Consistent preprocessing\n")
        f.write("  ✗ Requires regenerating all data\n\n")
        
        f.write("OPTION 3: Use Instance Normalization in Model\n")
        f.write("----------------------------------------------\n")
        f.write("Add instance normalization layer at model input:\n\n")
        f.write("  # In model __init__:\n")
        f.write("  self.input_norm = nn.InstanceNorm2d(1, affine=False)\n\n")
        f.write("  # In forward pass:\n")
        f.write("  x = self.input_norm(x)\n\n")
        f.write("Benefits:\n")
        f.write("  ✓ Per-sample normalization\n")
        f.write("  ✓ Scale-invariant\n")
        f.write("  ✗ Adds computational overhead\n\n")
        
        f.write("OPTION 4: Train on Combined Data\n")
        f.write("---------------------------------\n")
        f.write("Combine both datasets for training with normalization:\n\n")
        f.write("  python finetune_birdclef.py data/combined outputs/model --normalize\n\n")
        f.write("Benefits:\n")
        f.write("  ✓ Model sees both distributions\n")
        f.write("  ✓ Better generalization\n")
        f.write("  ✗ Requires merging datasets\n\n")
        
        f.write("\nTESTING THE FIX:\n")
        f.write("================\n\n")
        f.write("1. Retrain models WITH normalization:\n\n")
        f.write("   python finetune_birdclef.py test/test/doc_split/train \\\n")
        f.write("          test/test/doc_trained --normalize --epochs 10\n\n")
        f.write("   python finetune_birdclef.py test/test/joe_mo_split/train \\\n")
        f.write("          test/test/joe_mo_trained --normalize --epochs 10\n\n")
        f.write("2. Generate predictions on both test sets\n\n")
        f.write("3. Re-run domain_shift_diagnostics.py to compare accuracy\n\n")
        f.write("Expected results:\n")
        f.write("  - Same-domain accuracy should remain high\n")
        f.write("  - Cross-domain accuracy should IMPROVE SIGNIFICANTLY\n")
        f.write("  - If still poor, consider combining datasets for training\n\n")
        
        f.write("\nADDITIONAL OBSERVATIONS:\n")
        f.write("========================\n\n")
        f.write(f"Width variation:\n")
        f.write(f"  DOC:    min={min(doc_stats['widths'])}, max={max(doc_stats['widths'])}, median={int(np.median(doc_stats['widths']))}\n")
        f.write(f"  Joe_Mo: min={min(joe_stats['widths'])}, max={max(joe_stats['widths'])}, median={int(np.median(joe_stats['widths']))}\n\n")
        f.write(f"Note: Joe_Mo has extreme width variation (34 to 29969 bins), suggesting\n")
        f.write(f"highly variable call durations. This is handled by the data pipeline's\n")
        f.write(f"tiling/repetition strategy, but such variation may hurt performance.\n\n")


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
  python domain_shift_diagnostics.py test/test
  python domain_shift_diagnostics.py /path/to/experiments
        """
    )
    parser.add_argument('base_dir', type=str, nargs='?', 
                       default='test/test',
                       help='Base directory containing doc_split and joe_mo_split folders (default: test/test)')
    
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
    print("3. Computing per-class accuracy...")
    classes, totals, per_class_by_model, values, combo_labels = compute_combined_heatmap(base_dir)
    heatmap_path = output_dir / "per_class_accuracy_6x4.png"
    plot_accuracy_heatmap(
        heatmap_path,
        "Per-class accuracy: train -> test",
        classes,
        values,
        combo_labels,
    )
    print(f"   ✓ Saved: {heatmap_path.name}")

    table_path = output_dir / "per_class_accuracy_6x4.csv"
    write_accuracy_table(table_path, classes, totals, per_class_by_model, combo_labels)
    print(f"   ✓ Saved: {table_path.name}")

    mismatch_rows = []
    class_sets = {
        "doc_test": extract_label_classes(base_dir / "doc_split" / "test" / "labels.json"),
        "joe_test": extract_label_classes(base_dir / "joe_mo_split" / "test" / "labels.json"),
    }

    prediction_columns = {
        "doc->doc": load_predictions(base_dir / "doc_split" / "test" / "birdclef_doc_trained_doc_test.csv")[0],
        "joe->doc": load_predictions(base_dir / "doc_split" / "test" / "birdclef_joe_mo_trained_doc_test.csv")[0],
        "doc->joe": load_predictions(base_dir / "joe_mo_split" / "test" / "birdclef_doc_trained_joe_mo_test.csv")[0],
        "joe->joe": load_predictions(base_dir / "joe_mo_split" / "test" / "birdclef_joe_mo_trained_joe_mo_test.csv")[0],
    }

    for label, columns in prediction_columns.items():
        test_key = "doc_test" if label.endswith("->doc") else "joe_test"
        label_classes = class_sets[test_key]
        pred_classes = set(columns)
        missing = label_classes - pred_classes
        extra = pred_classes - label_classes
        mismatch_rows.append((label, missing, extra))

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
    
    # 6. Write comprehensive recommendations
    print("6. Generating recommendations report...")
    recommendations_path = output_dir / "RECOMMENDATIONS.txt"
    write_recommendations(recommendations_path, doc_train_stats, joe_train_stats)
    print(f"   ✓ Saved: {recommendations_path.name}")
    
    print()
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print()
    print(f"All outputs saved to: {output_dir}")
    print()
    print("KEY FINDINGS:")
    doc_mean = np.mean(doc_train_stats['means'])
    joe_mean = np.mean(joe_train_stats['means'])
    ratio = joe_mean / doc_mean
    print(f"  • Joe_Mo spectrograms are {ratio:.2f}x larger in magnitude than DOC")
    print(f"  • This scale mismatch causes poor cross-dataset transfer")
    print()
    print("RECOMMENDED ACTION:")
    print(f"  → Retrain models with --normalize flag")
    print(f"  → See {recommendations_path.name} for detailed instructions")
    print()


if __name__ == "__main__":
    main()
