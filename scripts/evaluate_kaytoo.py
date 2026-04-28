"""
Evaluate Kaytoo model on test datasets and compare with ground truth.

Expects each test folder to contain:
  labels.json   - with class_names field per sample
  audio/        - .wav files named file_XXXXXXXX.wav

Run build_matched_datasets.py with --with-audio first to generate audio files.
The split_matched_datasets.py script will carry the audio/ folder through to
the train/test splits automatically.

For longer clips, Kaytoo chunks the audio into fixed-length windows (typically
~5 seconds each, determined by the model config). Multiple chunk predictions
are aggregated here by taking the max score across chunks before picking the
top-1 species.

Usage:
    python scripts/evaluate_kaytoo.py \\
        /path/to/avianz_test /path/to/doc_test \\
        --kaytoo-root /path/to/Kaytoo \\
        --mapping data/DOC_bird_naming_map.csv \\
        --output results/kaytoo_eval
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd


def build_label_to_ebird(mapping_csv):
    """Build lowercase-normalized label string -> eBird code from DOC naming map."""
    df = pd.read_csv(mapping_csv)
    mapping = {}
    for _, row in df.iterrows():
        ebird = row.get('eBird')
        if pd.isna(ebird):
            continue
        ebird = str(ebird).strip()
        for col in ['CommonName', 'ExtraName', 'ListDOCBirds']:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                key = str(val).strip().lower()
                key = key.replace(' / ', '/').replace('/ ', '/').replace(' /', '/')
                mapping[key] = ebird
    return mapping


def find_bird_map(kaytoo_root):
    """Find bird_map.csv; tries resources/ subdirectory then the root itself."""
    for candidate in [
        Path(kaytoo_root) / 'resources' / 'bird_map.csv',
        Path(kaytoo_root) / 'bird_map.csv',
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"bird_map.csv not found under {kaytoo_root}")


def load_labels(test_folder):
    with open(Path(test_folder) / 'labels.json') as f:
        return json.load(f)


def collect_audio_files(test_folder):
    audio_dir = Path(test_folder) / 'audio'
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"No audio/ subfolder in {test_folder}. "
            "Run build_matched_datasets.py with --with-audio first."
        )
    return sorted(audio_dir.glob('*.wav'))


def aggregate_to_file(pred_df):
    """Max-pool per-chunk predictions to one row per source file."""
    species_cols = [c for c in pred_df.columns if c not in ('row_id', 'File_Path')]
    per_file = pred_df.groupby('File_Path')[species_cols].max().reset_index()
    return per_file, species_cols


def evaluate_folder(test_folder, dataset_name, models, label_to_ebird, threshold=0.5):
    """Run Kaytoo inference on one test folder and return accuracy statistics.
    
    Predictions are thresholded at `threshold` per class (multi-label), then
    exact-match accuracy is computed — identical to how our trained models are
    evaluated in evaluation_utils.py.
    """
    from kaytoo_infer import inference as kaytoo_inference

    labels_data = load_labels(test_folder)
    audio_files = collect_audio_files(test_folder)

    name_to_meta = {
        item['filename'].replace('.npy', '.wav'): item
        for item in labels_data.get('files', [])
    }

    # Build the ordered list of eBird codes present in this test set.
    # We only score over these classes — same constraint as our trained models.
    test_ebird_codes_ordered = []
    seen = set()
    for item in labels_data.get('files', []):
        for l in item.get('class_names', []):
            for part in l.split('/'):
                code = label_to_ebird.get(part.strip())
                if code and code not in seen:
                    seen.add(code)
                    test_ebird_codes_ordered.append(code)
    test_ebird_codes_ordered = sorted(test_ebird_codes_ordered)
    print(f"  Test set species ({len(test_ebird_codes_ordered)} eBird codes): {test_ebird_codes_ordered}")

    print(f"  {len(audio_files)} audio files")
    if not audio_files:
        print("  No audio files found, skipping.")
        return None

    pred_df = kaytoo_inference(audio_files, models, model_idx=0, cores=1)
    per_file_df, species_cols = aggregate_to_file(pred_df)

    valid_cols = [c for c in test_ebird_codes_ordered if c in species_cols]
    missing = set(test_ebird_codes_ordered) - set(species_cols)
    if missing:
        print(f"  WARNING: {len(missing)} test-set species not in Kaytoo vocab: {sorted(missing)}")
    if not valid_cols:
        print("  ERROR: no overlap between test-set species and Kaytoo vocab")
        return None
    print(f"  Scoring over {len(valid_cols)} species (threshold={threshold})")

    results = []
    for _, row in per_file_df.iterrows():
        wav_name = Path(row['File_Path']).name
        meta = name_to_meta.get(wav_name)
        if meta is None:
            continue

        gt_labels = meta.get('class_names', [meta.get('label')])
        gt_labels = [l for l in gt_labels if l]
        gt_codes = set()
        for l in gt_labels:
            for part in l.split('/'):
                code = label_to_ebird.get(part.strip())
                if code and code in set(valid_cols):
                    gt_codes.add(code)

        # Multi-label: threshold each class score independently
        scores = row[valid_cols].values
        pred_codes = {valid_cols[i] for i, s in enumerate(scores) if s > threshold}
        # If nothing clears the threshold, take the top-scoring class
        if not pred_codes:
            pred_codes = {valid_cols[int(np.argmax(scores))]}

        # Exact match: predicted set must equal ground-truth set exactly
        correct = pred_codes == gt_codes

        results.append({
            'wav_file': wav_name,
            'gt_codes': sorted(gt_codes),
            'pred_codes': sorted(pred_codes),
            'correct': correct,
        })

    n = len(results)
    n_correct = sum(r['correct'] for r in results)
    accuracy = 100.0 * n_correct / n if n else 0.0
    print(f"  Accuracy: {n_correct}/{n} = {accuracy:.1f}%")

    species_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        for code in r['gt_codes']:
            species_stats[code]['total'] += 1
            if r['correct']:
                species_stats[code]['correct'] += 1

    return {
        'dataset_name': dataset_name,
        'num_files': n,
        'num_correct': n_correct,
        'accuracy': accuracy,
        'species_stats': {k: dict(v) for k, v in species_stats.items()},
        'results': results,
    }


def plot_per_species(all_results, output_path):
    combined = defaultdict(lambda: {'correct': 0, 'total': 0})
    for result in all_results:
        for code, stats in result.get('species_stats', {}).items():
            combined[code]['correct'] += stats['correct']
            combined[code]['total'] += stats['total']

    species_list = sorted(combined.keys(), key=lambda c: -combined[c]['total'])
    if not species_list:
        return

    accuracies = []
    counts = []
    for c in species_list:
        total = combined[c]['total']
        correct = combined[c]['correct']
        accuracies.append(100.0 * correct / total if total else 0.0)
        counts.append(total)

    fig, ax = plt.subplots(figsize=(max(10, len(species_list) * 0.9), 6))
    bars = ax.bar(range(len(species_list)), accuracies, color='steelblue', alpha=0.8)
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.0f}%\n(n={count})', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Kaytoo Per-Species Accuracy (top-1, max-pooled across chunks)')
    ax.set_xticks(range(len(species_list)))
    ax.set_xticklabels(species_list, rotation=45, ha='right')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = output_path / 'per_species_accuracy.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def plot_dataset_comparison(all_results, output_path):
    names = [r['dataset_name'] for r in all_results]
    accs = [r['accuracy'] for r in all_results]
    counts = [r['num_files'] for r in all_results]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    bars = ax.bar(range(len(names)), accs, color='lightcoral', alpha=0.8)
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.1f}%\n(n={count})', ha='center', va='bottom')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Kaytoo Top-1 Accuracy by Dataset')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = output_path / 'dataset_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Kaytoo model on test datasets')
    parser.add_argument('test_folders', nargs='+',
                        help='Test dataset folders (must contain labels.json and audio/)')
    parser.add_argument('--kaytoo-root', required=True,
                        help='Root of the Kaytoo installation (parent of models/ and resources/)')
    parser.add_argument('--mapping', default='data/DOC_bird_naming_map.csv',
                        help='DOC bird naming map CSV (default: data/DOC_bird_naming_map.csv)')
    parser.add_argument('--output', required=True, help='Folder for results and plots')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold for multi-label prediction (default: 0.5)')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(args.kaytoo_root)))

    from bird_naming_utils import BirdNamer
    from kaytoo_infer import DefaultConfig, ModelParameters, Models

    bird_map_path = find_bird_map(args.kaytoo_root)
    bird_map_df = pd.read_csv(bird_map_path)
    birdnames = BirdNamer(bird_map_df)

    use_case = {
        'project_root': str(args.kaytoo_root),
        'experiment': None,
        'cpu_only': args.cpu,
        'num_cores': 1,
        'naming_scheme': 'eBird',
    }

    cfg = DefaultConfig(bird_namer=birdnames, options=use_case)
    parameters = ModelParameters(options=use_case)
    models = Models(config=cfg, model_parameters=parameters)

    label_to_ebird = build_label_to_ebird(args.mapping)

    all_results = []
    for folder in args.test_folders:
        # Use parent folder name (e.g. "avianz_split") to match the convention
        # used by run_cross_dataset_experiments.py: test1_name = test1_path.parent.name
        p = Path(folder).resolve()
        name = p.parent.name
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        result = evaluate_folder(folder, name, models, label_to_ebird, threshold=args.threshold)
        if result:
            all_results.append(result)

    if not all_results:
        print("No results to report.")
        return

    # Write standard result.json — same format as trained-model experiments so
    # analyze_all_results.py picks this up alongside all other results.
    result_json = {
        'name': output_path.name,
        'type': 'pretrained',
        'seed': 0,
        'status': 'completed',
    }
    if len(all_results) >= 1:
        result_json['test1_name'] = all_results[0]['dataset_name']
        result_json['test1_acc'] = all_results[0]['accuracy']
    if len(all_results) >= 2:
        result_json['test2_name'] = all_results[1]['dataset_name']
        result_json['test2_acc'] = all_results[1]['accuracy']

    with open(output_path / 'result.json', 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"\nSaved result.json to {output_path / 'result.json'}")

    # Detailed per-file predictions in a separate file
    with open(output_path / 'predictions.json', 'w') as f:
        json.dump({r['dataset_name']: r['results'] for r in all_results}, f, indent=2, default=str)

    print("\nGenerating plots...")
    plot_per_species(all_results, output_path)
    plot_dataset_comparison(all_results, output_path)

    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    total_files = sum(r['num_files'] for r in all_results)
    total_correct = sum(r['num_correct'] for r in all_results)
    for r in all_results:
        print(f"  {r['dataset_name']:30s}: {r['accuracy']:5.1f}%  ({r['num_correct']}/{r['num_files']})")
    if len(all_results) > 1:
        overall = 100.0 * total_correct / total_files if total_files else 0.0
        print(f"  {'Overall':30s}: {overall:5.1f}%  ({total_correct}/{total_files})")


if __name__ == '__main__':
    main()
