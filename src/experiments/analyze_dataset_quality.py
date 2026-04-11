import pandas as pd
import numpy as np
import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def norm_key(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    text = str(text).strip().lower()
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    text = ' '.join(text.split())
    return text

def load_bird_name_mapping(csv_path):
    df = pd.read_csv(csv_path)
    ebird_to_common = {}
    common_to_ebird = {}
    for _, row in df.iterrows():
        ebird = row['eBird']
        common = row['CommonName']
        extra = row['ExtraName']
        doc = row['ListDOCBirds']

        if pd.notna(common):
            common = str(common).strip()
        
        ebird_to_common[norm_key(ebird)] = common
        common_to_ebird[norm_key(common)] = norm_key(ebird)
        if pd.notna(extra):
            common_to_ebird[norm_key(extra)] = norm_key(ebird)
        if pd.notna(doc):
            common_to_ebird[norm_key(doc)] = norm_key(ebird)

        if pd.notna(common):
            common_to_ebird[norm_key(common).replace(' ', '')] = norm_key(ebird)
        if pd.notna(extra):
            common_to_ebird[norm_key(extra).replace(' ', '')] = norm_key(ebird)
        if pd.notna(doc):
            common_to_ebird[norm_key(doc).replace(' ', '')] = norm_key(ebird)
    
    common_to_ebird['kaka'] = 'nezkak1'
    common_to_ebird['kereru'] = 'nezpig2'
    common_to_ebird['kakariki'] = 'parake'
    
    return ebird_to_common, common_to_ebird

def build_group_cache(ebird_to_common):
    groups = {}
    for group in ['kiwi', 'robin', 'fantail', 'parakeet']:
        codes = [
            code for code, common in ebird_to_common.items()
            if group in norm_key(common)
        ]
        if codes:
            groups[group] = sorted(set(codes))
    if 'parakeet' in groups:
        groups['kakariki'] = groups['parakeet']
    return groups


def normalize_species_name_to_codes(name, common_to_ebird, ebird_to_common, group_cache):
    if pd.isna(name) or name == '':
        return []

    name_raw = str(name)
    name_key = norm_key(name_raw)

    non_birds = {
        'nothing',
        'at least 2 more species',
        'poor quality',
        'not a bird call',
        'fly',
        'very feint',
        'very faint',
        'short',
        'very short',
        'kereru flight sound as well',
        'cicada',
        'dog',
        'tree frog',
        'weta',
        'kereru flight',
        'korora',
    }

    if not name_key or name_key in non_birds:
        return []

    if name_key.startswith('?'):
        return []

    if name_key.startswith('unknown'):
        return []

    if '/' in name_key:
        codes = []
        for part in [p.strip() for p in name_key.split('/') if p.strip()]:
            codes.extend(normalize_species_name_to_codes(part, common_to_ebird, ebird_to_common, group_cache))
        return list(dict.fromkeys(codes))

    synonyms = {
        'tui/bellbird': ['tui', 'bellbird'],
        'nz kingfisher': 'new zealand kingfisher',
        'unknown high-pitched call': None,
        'unknown high-pitch call': None,
    }

    if name_key in synonyms:
        mapped = synonyms[name_key]
        if mapped is None:
            return []
        if isinstance(mapped, list):
            codes = []
            for item in mapped:
                codes.extend(normalize_species_name_to_codes(item, common_to_ebird, ebird_to_common, group_cache))
            return list(dict.fromkeys(codes))
        return normalize_species_name_to_codes(mapped, common_to_ebird, ebird_to_common, group_cache)

    if name_key in group_cache:
        return list(group_cache[name_key])

    if name_key in ebird_to_common:
        return [name_key]

    if name_key in common_to_ebird:
        return [common_to_ebird[name_key]]

    no_space = name_key.replace(' ', '')
    if no_space in common_to_ebird:
        return [common_to_ebird[no_space]]

    return []

def parse_species_list_to_codes(species_str, common_to_ebird, ebird_to_common, group_cache):
    if pd.isna(species_str) or species_str == '':
        return []
    
    species_str = species_str.replace('"', '')
    parts = [p.strip() for p in species_str.replace(';', ',').split(',')]
    
    codes = []
    for part in parts:
        codes.extend(normalize_species_name_to_codes(part, common_to_ebird, ebird_to_common, group_cache))

    return list(dict.fromkeys(codes))


def parse_species_list_to_codes_with_unmapped(species_str, common_to_ebird, ebird_to_common, group_cache):
    if pd.isna(species_str) or species_str == '':
        return [], []

    species_str = str(species_str).replace('"', '')
    parts = [p.strip() for p in species_str.replace(';', ',').split(',')]

    codes = []
    unmapped = []
    for part in parts:
        if not part:
            continue

        part_key = norm_key(part)
        part_codes = normalize_species_name_to_codes(part, common_to_ebird, ebird_to_common, group_cache)
        if part_codes:
            codes.extend(part_codes)
            continue

        if not part_key:
            continue
        if part_key.startswith('?') or part_key.startswith('unknown'):
            continue

        non_birds = {
            'nothing',
            'at least 2 more species',
            'poor quality',
            'not a bird call',
            'fly',
            'very feint',
            'very faint',
            'short',
            'very short',
            'kereru flight sound as well',
            'cicada',
            'dog',
            'tree frog',
            'weta',
            'kereru flight',
            'korora',
        }
        if part_key in non_birds:
            continue

        unmapped.append(part_key)

    return list(dict.fromkeys(codes)), unmapped

def extract_predicted_label(row, df_columns):
    if 'Label' in df_columns:
        value = row.get('Label')
    elif 'predicted_label' in df_columns:
        value = row.get('predicted_label')
    elif 'Unnamed: 3' in df_columns:
        value = row.get('Unnamed: 3')
    elif 'File' in df_columns:
        file_idx = list(df_columns).index('File')
        value = row.iloc[file_idx - 1] if file_idx > 0 else None
    else:
        value = None

    if pd.isna(value) or value == '':
        return None

    value = str(value).strip().lower()
    if '/' in value:
        value = value.split('/')[-1]
    if value.endswith('.wav') or value.endswith('.flac'):
        value = Path(value).stem
    return value

def is_poor_quality(note):
    if pd.isna(note):
        return False
    note_lower = note.lower()
    quality_markers = ['poor quality', 'nothing', 'not a bird call', 'very short', 
                       'very feint', 'short', 'rain']
    return any(marker in note_lower for marker in quality_markers)

def plot_confusion_matrix(confusion_df, ebird_to_common, output_dir, min_samples=5):
    if len(confusion_df) == 0:
        return

    # Build per-predicted total to filter low-sample species
    totals = confusion_df.groupby('predicted_code')['count'].sum()
    keep = totals[totals >= min_samples].index.tolist()
    sub = confusion_df[confusion_df['predicted_code'].isin(keep)].copy()

    wide = sub.pivot_table(
        index='predicted_code', columns='actual_code',
        values='count', aggfunc='sum', fill_value=0
    )
    # Keep only columns that are also rows (predicted species)
    all_codes = list(wide.index)
    wide = wide.reindex(columns=all_codes, fill_value=0)

    row_sums = wide.sum(axis=1)
    norm = wide.div(row_sums, axis=0)

    labels = [ebird_to_common.get(c, c) for c in all_codes]
    n = len(labels)
    cell = max(0.45, min(0.85, 16 / n))
    fig, ax = plt.subplots(figsize=(n * cell + 2.5, n * cell + 1.5))

    im = ax.imshow(norm.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Human label (actual)', labelpad=8)
    ax.set_ylabel('Dataset label (predicted)', labelpad=8)
    ax.set_title('Confusion matrix (row-normalised, n≥{})'.format(min_samples))

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = norm.values[i, j]
            cnt = wide.values[i, j]
            if cnt == 0:
                continue
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, f'{val:.2f}\n({cnt})', ha='center', va='center',
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Fraction of predictions')
    plt.tight_layout()
    out = Path(output_dir) / 'confusion_matrix.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix plot: {out}")


def plot_accuracy_bar(species_df, output_dir, min_samples=3):
    df = species_df[species_df['total'] >= min_samples].copy()
    df = df.sort_values('accuracy')

    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))

    cmap = plt.get_cmap('RdYlGn')
    colors = [cmap(v / 100) for v in df['accuracy']]

    bars = ax.barh(range(n), df['accuracy'], color=colors, edgecolor='grey', linewidth=0.4)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['accuracy'] + 0.5, i,
                f"{row['accuracy']:.1f}%  (n={row['total']})",
                va='center', fontsize=8)

    ax.set_yticks(range(n))
    ax.set_yticklabels(df['species'], fontsize=8)
    ax.set_xlim(0, 115)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title(f'Per-species accuracy (n≥{min_samples})')
    ax.axvline(df['accuracy'].mean(), color='steelblue', linestyle='--',
               linewidth=1, label=f"mean {df['accuracy'].mean():.1f}%")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = Path(output_dir) / 'per_species_accuracy.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Per-species accuracy plot: {out}")


def analyze_dataset_quality(review_csv, mapping_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading bird name mapping from {mapping_csv}")
    ebird_to_common, common_to_ebird = load_bird_name_mapping(mapping_csv)
    group_cache = build_group_cache(ebird_to_common)
    
    print(f"Loading reviewed dataset from {review_csv}")
    df = pd.read_csv(review_csv)
    
    print(f"Total samples reviewed: {len(df)}")
    
    results = []
    confusion_data = defaultdict(lambda: defaultdict(int))
    species_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong': 0, 
                                         'poor_quality': 0, 'multi_species': 0})
    unmapped_labels = Counter()
    
    for idx, row in df.iterrows():
        predicted_label = extract_predicted_label(row, df.columns)
        
        species1_raw = row['Species 1'] if pd.notna(row['Species 1']) else ''
        species2_raw = row['Species 2+'] if pd.notna(row['Species 2+']) else ''
        
        species1_codes, unmapped1 = parse_species_list_to_codes_with_unmapped(
            species1_raw, common_to_ebird, ebird_to_common, group_cache
        )
        species2_codes, unmapped2 = parse_species_list_to_codes_with_unmapped(
            species2_raw, common_to_ebird, ebird_to_common, group_cache
        )
        for token in unmapped1 + unmapped2:
            unmapped_labels[token] += 1
        actual_codes = list(dict.fromkeys(species1_codes + species2_codes))
        actual_species = [ebird_to_common.get(code, code) for code in actual_codes]
        
        poor_quality = is_poor_quality(row['Note'])
        
        predicted_name = None
        predicted_code = None
        if predicted_label:
            predicted_code = predicted_label
            predicted_name = ebird_to_common.get(predicted_code, predicted_code)
        
        is_correct = False
        
        is_multi_species = len(species2_codes) > 0

        if predicted_code and predicted_code in actual_codes:
            is_correct = True
        
        result = {
            'file': row.get('File', ''),
            'predicted_label': predicted_label,
            'predicted_code': predicted_code,
            'predicted_name': predicted_name,
            'actual_species': actual_species,
            'actual_codes': actual_codes,
            'is_correct': is_correct,
            'poor_quality': poor_quality,
            'multi_species': is_multi_species,
            'note': row['Note']
        }
        results.append(result)
        
        if predicted_code:
            species_stats[predicted_code]['total'] += 1
            
            if poor_quality:
                species_stats[predicted_code]['poor_quality'] += 1
            
            if is_multi_species:
                species_stats[predicted_code]['multi_species'] += 1
            
            if is_correct:
                species_stats[predicted_code]['correct'] += 1
                confusion_data[predicted_code][predicted_code] += 1
            else:
                species_stats[predicted_code]['wrong'] += 1
                for actual_code in actual_codes:
                    confusion_data[predicted_code][actual_code] += 1
    
    print("\n=== OVERALL STATISTICS ===")
    total_samples = len(results)
    correct_samples = sum(1 for r in results if r['is_correct'])
    poor_quality_samples = sum(1 for r in results if r['poor_quality'])
    multi_species_samples = sum(1 for r in results if r['multi_species'])
    no_species_samples = sum(1 for r in results if len(r['actual_species']) == 0)
    
    accuracy = correct_samples / total_samples * 100 if total_samples > 0 else 0
    
    print(f"Total samples: {total_samples}")
    print(f"Correct labels: {correct_samples} ({accuracy:.1f}%)")
    print(f"Incorrect labels: {total_samples - correct_samples} ({100-accuracy:.1f}%)")
    print(f"Poor quality: {poor_quality_samples} ({poor_quality_samples/total_samples*100:.1f}%)")
    print(f"Multi-species: {multi_species_samples} ({multi_species_samples/total_samples*100:.1f}%)")
    print(f"No species found: {no_species_samples} ({no_species_samples/total_samples*100:.1f}%)")
    
    print("\n=== PER-SPECIES STATISTICS ===")
    species_report = []
    for species_code in sorted(species_stats.keys()):
        stats = species_stats[species_code]
        species_name = ebird_to_common.get(species_code, species_code)
        acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        species_report.append({
            'species': species_name,
            'species_code': species_code,
            'total': stats['total'],
            'correct': stats['correct'],
            'wrong': stats['wrong'],
            'accuracy': acc,
            'poor_quality': stats['poor_quality'],
            'multi_species': stats['multi_species']
        })
        print(f"{species_name:30s} | Total: {stats['total']:4d} | Acc: {acc:5.1f}% | "
              f"Poor: {stats['poor_quality']:3d} | Multi: {stats['multi_species']:3d}")
    
    print("\n=== CONFUSION ANALYSIS ===")
    confusion_report = []
    for predicted_code in sorted(confusion_data.keys()):
        actual_counts = confusion_data[predicted_code]
        total_pred = sum(actual_counts.values())
        predicted_name = ebird_to_common.get(predicted_code, predicted_code)
        
        print(f"\n{predicted_name} (n={total_pred}):")
        for actual, count in sorted(actual_counts.items(), key=lambda x: -x[1]):
            pct = count / total_pred * 100
            actual_name = ebird_to_common.get(actual, actual)
            confusion_report.append({
                'predicted': predicted_name,
                'predicted_code': predicted_code,
                'actual': actual_name,
                'actual_code': actual,
                'count': count,
                'percentage': pct
            })
            marker = "✓" if predicted_code == actual else "✗"
            print(f"  {marker} {actual_name:30s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n=== SAVING RESULTS ===")
    
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'detailed_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Detailed results: {results_path}")
    
    species_df = pd.DataFrame(species_report)
    species_path = output_dir / 'per_species_accuracy.csv'
    species_df.to_csv(species_path, index=False)
    print(f"Per-species stats: {species_path}")
    
    confusion_df = pd.DataFrame(confusion_report)
    confusion_path = output_dir / 'confusion_matrix.csv'
    confusion_df.to_csv(confusion_path, index=False)
    print(f"Confusion data: {confusion_path}")

    if len(confusion_df) > 0:
        wide = confusion_df.pivot_table(
            index='predicted',
            columns='actual',
            values='count',
            aggfunc='sum',
            fill_value=0,
        )
        wide_path = output_dir / 'confusion_matrix_wide.csv'
        wide.to_csv(wide_path)
        print(f"Confusion matrix (wide): {wide_path}")

    print("\n=== PLOTS ===")
    plot_confusion_matrix(confusion_df, ebird_to_common, output_dir)
    plot_accuracy_bar(pd.DataFrame(species_report), output_dir)
    
    summary = {
        'total_samples': total_samples,
        'correct_samples': correct_samples,
        'accuracy': accuracy,
        'poor_quality_samples': poor_quality_samples,
        'multi_species_samples': multi_species_samples,
        'no_species_samples': no_species_samples,
        'species_stats': {k: dict(v) for k, v in species_stats.items()}
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    if len(unmapped_labels) > 0:
        unmapped_df = pd.DataFrame(
            [{'label': k, 'count': v} for k, v in unmapped_labels.most_common()]
        )
        unmapped_path = output_dir / 'unmapped_human_labels.csv'
        unmapped_df.to_csv(unmapped_path, index=False)
        print(f"Unmapped human labels: {unmapped_path}")
    
    print("\n=== RECOMMENDATIONS ===")
    print(f"1. Consider excluding {poor_quality_samples} poor quality samples")
    print(f"2. Multi-label handling needed for {multi_species_samples} samples")
    print(f"3. Clean/remove {no_species_samples} samples with no bird calls")
    
    usable_samples = total_samples - poor_quality_samples - no_species_samples
    print(f"4. Usable samples (excluding poor/empty): {usable_samples} ({usable_samples/total_samples*100:.1f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze dataset quality from human review')
    parser.add_argument('review_csv', nargs='?', default='data/doc_reviewed.csv',
                       help='Path to reviewed dataset CSV (default: data/doc_reviewed.csv)')
    parser.add_argument('--mapping', default='data/DOC_bird_naming_map.csv',
                       help='Path to bird name mapping CSV')
    parser.add_argument('--output', default='dataset_quality_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyze_dataset_quality(args.review_csv, args.mapping, args.output)
