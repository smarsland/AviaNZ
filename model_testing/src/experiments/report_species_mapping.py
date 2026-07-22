"""
Report species mapping/keeping decisions for DOC and AviaNZ raw data.

Walks the same folders as build_large_datasets.py and, for every file and every
species label found, reports:
  - the raw species name / eBird code seen
  - whether it normalised to an eBird code
  - whether that code mapped to a common name (via DOC_bird_naming_map.csv)
  - the final label used, and
  - whether it was kept (and if not, why it was dropped).

Nothing is extracted or saved — this is a dry-run audit only.

Usage:
    python model_testing/src/experiments/report_species_mapping.py \\
        --doc-raw /path/to/NZBirds \\
        --avianz-raw /path/to/Joe_MoDone [--avianz-raw /another] \\
        --mapping data/DOC_bird_naming_map.csv \\
        --out mapping_report.csv
"""

import argparse
import csv
import os

from model_testing.src.data.dataset_builder import AviaNZDataProcessor
from model_testing.src.experiments.analyze_dataset_quality import (
    load_bird_name_mapping,
    norm_key,
)
from model_testing.src.experiments.build_large_datasets import (
    _SKIP_SPECIES,
    scan_doc_by_species,
)
from model_testing.src.experiments.build_matched_datasets import load_avianz_name_mapping


def report_doc(doc_raw, ebird_to_common, rows):
    species_files = scan_doc_by_species(doc_raw)
    print(f'\nDOC: {len(species_files)} eBird codes found in folder structure')
    for code, files in sorted(species_files.items()):
        common = ebird_to_common.get(norm_key(code))
        mapped = bool(common)
        label = norm_key(common) if mapped else ''
        kept = mapped
        reason = '' if kept else 'ebird code not in naming map'
        rows.append({
            'dataset': 'DOC',
            'file': '(folder)',
            'raw_species': code,
            'ebird_code': code,
            'common_name': common or '',
            'label': label,
            'n_files': len(files),
            'mapped': 'yes' if mapped else 'no',
            'kept': 'yes' if kept else 'no',
            'reason': reason,
        })
        flag = 'KEEP' if kept else 'DROP'
        print(f'  [{flag}] {code:14s} -> {label or "???":24s} ({len(files)} files)'
              f'{"" if kept else "  (" + reason + ")"}')


def report_avianz(avianz_raws, mapping_csv, ebird_to_common, rows):
    name_mapping = load_avianz_name_mapping(mapping_csv)
    proc = AviaNZDataProcessor(name_mapping=name_mapping)

    wav_files = []
    for raw in avianz_raws:
        wav_files.extend(proc.find_wav_files(raw))
    print(f'\nAviaNZ: scanning {len(wav_files)} wav files across '
          f'{len(avianz_raws)} folder(s)')

    for wav_file in sorted(wav_files):
        data_file = wav_file + '.data'
        if not os.path.exists(data_file):
            continue
        segments = proc.load_annotation_file(data_file)
        for seg in segments:
            for lab in seg.labels:
                raw = lab['species']
                certainty = lab['certainty']

                # Reproduce build_large_datasets.py's decisions in order.
                if certainty < 50:
                    mapped, kept, reason = 'n/a', 'no', f'certainty {certainty} < 50'
                    code = common = label = ''
                elif raw in _SKIP_SPECIES:
                    mapped, kept, reason = 'n/a', 'no', 'in skip list'
                    code = common = label = ''
                else:
                    code = proc.normalize_to_ebird(raw)
                    normalised = bool(code) and code != raw
                    common = ebird_to_common.get(norm_key(code)) if code else None
                    if not common:
                        mapped, kept = 'no', 'no'
                        reason = ('name not normalised to an eBird code'
                                  if not normalised else
                                  'eBird code not in naming map')
                        label = ''
                    else:
                        mapped, kept, reason = 'yes', 'yes', ''
                        label = norm_key(common)

                rows.append({
                    'dataset': 'AviaNZ',
                    'file': os.path.relpath(wav_file),
                    'raw_species': raw,
                    'ebird_code': code or '',
                    'common_name': common or '',
                    'label': label,
                    'n_files': 1,
                    'mapped': mapped,
                    'kept': kept,
                    'reason': reason,
                })


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--doc-raw', default=None)
    p.add_argument('--avianz-raw', default=None, action='append')
    p.add_argument('--mapping', default='data/DOC_bird_naming_map.csv')
    p.add_argument('--out', default='mapping_report.csv')
    args = p.parse_args()

    if not args.doc_raw and not args.avianz_raw:
        p.error('provide --doc-raw and/or --avianz-raw')

    ebird_to_common, _ = load_bird_name_mapping(args.mapping)

    rows = []
    if args.doc_raw:
        report_doc(args.doc_raw, ebird_to_common, rows)
    if args.avianz_raw:
        report_avianz(args.avianz_raw, args.mapping, ebird_to_common, rows)

    fieldnames = ['dataset', 'file', 'raw_species', 'ebird_code', 'common_name',
                  'label', 'n_files', 'mapped', 'kept', 'reason']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    kept = sum(1 for r in rows if r['kept'] == 'yes')
    print(f'\n{len(rows)} (file, species) rows — {kept} kept, {len(rows) - kept} dropped')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
