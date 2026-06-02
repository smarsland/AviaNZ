"""
Fine-tune the pre-trained Kaytoo model on a matched/corrected training set and
then evaluate it on the test sets, producing outputs in the same format as
evaluate_kaytoo.py so that plain Kaytoo and fine-tuned Kaytoo can be compared
directly with analyze_all_results.py.

Each training split folder must contain:
  labels.json   - with class_names field per sample (same format as test sets)
  audio/        - .wav files named file_XXXXXXXX.wav

Usage:
    python scripts/finetune_kaytoo.py \\
        /path/to/avianz_split/train /path/to/avianz_split/test \\
        /path/to/doc_split/train    /path/to/doc_split/test \\
        --kaytoo-root /path/to/Kaytoo \\
        --mapping     data/DOC_bird_naming_map.csv \\
        --output      results/kaytoo_finetuned_seed0 \\
        [--epochs 10] [--lr 1e-4] [--batch-size 16] [--num-workers 4] [--cpu]
"""

import ast
import csv
import json
import os
import sys
import argparse
import gc
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Shared helpers (duplicated from evaluate_kaytoo.py to keep scripts self-
# contained; keep in sync if the evaluation logic changes).
# ---------------------------------------------------------------------------

COMBINED_CLASSES = {
    'tui/bellbird': frozenset({'tui1', 'nezbel1'}),
}


def build_label_to_ebird(mapping_csv):
    """Build lowercase-normalised label string -> eBird code from DOC naming map."""
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
    for candidate in [
        Path(kaytoo_root) / 'resources' / 'bird_map.csv',
        Path(kaytoo_root) / 'bird_map.csv',
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"bird_map.csv not found under {kaytoo_root}")


def load_labels(folder):
    with open(Path(folder) / 'labels.json') as f:
        return json.load(f)


def collect_audio_files(folder):
    audio_dir = Path(folder) / 'audio'
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"No audio/ subfolder in {folder}. "
            "Run build_matched_datasets.py with --with-audio first."
        )
    return sorted(audio_dir.glob('*.wav'))


def norm_label(label):
    return label.strip().lower().replace(' / ', '/').replace('/ ', '/').replace(' /', '/')


# ---------------------------------------------------------------------------
# Build the training DataFrame that Kaytoo's WaveformDataset expects.
#
# Column layout:  filepath | primary_label | centres | <ebird_code1> | <ebird_code2> ...
# (columns 0-2 are metadata; columns 3+ are one-hot/multi-hot targets)
# ---------------------------------------------------------------------------

def build_training_df(train_folder, label_to_ebird):
    """Convert a labels.json + audio/ folder into a Kaytoo training DataFrame."""
    labels_data = load_labels(train_folder)
    audio_dir = Path(train_folder) / 'audio'

    rows = []
    for item in labels_data.get('files', []):
        wav_name = item['filename'].replace('.npy', '.wav')
        wav_path = audio_dir / wav_name
        if not wav_path.exists():
            print(f"  [WARN] Missing audio file: {wav_path}, skipping.")
            continue

        class_names = item.get('class_names', [])
        ebird_codes = set()
        for lbl in class_names:
            lbl_norm = norm_label(lbl)
            if lbl_norm in COMBINED_CLASSES:
                ebird_codes |= COMBINED_CLASSES[lbl_norm]
            else:
                code = label_to_ebird.get(lbl_norm)
                if code:
                    ebird_codes.add(code)

        if not ebird_codes:
            # background / unlabelled sample — skip during fine-tuning
            continue

        rows.append({
            'filepath': str(wav_path),
            'primary_label': sorted(ebird_codes)[0],   # pick one as primary
            'secondary_labels': [c for c in sorted(ebird_codes)[1:]],
            'centres': [],
            '_all_codes': sorted(ebird_codes),
        })

    if not rows:
        raise ValueError(f"No usable labelled samples found in {train_folder}")

    df = pd.DataFrame(rows)
    return df


def encode_training_df(df, unique_birds):
    """Add one-hot label columns and drop helper columns.

    Output column order: filepath | primary_label | centres | <bird cols ...>
    This matches what WaveformDataset expects (cols[3:] = class targets).
    """
    # Build multi-hot matrix
    label_matrix = np.zeros((len(df), len(unique_birds)), dtype=np.uint8)
    bird_idx = {b: i for i, b in enumerate(unique_birds)}
    for row_i, codes in enumerate(df['_all_codes']):
        for code in codes:
            if code in bird_idx:
                label_matrix[row_i, bird_idx[code]] = 1

    label_df = pd.DataFrame(label_matrix, columns=unique_birds, index=df.index)
    keep = df[['filepath', 'primary_label', 'centres']].copy()
    result = pd.concat([keep, label_df], axis=1)
    return result


# ---------------------------------------------------------------------------
# Evaluation (identical logic to evaluate_kaytoo.py)
# ---------------------------------------------------------------------------

def aggregate_to_file(pred_df):
    species_cols = [c for c in pred_df.columns if c not in ('row_id', 'File_Path')]
    per_file = pred_df.groupby('File_Path')[species_cols].max().reset_index()
    return per_file, species_cols


def evaluate_folder(test_folder, dataset_name, models, label_to_ebird, threshold=0.5):
    from kaytoo_infer import inference as kaytoo_inference

    labels_data = load_labels(test_folder)
    audio_files = collect_audio_files(test_folder)

    name_to_meta = {
        item['filename'].replace('.npy', '.wav'): item
        for item in labels_data.get('files', [])
    }

    test_ebird_codes_ordered = []
    seen = set()
    for item in labels_data.get('files', []):
        for l in item.get('class_names', []):
            l_norm = norm_label(l)
            if l_norm in COMBINED_CLASSES:
                for code in COMBINED_CLASSES[l_norm]:
                    if code not in seen:
                        seen.add(code)
                        test_ebird_codes_ordered.append(code)
            else:
                code = label_to_ebird.get(l_norm)
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

    dataset_class_names = sorted({
        cls
        for item in labels_data.get('files', [])
        for cls in item.get('class_names', [])
    })
    cls_to_ebird_codes = {}
    for cls in dataset_class_names:
        cls_norm = norm_label(cls)
        if cls_norm in COMBINED_CLASSES:
            cls_to_ebird_codes[cls] = list(COMBINED_CLASSES[cls_norm])
        else:
            code = label_to_ebird.get(cls_norm)
            cls_to_ebird_codes[cls] = [code] if code else []

    missing = set(test_ebird_codes_ordered) - set(species_cols)
    if missing:
        print(f"  WARNING: {len(missing)} test-set species not in model vocab: {sorted(missing)}")
    if not valid_cols:
        print("  ERROR: no overlap between test-set species and model vocab")
        return None
    print(f"  Scoring over {len(valid_cols)} species (threshold={threshold})")

    results = []
    raw_score_records = []
    for _, row in per_file_df.iterrows():
        wav_name = Path(row['File_Path']).name
        meta = name_to_meta.get(wav_name)
        if meta is None:
            continue

        gt_labels = meta.get('class_names', [meta.get('label')])
        gt_labels = [l for l in gt_labels if l]

        gt_slots = []
        all_acceptable_codes = set()
        for l in gt_labels:
            l_norm = norm_label(l)
            if l_norm in COMBINED_CLASSES:
                slot = frozenset(c for c in COMBINED_CLASSES[l_norm] if c in set(valid_cols))
            else:
                code = label_to_ebird.get(l_norm)
                slot = frozenset([code]) if code and code in set(valid_cols) else frozenset()
            if slot:
                gt_slots.append(slot)
                all_acceptable_codes |= slot

        scores = row[valid_cols].values
        pred_codes = {valid_cols[i] for i, s in enumerate(scores) if s > threshold}

        if gt_slots:
            slots_satisfied = all(pred_codes & slot for slot in gt_slots)
            no_false_positives = all(code in all_acceptable_codes for code in pred_codes)
            correct = slots_satisfied and no_false_positives
        else:
            correct = len(pred_codes) == 0

        gt_codes_flat = sorted({code for slot in gt_slots for code in slot})
        results.append({
            'wav_file': wav_name,
            'gt_codes': gt_codes_flat,
            'pred_codes': sorted(pred_codes),
            'correct': correct,
        })

        npy_name = wav_name.replace('.wav', '.npy')
        npy_path = str(Path(test_folder) / 'data' / npy_name)
        raw_rec = {'filename': npy_path, 'gt_classes': gt_labels}
        for cls in dataset_class_names:
            codes = cls_to_ebird_codes.get(cls, [])
            cls_scores = [float(row[c]) for c in codes if c in row.index and not pd.isna(row[c])]
            raw_rec[cls] = max(cls_scores) if cls_scores else 0.0
        raw_score_records.append(raw_rec)

    n = len(results)
    n_correct = sum(r['correct'] for r in results)
    accuracy = 100.0 * n_correct / n if n else 0.0
    print(f"  Accuracy (all):      {n_correct}/{n} = {accuracy:.1f}%")

    labelled = [r for r in results if r['gt_codes']]
    n_labelled = len(labelled)
    n_labelled_correct = sum(r['correct'] for r in labelled)
    accuracy_labelled = 100.0 * n_labelled_correct / n_labelled if n_labelled else float('nan')
    print(f"  Accuracy (labelled): {n_labelled_correct}/{n_labelled} = {accuracy_labelled:.1f}%")

    background = [r for r in results if not r['gt_codes']]
    n_background = len(background)
    n_background_correct = sum(r['correct'] for r in background)
    accuracy_background = 100.0 * n_background_correct / n_background if n_background else float('nan')
    print(f"  Accuracy (background): {n_background_correct}/{n_background} = {accuracy_background:.1f}%")

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
        'accuracy_labelled': accuracy_labelled,
        'accuracy_background': accuracy_background,
        'num_labelled': n_labelled,
        'num_background': n_background,
        'species_stats': {k: dict(v) for k, v in species_stats.items()},
        'results': results,
        'dataset_class_names': dataset_class_names,
        'raw_score_records': raw_score_records,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Kaytoo on a matched training set and evaluate.')
    parser.add_argument('--doc-train', required=True,
                        help='DOC training split folder (labels.json + audio/)')
    parser.add_argument('--avianz-test', required=True,
                        help='AviaNZ test split folder')
    parser.add_argument('--doc-test', required=True,
                        help='DOC test split folder')
    parser.add_argument('--kaytoo-root', required=True,
                        help='Root of the Kaytoo installation')
    parser.add_argument('--mapping', default='data/DOC_bird_naming_map.csv',
                        help='DOC bird naming map CSV')
    parser.add_argument('--output', required=True,
                        help='Folder for checkpoints, model artefacts and results')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of fine-tuning epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size (default: 16)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Fraction of training data held out for validation (default: 0.1)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training/inference')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs to use for training (default: 1)')
    parser.add_argument('--gpu-id', type=int, default=None,
                        help='GPU index to use (sets CUDA_VISIBLE_DEVICES; default: let PyTorch pick)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Score threshold for evaluation (default: 0.5)')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_path / 'training_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir = output_path / 'deploy'
    deploy_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Set up Kaytoo imports
    # ------------------------------------------------------------------
    kaytoo_root = str(Path(args.kaytoo_root).resolve())
    sys.path.insert(0, kaytoo_root)

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    from bird_naming_utils import BirdNamer
    from kaytoo_infer import DefaultConfig, ModelParameters, Models
    import kaytoo_train_2_07 as _kt
    from kaytoo_train_2_07 import (
        AudioConfig,
        TrainingParameters,
        BirdData,
        WaveformDataset,
        get_dataloaders,
        run_training,
        load_pt_model,
        save_model_config,
    )

    # Pin GPU / disable GPU before importing torch so CUDA sees the right device.
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Patch pl.Trainer in kaytoo's own namespace so that DDP always uses
    # find_unused_parameters=True when running multi-GPU.  For a single device
    # we use strategy='auto' so PL picks SingleDeviceStrategy — that avoids
    # NCCL / process-group init entirely.  We must NOT touch kaytoo_train_2_07.py.
    _n_devices = args.devices
    _OrigTrainer = pl.Trainer
    class _FTTrainer(_OrigTrainer):
        def __init__(self, *args, **kwargs):
            n = kwargs.get('devices', _n_devices)
            if isinstance(n, int) and n > 1:
                kwargs.setdefault('strategy', 'ddp_find_unused_parameters_true')
            kwargs.setdefault('devices', _n_devices)
            super().__init__(*args, **kwargs)
    _kt.pl.Trainer = _FTTrainer

    # ------------------------------------------------------------------
    # Labels / mapping
    # ------------------------------------------------------------------
    label_to_ebird = build_label_to_ebird(args.mapping)
    bird_map_path = find_bird_map(kaytoo_root)
    bird_map_df = pd.read_csv(bird_map_path)
    birdnames = BirdNamer(bird_map_df)

    # ------------------------------------------------------------------
    # Build training DataFrame from corrected DOC labels only
    # ------------------------------------------------------------------
    print("\n=== Preparing training data ===")
    combined_df = build_training_df(args.doc_train, label_to_ebird)
    print(f"  DOC train:    {len(combined_df)} samples")

    # Collect all eBird codes present in the training data
    all_codes = sorted({code for codes in combined_df['_all_codes'] for code in codes})
    print(f"  Unique species: {len(all_codes)} -> {all_codes}")

    # Encode to one-hot
    train_encoded = encode_training_df(combined_df, all_codes)

    # Train / val split
    val_n = max(1, int(len(train_encoded) * args.val_fraction))
    val_df   = train_encoded.sample(n=val_n, random_state=42)
    train_df = train_encoded.drop(val_df.index).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"  Train rows: {len(train_df)},  Val rows: {len(val_df)}")

    # ------------------------------------------------------------------
    # Kaytoo configs
    # ------------------------------------------------------------------
    audio_cfg = AudioConfig()

    # Minimal TrainingParameters that mirror AudioConfig defaults but allow
    # our CLI overrides for LR / epochs / batch size.
    train_cfg = TrainingParameters()
    train_cfg.TRAIN  = True
    train_cfg.EPOCHS = args.epochs
    train_cfg.LR     = args.lr
    train_cfg.INITIAL_LR = args.lr / 10.0
    train_cfg.MIN_LR     = args.lr / 100.0
    train_cfg.BATCH_SIZE = args.batch_size
    train_cfg.NUM_WORKERS = args.num_workers
    train_cfg.EPOCHS_TO_UNFREEZE_BACKBONE = max(1, args.epochs // 3)
    # Disable early-stopping resets for short fine-tuning runs
    train_cfg.RESET_EPOCH = args.epochs + 1
    train_cfg.PATIENCE    = args.epochs  # effectively disabled
    if args.cpu:
        train_cfg.GPU       = 'cpu'
        train_cfg.PRECISION = 32
        train_cfg.DEVICE    = torch.device('cpu')

    data_cfg = BirdData()

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    augmentation_updates = [
        train_cfg.FIRST_AUGMENTATION_UPDATE,
        train_cfg.SECOND_AUGMENTATION_UPDATE,
    ]
    dl_train, dl_val, ds_train, ds_val = get_dataloaders(
        train_df,
        val_df,
        None,          # train_df_short
        audio_cfg,
        batch_size=train_cfg.BATCH_SIZE,
        num_workers=train_cfg.NUM_WORKERS,
        augmentation_updates=augmentation_updates,
    )
    print(f"  DataLoaders ready ({len(ds_train)} train, {len(ds_val)} val).")

    # ------------------------------------------------------------------
    # Find the pretrained checkpoint from Kaytoo's models/ directory
    # ------------------------------------------------------------------
    models_dir = Path(kaytoo_root) / 'models'
    pt_files = list(models_dir.rglob('*.pt'))
    ckpt_files = list(models_dir.rglob('*.ckpt'))
    pretrained_path = None
    if ckpt_files:
        pretrained_path = str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
        print(f"  Using pretrained checkpoint: {pretrained_path}")
    elif pt_files:
        pretrained_path = str(max(pt_files, key=lambda p: p.stat().st_mtime))
        print(f"  Using pretrained .pt weights: {pretrained_path}")
    else:
        print("  WARNING: No pretrained model found in models/. Training from scratch.")

    # Temporary paths object (duck-typing FilePaths)
    class _Paths:
        out_dir   = results_dir
        temp_dir  = str(checkpoint_dir)
        chkpt_dir = str(checkpoint_dir)

    # Monkey-patch the global `paths` in kaytoo_train_2_07 so TrainingModel
    # can write val pickles to results_dir.
    import kaytoo_train_2_07 as _kt
    _kt.paths = _Paths()

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    print("\n=== Fine-tuning ===")
    metrics = run_training(
        dl_train,
        dl_val,
        data_cfg,
        train_cfg,
        audio_cfg,
        str(checkpoint_dir),
        None,            # background_df
        pretrained_path, # pretrained_path
    )

    # ------------------------------------------------------------------
    # Save the fine-tuned model weights in Kaytoo's deploy format so
    # inference can find them exactly as in the pretrained eval.
    # ------------------------------------------------------------------
    print("\n=== Saving fine-tuned model ===")
    ckpts = sorted(checkpoint_dir.glob('*.ckpt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        print("ERROR: No checkpoints found after training!")
        sys.exit(1)

    best_ckpt = ckpts[0]
    print(f"  Best checkpoint: {best_ckpt}")

    num_classes = len(all_codes)
    pt_model = load_pt_model(
        str(best_ckpt),
        train_cfg.BACKBONE_NAME,
        audio_cfg.IMAGE_SHAPE,
        num_classes,
    )
    deploy_pt = deploy_dir / 'finetuned_kaytoo.pt'
    torch.save(pt_model.state_dict(), deploy_pt)
    print(f"  Saved PT weights → {deploy_pt}")

    # Save bird map CSV (needed by inference)
    bird_map_deploy = deploy_dir / 'finetuned_kaytoo_bird_map.csv'
    # Build bird map with the exact columns BirdNamer expects:
    # CommonName, eBird, ScientificName, ExtraName, TrainSamples, ValSamples
    # Pull as much info as possible from the pretrained bird map; fall back to
    # the eBird code for anything that isn't in it.
    pretrained_bm = bird_map_df.set_index('eBird') if 'eBird' in bird_map_df.columns else pd.DataFrame()
    rows_out = []
    for code in all_codes:
        if not pretrained_bm.empty and code in pretrained_bm.index:
            row = pretrained_bm.loc[code]
            rows_out.append({
                'CommonName':     row.get('CommonName', code),
                'eBird':          code,
                'ScientificName': row.get('ScientificName', code),
                'ExtraName':      row.get('ExtraName', code),
                'TrainSamples':   0,
                'ValSamples':     0,
            })
        else:
            rows_out.append({
                'CommonName':     code,
                'eBird':          code,
                'ScientificName': code,
                'ExtraName':      code,
                'TrainSamples':   0,
                'ValSamples':     0,
            })
    pd.DataFrame(rows_out).to_csv(bird_map_deploy, index=False)
    print(f"  Saved bird map → {bird_map_deploy}")

    # Save model config YAML
    config_deploy = deploy_dir / 'finetuned_kaytoo_config.yaml'
    import yaml
    model_config = {
        'basename': train_cfg.BACKBONE_NAME,
        'image_shape': audio_cfg.IMAGE_SHAPE,
        'image_time': audio_cfg.DURATION,
        'n_mels': audio_cfg.N_MELS,
        'n_fft': audio_cfg.N_FFT,
        'double_audio': audio_cfg.DOUBLE_AUDIO,
        'buffer_audio': audio_cfg.BUFFER_AUDIO,
        'use_deltas': audio_cfg.USE_DELTAS,
        'hop_length': audio_cfg.HOP_LENGTH,
        'spec_width': audio_cfg.SPEC_WIDTH,
    }
    with open(config_deploy, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)
    print(f"  Saved config → {config_deploy}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Evaluate the fine-tuned model — load it via Kaytoo's standard
    # inference machinery pointing at the deploy/ subfolder.
    # ------------------------------------------------------------------
    print("\n=== Evaluating fine-tuned model ===")

    use_case_infer = {
        'project_root': kaytoo_root,
        'experiment': None,      # forces ModelParameters to scan models/
        'cpu_only': args.cpu,
        'num_cores': 1,
        'naming_scheme': 'eBird',
        '_override_deploy_dir': str(deploy_dir),  # used below
    }

    # We need inference to use the *fine-tuned* bird map, not the original one.
    # Build a temporary BirdNamer from the fine-tuned bird map.
    ft_birdnames = BirdNamer(pd.read_csv(bird_map_deploy))
    cfg_infer = DefaultConfig(bird_namer=ft_birdnames, options={
        'project_root': kaytoo_root,
        'cpu_only': args.cpu,
        'num_cores': 1,
        'naming_scheme': 'eBird',
    })

    # Build ModelParameters manually pointing at the deploy directory.
    class _FTModelParameters:
        def __init__(self):
            import yaml as _yaml
            with open(config_deploy, 'r') as f_cfg:
                mc = _yaml.load(f_cfg, Loader=_yaml.FullLoader)
            mc['pt_path'] = deploy_pt
            mc['pcen'] = False
            mc['ckpt_path'] = None
            self.parameters = [mc]

    ft_params = _FTModelParameters()
    ft_models = Models(config=cfg_infer, model_parameters=ft_params)

    all_results = []
    for test_folder, split_label in [
        (args.avianz_test, Path(args.avianz_test).resolve().parent.name),
        (args.doc_test,    Path(args.doc_test).resolve().parent.name),
    ]:
        print(f"\n{'='*60}")
        print(f"Dataset: {split_label}")
        print(f"{'='*60}")
        result = evaluate_folder(
            test_folder, split_label, ft_models, label_to_ebird,
            threshold=args.threshold,
        )
        if result:
            all_results.append(result)

    if not all_results:
        print("No evaluation results to report.")
        return

    # ------------------------------------------------------------------
    # Write result.json (same format as evaluate_kaytoo.py / trained models)
    # ------------------------------------------------------------------
    result_json = {
        'name': output_path.name,
        'type': 'finetuned',
        'model': 'kaytoo_finetuned',
        'seed': 0,
        'status': 'completed',
        'epochs': args.epochs,
        'lr': args.lr,
    }
    if len(all_results) >= 1:
        result_json['test1_name']           = all_results[0]['dataset_name']
        result_json['test1_acc']            = all_results[0]['accuracy']
        result_json['test1_acc_labelled']   = all_results[0].get('accuracy_labelled', float('nan'))
        result_json['test1_acc_background'] = all_results[0].get('accuracy_background', float('nan'))
    if len(all_results) >= 2:
        result_json['test2_name']           = all_results[1]['dataset_name']
        result_json['test2_acc']            = all_results[1]['accuracy']
        result_json['test2_acc_labelled']   = all_results[1].get('accuracy_labelled', float('nan'))
        result_json['test2_acc_background'] = all_results[1].get('accuracy_background', float('nan'))

    with open(output_path / 'result.json', 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"\nSaved result.json → {output_path / 'result.json'}")

    # ------------------------------------------------------------------
    # Save per-split raw score CSVs  (same format as evaluate_kaytoo.py)
    # ------------------------------------------------------------------
    for result in all_results:
        raw_records = result.get('raw_score_records', [])
        if not raw_records:
            continue
        split_name = result['dataset_name']
        csv_path = output_path / f'predictions_{split_name}.csv'
        class_cols = sorted(c for c in raw_records[0] if c not in ('filename', 'gt_codes', 'gt_classes'))
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename'] + class_cols + [f'true_{c}' for c in class_cols])
            for rec in raw_records:
                gt_set = set(rec.get('gt_classes', []))
                writer.writerow(
                    [rec['filename']]
                    + [f"{rec.get(c, 0.0):.6f}" for c in class_cols]
                    + [int(c in gt_set) for c in class_cols]
                )
        print(f"Saved {len(raw_records)} raw score rows → {csv_path.name}")

    # Detailed predictions JSON
    with open(output_path / 'predictions.json', 'w') as f:
        json.dump(
            [{k: v for k, v in r.items() if k != 'raw_score_records'} for r in all_results],
            f, indent=2,
        )
    print(f"Saved predictions.json → {output_path / 'predictions.json'}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {r['dataset_name']:30s}  acc={r['accuracy']:.1f}%"
              f"  labelled={r['accuracy_labelled']:.1f}%"
              f"  background={r['accuracy_background']:.1f}%")
    print("="*60)


if __name__ == '__main__':
    main()
