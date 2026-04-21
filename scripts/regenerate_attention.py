"""Regenerate attention visualizations from a saved checkpoint."""

import argparse
import json
import os
from pathlib import Path

import torch

from src.core.models import AST, RegNetModel
from src.data.data_utils import DataLoader, SpectrogramDataset
from src.evaluation.attention_viz import visualize_attention
from src.core import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate attention visualizations without retraining"
    )
    parser.add_argument(
        "run_dir",
        type=str,
        help="Directory containing the saved model checkpoint and model config JSON",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Dataset folder with labels.json and data/. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Optional explicit path to model config JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for loading dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of samples to visualize per dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--per-chunk-norm",
        action="store_true",
        help="Enable AST per-chunk normalization when rebuilding the model",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=2,
        help="Number of chunks for AST per-chunk normalization",
    )
    parser.add_argument(
        "--use-adapters",
        action="store_true",
        help="Enable AST adapters when rebuilding the model",
    )
    return parser.parse_args()


def find_existing_path(candidates, description):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {description}. Checked: {', '.join(str(path) for path in candidates)}")


def resolve_artifacts(run_dir, model_path, config_path):
    run_dir = Path(run_dir)

    if config_path is None:
        config_path = find_existing_path(
            [
                run_dir / 'ast_model_config.json',
                run_dir / 'regnet_model_config.json',
            ],
            'model config JSON',
        )
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as handle:
        model_config = json.load(handle)

    model_type = model_config.get('model_type', 'AST').lower()

    if model_path is None:
        if model_type == 'ast':
            model_path = find_existing_path(
                [run_dir / 'ast_model_best.pt', run_dir / 'ast_model.pt'],
                'AST checkpoint',
            )
        elif model_type == 'regnet':
            model_path = find_existing_path(
                [run_dir / 'regnet_model_best.pt', run_dir / 'regnet_model.pt'],
                'RegNet checkpoint',
            )
        else:
            raise ValueError(f"Unsupported model type in config: {model_type}")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    return run_dir, model_path, config_path, model_config, model_type


def build_model(model_config, model_type, args, device):
    num_classes = model_config['num_classes']
    freq_bins = model_config.get('freq_bins', config.SPECTROGRAM_PARAMS['nfilters'])
    time_bins = model_config.get('time_bins', config.DEFAULT_TIME_BINS)

    if model_type == 'ast':
        model = AST(
            num_classes,
            input_size=(freq_bins, time_bins),
            dropout=0.0,
            use_reconstruction=model_config.get('use_reconstruction', False),
            use_adapters=args.use_adapters,
            per_chunk_norm=args.per_chunk_norm,
            num_chunks=args.num_chunks,
        )
    elif model_type == 'regnet':
        model = RegNetModel(
            num_classes,
            pretrained_path=None,
            model_name=model_config.get('model_name', 'regnety_008'),
            freeze_backbone=False,
            freeze_stages=0,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)
    return model, (freq_bins, time_bins)


def load_checkpoint(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: missing keys when loading checkpoint: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected_keys[:10]}")

    model.eval()


def build_dataloader(dataset_path, image_size, model_config, batch_size):
    dataset_loader = DataLoader(dataset_path, noise_folder=None)
    dataset_data = dataset_loader.load_data(use_multilabel=True, validation_share=0.0)

    dataset = SpectrogramDataset(
        dataset_data['train_filenames'],
        dataset_data['train_labels'],
        image_size[0],
        image_size[1],
        config.DEFAULT_CHANNELS,
        'center',
        noise_filenames=None,
        noise_ratio=0.0,
        spec_transform=model_config.get('spec_transform', config.DEFAULT_SPEC_TRANSFORM),
        training=False,
        width_downsizing=None,
        bg_subtract=model_config.get('bg_subtract', False),
        median_filter=model_config.get('median_filter', False),
        use_temporal_roll=False,
        noise_mode='full',
        background_prob=0.0,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset_data


def dataset_output_folder(run_dir, dataset_path):
    test_name = Path(dataset_path).parent.name
    return run_dir / f'attention_{test_name}'


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    run_dir, model_path, config_path, model_config, model_type = resolve_artifacts(
        args.run_dir,
        args.model_path,
        args.config_path,
    )

    print(f"Run directory: {run_dir}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {model_path}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")

    model, image_size = build_model(model_config, model_type, args, device)
    load_checkpoint(model, model_path, device)

    for dataset_path in args.datasets:
        output_folder = dataset_output_folder(run_dir, dataset_path)
        dataloader, dataset_data = build_dataloader(
            dataset_path,
            image_size,
            model_config,
            args.batch_size,
        )
        print(f"Regenerating attention for {dataset_path} -> {output_folder}")
        visualize_attention(
            model,
            dataloader,
            str(output_folder),
            model_type=model_type,
            num_samples=args.num_samples,
            device=device,
            class_names=dataset_data['class_names'],
        )


if __name__ == '__main__':
    main()