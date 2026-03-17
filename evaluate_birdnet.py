#!/usr/bin/env python3
"""
Evaluate BirdNET on test datasets and compare with ground truth.

This script:
1. Reads test datasets with audio/ subfolders (created by data_loader.py --with-audio)
2. Runs BirdNET predictions on each audio file
3. Compares BirdNET predictions with ground truth labels
4. Generates detailed analysis and confusion matrices

Usage:
    # Install BirdNET first:
    pip install birdnetlib

    # Create datasets with --with-audio flag:
    python data_loader.py avianz /path/to/raw /path/to/joe_mo \
        --species nezfan1,silver3,comcha,nezbel1,eurbla,morepo2 --with-audio
    
    # Split datasets (audio files are automatically split too):
    python split_dataset.py /path/to/joe_mo /path/to/joe_mo_split

    # Then run BirdNET evaluation on test sets:
    python evaluate_birdnet.py \
        /path/to/joe_mo_split/test /path/to/doc_split/test \
        --output results/birdnet_eval
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import subprocess

try:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
except ImportError:
    print("ERROR: birdnetlib not installed!")
    print("Install with: pip install birdnetlib")
    sys.exit(1)


SPECIES_MAPPING = {
    'nezfan1': 'Fantail',
    'silver3': 'Silvereye',
    'comcha': 'Chaffinch',
    'nezbel1': 'Bellbird',
    'eurbla': 'Blackbird',
    'morepo2': 'Morepork'
}

SPECIES_SCIENTIFIC = {
    'nezfan1': 'Rhipidura fuliginosa',
    'silver3': 'Zosterops lateralis',
    'comcha': 'Fringilla coelebs',
    'nezbel1': 'Anthornis melanura',
    'eurbla': 'Turdus merula',
    'morepo2': 'Ninox novaeseelandiae'
}


class BirdNETEvaluator:
    def __init__(self, output_folder, min_confidence=0.1, latitude=-41.2865, longitude=174.7762):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self.latitude = latitude
        self.longitude = longitude
        
        print(f"\nInitializing BirdNET Analyzer...")
        print(f"Location: {latitude}, {longitude} (New Zealand)")
        print(f"Min confidence: {min_confidence}")
        
        self.analyzer = Analyzer()
        
        self.results = []
        self.species_codes = list(SPECIES_MAPPING.keys())
        self.species_names = [SPECIES_MAPPING[code] for code in self.species_codes]
        
        print(f"✓ BirdNET Analyzer ready")
        print(f"\nTarget species ({len(self.species_codes)}):")
        for code in self.species_codes:
            name = SPECIES_MAPPING[code]
            sci = SPECIES_SCIENTIFIC[code]
            print(f"  {code:12s} → {name:12s} ({sci})")
    

    
    def predict_file(self, wav_path):
        recording = Recording(
            self.analyzer,
            str(wav_path),
            lat=self.latitude,
            lon=self.longitude,
            min_conf=self.min_confidence
        )
        
        recording.analyze()
        
        detections = []
        for detection in recording.detections:
            detections.append({
                'common_name': detection['common_name'],
                'scientific_name': detection['scientific_name'],
                'confidence': detection['confidence'],
                'start_time': detection['start_time'],
                'end_time': detection['end_time']
            })
        
        return detections
    
    def get_top_prediction(self, detections):
        if not detections:
            return None, 0.0
        
        best = max(detections, key=lambda x: x['confidence'])
        return best['scientific_name'], best['confidence']
    
    def find_matching_species(self, scientific_name):
        for code, sci_name in SPECIES_SCIENTIFIC.items():
            if sci_name.lower() == scientific_name.lower():
                return code
        
        return None
    
    def evaluate_folder(self, test_folder, dataset_name):
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"Folder: {test_folder}")
        print(f"{'='*60}")
        
        test_path = Path(test_folder)
        
        if not test_path.exists():
            print(f"ERROR: Folder not found: {test_folder}")
            return None
        
        labels_path = test_path / 'labels.json'
        if not labels_path.exists():
            print(f"ERROR: No labels.json found")
            return None
        
        audio_path = test_path / 'audio'
        if not audio_path.exists() or not audio_path.is_dir():
            print(f"ERROR: No audio/ subfolder found")
            print(f"Make sure you used --with-audio when creating the dataset")
            return None
        
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        files = labels_data.get('files', [])
        if not files:
            print(f"ERROR: No files found in labels.json")
            return None
        
        print(f"Found {len(files)} files in dataset")
        print(f"Audio files location: {audio_path}")
        
        predictions = []
        ground_truth = []
        file_results = []
        
        birdnet_species_seen = set()
        
        print(f"\nRunning BirdNET predictions...")
        for i, file_info in enumerate(files, 1):
            npy_filename = file_info['filename']
            
            # Handle both 'label' (single-label) and 'class_names' (multi-label) formats
            if 'label' in file_info:
                label = file_info['label']
                gt_labels = [label]  # Single label as list for consistency
            elif 'class_names' in file_info:
                gt_labels = file_info['class_names']
                # For single-label evaluation, use first label (or handle multi-label appropriately)
                label = gt_labels[0] if gt_labels else None
            else:
                print(f"  [{i}/{len(files)}] SKIP: {npy_filename} (no label or class_names field)")
                continue
            
            if label is None:
                print(f"  [{i}/{len(files)}] SKIP: {npy_filename} (empty label)")
                continue
            
            # Convert .npy filename to .wav filename
            wav_filename = npy_filename.replace('.npy', '.wav')
            wav_file = audio_path / wav_filename
            
            if not wav_file.exists():
                print(f"  [{i}/{len(files)}] SKIP: {wav_filename} (audio file not found)")
                continue
            
            detections = self.predict_file(wav_file)
            
            pred_species, confidence = self.get_top_prediction(detections)
            pred_code = self.find_matching_species(pred_species) if pred_species else None
            
            if detections:
                for det in detections:
                    birdnet_species_seen.add(det['scientific_name'])
            
            predictions.append(pred_code)
            ground_truth.append(label)
            
            # For multi-label ground truth, prediction is correct if it matches ANY ground truth label
            is_correct = pred_code in gt_labels if pred_code else False
            
            # For display, show all ground truth labels if multiple exist
            gt_display = label if len(gt_labels) == 1 else f"{label}+{len(gt_labels)-1}"
            
            file_results.append({
                'filename': wav_filename,
                'ground_truth': label,
                'gt_labels': gt_labels,  # Store all ground truth labels
                'gt_name': SPECIES_MAPPING.get(label, label),
                'predicted_species': pred_species,
                'predicted_code': pred_code,
                'pred_name': SPECIES_MAPPING.get(pred_code, pred_code) if pred_code else None,
                'confidence': confidence,
                'num_detections': len(detections),
                'correct': is_correct
            })
            
            status = '✓' if is_correct else '✗'
            if i % 10 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {status} {wav_filename[:40]:40s} GT:{gt_display:10s} → Pred:{pred_code or 'None':10s} ({confidence:.2f})")
        
        n_correct = sum(1 for r in file_results if r['correct'])
        accuracy = 100.0 * n_correct / len(file_results) if file_results else 0.0
        
        print(f"\n{'='*60}")
        print(f"Results for {dataset_name}:")
        print(f"  Total files: {len(file_results)}")
        print(f"  Correct: {n_correct}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  BirdNET detected {len(birdnet_species_seen)} unique species")
        print(f"{'='*60}")
        
        result = {
            'dataset_name': dataset_name,
            'test_folder': str(test_folder),
            'audio_folder': str(audio_path),
            'num_files': len(file_results),
            'num_correct': n_correct,
            'accuracy': accuracy,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'file_results': file_results,
            'birdnet_species': sorted(birdnet_species_seen)
        }
        
        self.results.append(result)
        return result
    
    def generate_confusion_matrix(self, result):
        print(f"\nGenerating confusion matrix for {result['dataset_name']}...")
        
        gt = result['ground_truth']
        pred = result['predictions']
        
        species_with_none = self.species_codes + ['None']
        species_labels = self.species_names + ['None']
        n_species = len(species_with_none)
        
        cm = np.zeros((n_species, n_species), dtype=int)
        
        for g, p in zip(gt, pred):
            g_idx = species_with_none.index(g) if g in species_with_none else -1
            p_idx = species_with_none.index(p) if p in species_with_none else species_with_none.index('None')
            
            if g_idx >= 0:
                cm[g_idx, p_idx] += 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=species_labels, yticklabels=species_labels,
                    ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {result["dataset_name"]}\nBirdNET Evaluation', 
                     fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        plot_path = self.output_folder / f'confusion_matrix_{result["dataset_name"].lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def generate_per_species_accuracy(self):
        print(f"\nGenerating per-species accuracy plot...")
        
        species_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for result in self.results:
            for gt, pred in zip(result['ground_truth'], result['predictions']):
                if gt in self.species_codes:
                    species_stats[gt]['total'] += 1
                    if gt == pred:
                        species_stats[gt]['correct'] += 1
        
        species = []
        accuracies = []
        counts = []
        
        for code in self.species_codes:
            if species_stats[code]['total'] > 0:
                species.append(SPECIES_MAPPING[code])
                acc = 100.0 * species_stats[code]['correct'] / species_stats[code]['total']
                accuracies.append(acc)
                counts.append(species_stats[code]['total'])
            else:
                species.append(SPECIES_MAPPING[code])
                accuracies.append(0.0)
                counts.append(0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(species)), accuracies, color='steelblue', alpha=0.8)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%\n(n={count})',
                    ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Species', fontsize=12, fontweight='bold')
        ax.set_title('BirdNET Per-Species Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(len(species)))
        ax.set_xticklabels(species, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'per_species_accuracy.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def generate_dataset_comparison(self):
        print(f"\nGenerating dataset comparison plot...")
        
        dataset_names = [r['dataset_name'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        counts = [r['num_files'] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(dataset_names)), accuracies, color='lightcoral', alpha=0.8)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%\n(n={count})',
                    ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_title('BirdNET Accuracy by Dataset', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(len(dataset_names)))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_folder / 'dataset_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {plot_path}")
        plt.close()
    
    def generate_birdnet_species_report(self):
        print(f"\nGenerating BirdNET species report...")
        
        all_species = set()
        for result in self.results:
            all_species.update(result.get('birdnet_species', []))
        
        report_path = self.output_folder / 'birdnet_species_detected.txt'
        
        with open(report_path, 'w') as f:
            f.write("BirdNET Species Detection Report\n")
            f.write("="*60 + "\n\n")
            
            f.write("Target Species (Ground Truth):\n")
            f.write("-"*60 + "\n")
            for code in self.species_codes:
                name = SPECIES_MAPPING[code]
                sci = SPECIES_SCIENTIFIC[code]
                f.write(f"  {code:12s} → {name:12s} ({sci})\n")
            f.write("\n")
            
            f.write(f"BirdNET Detected {len(all_species)} Unique Species:\n")
            f.write("-"*60 + "\n")
            for i, species in enumerate(sorted(all_species), 1):
                matched = self.find_matching_species(species)
                if matched:
                    f.write(f"  {i:3d}. {species:40s} ✓ MATCH → {matched}\n")
                else:
                    f.write(f"  {i:3d}. {species:40s}\n")
            f.write("\n")
            
            f.write("Target Species Coverage:\n")
            f.write("-"*60 + "\n")
            for code in self.species_codes:
                sci = SPECIES_SCIENTIFIC[code]
                if sci in all_species:
                    f.write(f"  ✓ {code:12s} ({sci:40s}) - DETECTED\n")
                else:
                    f.write(f"  ✗ {code:12s} ({sci:40s}) - NOT DETECTED\n")
            f.write("\n")
        
        print(f"  Saved to: {report_path}")
        
        with open(report_path, 'r') as f:
            print(f"\n{f.read()}")
    
    def save_results(self):
        print(f"\nSaving detailed results...")
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'min_confidence': self.min_confidence,
            'location': {
                'latitude': self.latitude,
                'longitude': self.longitude
            },
            'target_species': {
                code: {
                    'common_name': SPECIES_MAPPING[code],
                    'scientific_name': SPECIES_SCIENTIFIC[code]
                }
                for code in self.species_codes
            },
            'results': self.results
        }
        
        json_path = self.output_folder / 'birdnet_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"  Saved to: {json_path}")
    
    def generate_summary_report(self):
        print(f"\nGenerating summary report...")
        
        report_path = self.output_folder / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("BirdNET Evaluation Summary\n")
            f.write("="*60 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Min Confidence: {self.min_confidence}\n")
            f.write(f"Location: {self.latitude}, {self.longitude}\n")
            f.write("\n")
            
            f.write("Target Species:\n")
            f.write("-"*60 + "\n")
            for code in self.species_codes:
                name = SPECIES_MAPPING[code]
                sci = SPECIES_SCIENTIFIC[code]
                f.write(f"  {code:12s} → {name:12s} ({sci})\n")
            f.write("\n")
            
            f.write("Overall Results:\n")
            f.write("-"*60 + "\n")
            total_files = sum(r['num_files'] for r in self.results)
            total_correct = sum(r['num_correct'] for r in self.results)
            overall_acc = 100.0 * total_correct / total_files if total_files > 0 else 0.0
            
            f.write(f"  Total files: {total_files}\n")
            f.write(f"  Total correct: {total_correct}\n")
            f.write(f"  Overall accuracy: {overall_acc:.2f}%\n")
            f.write("\n")
            
            f.write("Per-Dataset Results:\n")
            f.write("-"*60 + "\n")
            for result in self.results:
                f.write(f"  {result['dataset_name']}:\n")
                f.write(f"    Files: {result['num_files']}\n")
                f.write(f"    Correct: {result['num_correct']}\n")
                f.write(f"    Accuracy: {result['accuracy']:.2f}%\n")
                f.write("\n")
            
            f.write("Per-Species Accuracy:\n")
            f.write("-"*60 + "\n")
            species_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            for result in self.results:
                for gt, pred in zip(result['ground_truth'], result['predictions']):
                    if gt in self.species_codes:
                        species_stats[gt]['total'] += 1
                        if gt == pred:
                            species_stats[gt]['correct'] += 1
            
            for code in self.species_codes:
                name = SPECIES_MAPPING[code]
                if species_stats[code]['total'] > 0:
                    acc = 100.0 * species_stats[code]['correct'] / species_stats[code]['total']
                    f.write(f"  {name:12s}: {acc:6.2f}% ({species_stats[code]['correct']}/{species_stats[code]['total']})\n")
                else:
                    f.write(f"  {name:12s}: No samples\n")
        
        print(f"  Saved to: {report_path}")
        
        with open(report_path, 'r') as f:
            print(f"\n{f.read()}")
    
    def run(self, test_folders):
        for i, test_folder in enumerate(test_folders, 1):
            dataset_name = Path(test_folder).parent.name + "/" + Path(test_folder).name
            self.evaluate_folder(test_folder, dataset_name)
        
        if len(self.results) > 0:
            self.generate_birdnet_species_report()
            
            for result in self.results:
                self.generate_confusion_matrix(result)
            
            if len(self.results) > 1:
                self.generate_dataset_comparison()
            
            self.generate_per_species_accuracy()
            self.save_results()
            self.generate_summary_report()
            
            print(f"\n{'='*60}")
            print(f"✓ BirdNET evaluation complete!")
            print(f"  Results saved to: {self.output_folder}")
            print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BirdNET on test datasets with audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create datasets with audio:
  python data_loader.py avianz /raw/joe_mo /data/joe_mo --with-audio
  python split_dataset.py /data/joe_mo /data/joe_mo_split
  
  # Evaluate BirdNET on test sets:
  python evaluate_birdnet.py \
      /data/joe_mo_split/test /data/doc_split/test \
      --output results/birdnet_eval
        """
    )
    
    parser.add_argument('test_folders', nargs='+',
                       help='Test folders with audio/ subfolders (created with --with-audio)')
    parser.add_argument('--output', default='results/birdnet_evaluation',
                       help='Output folder (default: results/birdnet_evaluation)')
    parser.add_argument('--min-confidence', type=float, default=0.1,
                       help='Minimum BirdNET confidence threshold (default: 0.1)')
    parser.add_argument('--latitude', type=float, default=-41.2865,
                       help='Latitude for BirdNET (default: -41.2865, Wellington NZ)')
    parser.add_argument('--longitude', type=float, default=174.7762,
                       help='Longitude for BirdNET (default: 174.7762, Wellington NZ)')
    
    args = parser.parse_args()
    
    evaluator = BirdNETEvaluator(
        output_folder=args.output,
        min_confidence=args.min_confidence,
        latitude=args.latitude,
        longitude=args.longitude
    )
    
    evaluator.run(test_folders=args.test_folders)


if __name__ == '__main__':
    main()
