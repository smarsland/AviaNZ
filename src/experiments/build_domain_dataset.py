"""
Build domain classification datasets from matched train/test splits.

Creates datasets where the label is the data source ("avianz" or "doc")
rather than the bird species. Used to train a domain classifier to understand
what visual differences exist between AviaNZ and DOC spectrograms.

Usage:
    PYTHONPATH="$PWD" python3 src/experiments/build_domain_dataset.py \\
        /path/to/matched \\
        /path/to/domain_output
"""

import argparse
import json
import os
from pathlib import Path


AVIANZ_LABEL = "avianz"
DOC_LABEL = "doc"


def build_domain_split(avianz_folder, doc_folder, output_folder):
    avianz_folder = Path(avianz_folder)
    doc_folder = Path(doc_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    data_dir = output_folder / "data"
    data_dir.mkdir(exist_ok=True)

    with open(avianz_folder / "labels.json") as f:
        avianz_labels = json.load(f)
    with open(doc_folder / "labels.json") as f:
        doc_labels = json.load(f)

    output_files = []

    for source_folder, source_data, prefix, domain_label in [
        (avianz_folder, avianz_labels, "avianz_", AVIANZ_LABEL),
        (doc_folder, doc_labels, "doc_", DOC_LABEL),
    ]:
        source_data_dir = source_folder / "data"
        for file_info in source_data["files"]:
            orig_filename = file_info["filename"]
            new_filename = prefix + orig_filename

            src = (source_data_dir / orig_filename).resolve()
            dst = data_dir / new_filename
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)

            output_files.append({
                "filename": new_filename,
                "class_names": [domain_label],
                "source_file": file_info.get("source_file"),
                "original_filename": orig_filename,
            })

    labels_out = {
        "files": output_files,
        "categories": [AVIANZ_LABEL, DOC_LABEL],
        "num_classes": 2,
    }
    with open(output_folder / "labels.json", "w") as f:
        json.dump(labels_out, f, indent=2)

    n_avianz = sum(1 for e in output_files if e["class_names"] == [AVIANZ_LABEL])
    n_doc = sum(1 for e in output_files if e["class_names"] == [DOC_LABEL])
    print(f"  {output_folder.name}: {n_avianz} avianz + {n_doc} doc = {len(output_files)} total")


def build_single_domain(source_folder, domain_label, output_folder):
    """Build a dataset containing only one domain's spectrograms."""
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    data_dir = output_folder / "data"
    data_dir.mkdir(exist_ok=True)

    with open(source_folder / "labels.json") as f:
        source_labels = json.load(f)

    prefix = domain_label + "_"
    source_data_dir = source_folder / "data"
    output_files = []

    for file_info in source_labels["files"]:
        orig_filename = file_info["filename"]
        new_filename = prefix + orig_filename

        src = (source_data_dir / orig_filename).resolve()
        dst = data_dir / new_filename
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

        output_files.append({
            "filename": new_filename,
            "class_names": [domain_label],
            "source_file": file_info.get("source_file"),
            "original_filename": orig_filename,
        })

    labels_out = {
        "files": output_files,
        "categories": [AVIANZ_LABEL, DOC_LABEL],
        "num_classes": 2,
    }
    with open(output_folder / "labels.json", "w") as f:
        json.dump(labels_out, f, indent=2)

    print(f"  {output_folder.name}: {len(output_files)} {domain_label} samples")


def main():
    parser = argparse.ArgumentParser(description="Build domain classification datasets")
    parser.add_argument("matched_base", help="Path to matched base folder (containing avianz_split/, doc_split/)")
    parser.add_argument("output_base", help="Path to output folder for domain datasets")
    parser.add_argument("--overwrite", action="store_true", help="Re-build even if output already exists")
    args = parser.parse_args()

    matched = Path(args.matched_base)
    output = Path(args.output_base)

    # Combined train (mixed AviaNZ + DOC, for training the classifier)
    train_dir = output / "domain_train"
    if not args.overwrite and (train_dir / "labels.json").exists():
        print("  [skip] domain_train already exists (use --overwrite to force)")
    else:
        print("\n=== Building domain_train ===")
        build_domain_split(
            matched / "avianz_split" / "train",
            matched / "doc_split" / "train",
            train_dir,
        )

    # Per-domain test sets placed at avianz_split/test and doc_split/test so that
    # train.py's Path(test_folder).parent.name gives 'avianz_split' and 'doc_split',
    # producing distinct attention_avianz_split/ and attention_doc_split/ output dirs.
    for domain_label, split_name in [(AVIANZ_LABEL, "avianz_split"), (DOC_LABEL, "doc_split")]:
        out_dir = output / split_name / "test"
        if not args.overwrite and (out_dir / "labels.json").exists():
            print(f"  [skip] {split_name}/test already exists (use --overwrite to force)")
            continue
        print(f"\n=== Building {split_name}/test ===")
        build_single_domain(
            matched / split_name / "test",
            domain_label,
            out_dir,
        )

    print(f"\nDone.")
    print(f"  Train        : {output / 'domain_train'}")
    print(f"  Test AviaNZ  : {output / 'avianz_split' / 'test'}")
    print(f"  Test DOC     : {output / 'doc_split' / 'test'}")


if __name__ == "__main__":
    main()
