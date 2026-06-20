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


def main():
    parser = argparse.ArgumentParser(description="Build domain classification datasets")
    parser.add_argument("matched_base", help="Path to matched base folder (containing avianz_split/, doc_split/)")
    parser.add_argument("output_base", help="Path to output folder for domain datasets")
    parser.add_argument("--overwrite", action="store_true", help="Re-build even if output already exists")
    args = parser.parse_args()

    matched = Path(args.matched_base)
    output = Path(args.output_base)

    for split in ["train", "test"]:
        out_dir = output / f"domain_{split}"
        if not args.overwrite and (out_dir / "labels.json").exists():
            print(f"  [skip] domain_{split} already exists (use --overwrite to force)")
            continue
        print(f"\n=== Building domain_{split} ===")
        build_domain_split(
            matched / "avianz_split" / split,
            matched / "doc_split" / split,
            out_dir,
        )

    print(f"\nDone.")
    print(f"  Train : {output / 'domain_train'}")
    print(f"  Test  : {output / 'domain_test'}")


if __name__ == "__main__":
    main()
