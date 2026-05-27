#!/usr/bin/env python3
"""
Collect all images from matched_tests into a single flat folder.

Each image is renamed to:
    <test_folder>__<relative_subpath>__<filename>.ext
so files from the same experiment sit together when sorted alphabetically.

Usage:
    python collect_images.py [--src matched_tests] [--dst collected_images]
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


def collect(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)

    # Each immediate subdirectory of src is a test folder
    test_dirs = sorted(p for p in src.iterdir() if p.is_dir())

    copied = 0
    for test_dir in test_dirs:
        for img_path in sorted(test_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if "attention" not in img_path.parts[-2] and "attention" not in img_path.name:
                continue

            # Build a flat name: testfolder__sub__dir__filename.ext
            rel = img_path.relative_to(test_dir)
            parts = list(rel.parts)          # e.g. ['attention_avianz_split', 'multiclass_attention_0000.png']
            flat_name = test_dir.name + "__" + "__".join(parts)

            dst_file = dst / flat_name
            # Avoid silent overwrites in the unlikely event of a name clash
            if dst_file.exists():
                stem = dst_file.stem
                suffix = dst_file.suffix
                dst_file = dst / f"{stem}_dup{suffix}"

            shutil.copy2(img_path, dst_file)
            copied += 1

    print(f"Copied {copied} images from {len(test_dirs)} test folders → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten matched_tests images into one folder.")
    parser.add_argument("--src", default="matched_tests", help="Source folder (default: matched_tests)")
    parser.add_argument("--dst", default="collected_images", help="Destination folder (default: collected_images)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    collect(src, dst)


if __name__ == "__main__":
    main()
