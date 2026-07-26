#!/usr/bin/env python3
"""Move interrupted/corrupt image downloads aside so curl can retry them."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from image_integrity import is_complete_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    image_dir = args.dataset_dir / "images"
    quarantine = args.dataset_dir / "incomplete_images"
    moved = 0
    valid = 0
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if is_complete_image(path):
                valid += 1
                continue
            else:
                raise ValueError("incomplete or unsupported image")
        except Exception:
            quarantine.mkdir(exist_ok=True)
            target = quarantine / path.name
            if target.exists():
                target = quarantine / f"{path.stem}.{path.stat().st_size}{path.suffix}"
            shutil.move(path, target)
            moved += 1
    print({"valid": valid, "quarantined": moved})


if __name__ == "__main__":
    main()
