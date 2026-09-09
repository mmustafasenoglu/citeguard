#!/usr/bin/env python3
"""Validate a locally supplied Webis-CPC-11 directory without redistribution."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.dataset_dir.exists():
        raise SystemExit("Webis-CPC-11 directory does not exist.")
    print("Webis-CPC-11 adapter is dataset-layout dependent; no plagiarism metric is emitted.")


if __name__ == "__main__":
    main()
