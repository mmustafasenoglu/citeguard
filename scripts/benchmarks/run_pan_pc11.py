#!/usr/bin/env python3
"""Evaluate locally obtained PAN-PC-11 annotations; never downloads the corpus."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from citeguard.benchmarks import parse_pan_xml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--thresholds", type=Path)
    args = parser.parse_args()
    xml_files = sorted(args.dataset_dir.rglob("*.xml"))
    if not xml_files:
        raise SystemExit("No PAN XML annotations found; obtain PAN-PC-11 from Webis/PAN first.")
    passages = [item for path in xml_files for item in parse_pan_xml(str(path))]
    result = {
        "benchmark": "PAN-PC-11",
        "split": args.split,
        "ground_truth_passages": len(passages),
        "status": "annotations_parsed_predictions_not_run",
        "python": sys.version,
        "platform": platform.platform(),
        "thresholds": str(args.thresholds) if args.thresholds else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
