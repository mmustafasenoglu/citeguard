"""Smoke coverage for the deterministic plagiarism benchmark."""

import json
import subprocess
import sys
from pathlib import Path


def test_plagiarism_engineering_benchmark_is_reproducible() -> None:
    command = [sys.executable, "scripts/benchmarks/run_plagiarism_engineering.py"]
    first_run = subprocess.run(command, check=True, capture_output=True, text=True)
    second_run = subprocess.run(command, check=True, capture_output=True, text=True)
    first = json.loads(first_run.stdout)
    second = json.loads(second_run.stdout)
    assert first == second
    assert first["exact"]["f1"] >= 0.80
    assert first["lexical"]["f1"] >= 0.50
    assert first["attribution"]["f1"] >= 0.70
    assert first["passage_localization_recall"] >= 0.80
    assert Path(first["dataset"]).name == "engineering.json"
