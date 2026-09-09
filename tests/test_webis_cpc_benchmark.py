import json
import subprocess
import sys
from pathlib import Path


def test_webis_adapter_scores_labeled_triplets(tmp_path: Path) -> None:
    dataset = tmp_path / "Webis-CPC-11"
    dataset.mkdir()
    (dataset / "1-original.txt").write_text("alpha beta gamma", encoding="utf-8")
    (dataset / "1-paraphrase.txt").write_text("alpha beta gamma", encoding="utf-8")
    (dataset / "1-metadata.txt").write_text("Paraphrase: Yes\n", encoding="utf-8")
    (dataset / "2-original.txt").write_text("one two three", encoding="utf-8")
    (dataset / "2-paraphrase.txt").write_text("red blue green", encoding="utf-8")
    (dataset / "2-metadata.txt").write_text("Paraphrase: No\n", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts/benchmarks/run_webis_cpc11.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dataset-dir", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert report["kind"] == "paraphrase_not_plagiarism"
    assert report["cases"] == 2
    assert report["precision"] == report["recall"] == 1.0
