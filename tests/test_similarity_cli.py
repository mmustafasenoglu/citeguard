"""CLI regression tests for similarity output and license defaults."""

from __future__ import annotations

import json

from click.testing import CliRunner

from citeguard.cli import main
from citeguard.corpus.licenses import LicenseType, get_permissions


def test_unknown_license_is_not_open_for_commercial_use() -> None:
    perms = get_permissions(LicenseType.UNKNOWN)
    assert perms["similarity_index_allowed"] is False
    assert perms["commercial_use_allowed"] is False
    assert perms["training_use_allowed"] is False


def test_open_access_is_not_treated_as_commercial_license() -> None:
    perms = get_permissions(LicenseType.OPEN_ACCESS)
    assert perms["similarity_index_allowed"] is False
    assert perms["commercial_use_allowed"] is False
    assert perms["training_use_allowed"] is False


def test_similarity_json_and_show_sentences(tmp_path) -> None:
    paper = tmp_path / "paper.txt"
    corpus = tmp_path / "corpus.txt"
    text = (
        "Transformer architectures were introduced in 2017 by Vaswani et al. "
        "The attention mechanism revolutionized natural language processing."
    )
    paper.write_text(text, encoding="utf-8")
    corpus.write_text(text, encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "similarity",
            str(paper),
            "--corpus",
            str(corpus),
            "--corpus-license",
            "CC0",
            "--format",
            "json",
            "--show-sentences",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "3"
    assert "summary" in payload
    assert "results" in payload
    assert "attribution_risk" in payload["results"][0]


def test_similarity_skips_unknown_license_by_default(tmp_path) -> None:
    paper = tmp_path / "paper.txt"
    corpus = tmp_path / "corpus.txt"
    text = "Transformer architectures were introduced in 2017 by Vaswani et al."
    paper.write_text(text, encoding="utf-8")
    corpus.write_text(text, encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["similarity", str(paper), "--corpus", str(corpus)],
    )
    assert result.exit_code == 0
    assert "0 unique segments" in result.output
