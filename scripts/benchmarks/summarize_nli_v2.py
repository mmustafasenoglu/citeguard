#!/usr/bin/env python3
"""Create transparent NLI-v2 result artifacts from completed manual runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CURRENT = "cross-encoder/nli-deberta-v3-base"
PRIMARY = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
THRESHOLD = 0.50


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in rows if row["threshold"] == THRESHOLD)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    development, external = _load(args.development), _load(args.external)
    external_en = external["external_en"]
    external_tr = external["external_tr"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = development["models"]
    model_meaning = development["model_meaning"]
    model_pareto = []
    for model_id, metadata in models.items():
        en_meaning = _threshold(model_meaning["en"][model_id])
        tr_meaning = _threshold(model_meaning["tr"][model_id])
        en_external = external_en["models"][model_id]
        tr_external = external_tr["models"][model_id]
        model_pareto.append(
            {
                "model_id": model_id,
                "english_macro_f1": en_external["macro_f1"],
                "english_contradiction_recall": en_external["per_class"]["contradiction"]["recall"],
                "turkish_macro_f1": tr_external["macro_f1"],
                "turkish_contradiction_recall": tr_external["per_class"]["contradiction"]["recall"],
                "english_safe_preserved_recall": en_meaning["safe_preserved_recall"],
                "english_unsafe_preserved_rate": en_meaning["unsafe_preserved_rate"],
                "turkish_safe_preserved_recall": tr_meaning["safe_preserved_recall"],
                "turkish_unsafe_preserved_rate": tr_meaning["unsafe_preserved_rate"],
                "single_pair_latency_ms_estimate": metadata["single_pair_latency_ms_estimate"],
                "p95_pair_latency_ms_estimate": metadata["p95_pair_latency_ms_estimate"],
                "peak_rss_bytes": metadata["peak_rss_bytes"],
                "cached_disk_bytes": metadata["cached_disk_bytes"],
            }
        )
    current = external_en["models"][CURRENT]
    candidate = external_en["models"][PRIMARY]
    candidate_tr = external_tr["models"][PRIMARY]
    current_tr = external_tr["models"][CURRENT]
    english_gate = (
        candidate["macro_f1"] >= current["macro_f1"] - 0.02
        and candidate["per_class"]["contradiction"]["recall"]
        >= current["per_class"]["contradiction"]["recall"] - 0.03
    )
    turkish_gate = candidate_tr["macro_f1"] > current_tr["macro_f1"]
    selection = {
        "decision": "RETAIN_CURRENT_ARCHITECTURE",
        "selected_architecture": "A_current_global",
        "selected_models": [CURRENT],
        "selected_thresholds": {},
        "development_only_selection": {"candidate": PRIMARY, "threshold": THRESHOLD},
        "english_external_gate": english_gate,
        "turkish_external_improvement_gate": turkish_gate,
        "holdout_touched_before_selection": False,
        "unmet_promotion_requirements": [
            "Validation and a single locked-holdout pass were not run after an external "
            "gate failure.",
            "The expanded English and Turkish rewrite-safety suites were not completed.",
            "Synthetic development data alone cannot establish a production safety rate.",
        ],
        "safety_claim": (
            "INSUFFICIENT_SAMPLE_FOR_1_PERCENT_CLAIM: synthetic development data alone does "
            "not establish production safety."
        ),
        "reason": (
            "The full official external gate is decisive before any production routing or model "
            "replacement."
        ),
    }
    architecture = {
        "phase": development["phase"],
        "threshold": THRESHOLD,
        "english": [
            row for row in development["architectures"]["en"] if row["threshold"] == THRESHOLD
        ],
        "turkish": [
            row for row in development["architectures"]["tr"] if row["threshold"] == THRESHOLD
        ],
        "routing_note": "C uses benchmark metadata only and was never production code.",
    }
    _write(
        args.output_dir / "models.json",
        {
            "models": models,
            "thresholds_tested": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        },
    )
    _write(args.output_dir / "model_pareto.json", {"threshold": THRESHOLD, "models": model_pareto})
    _write(args.output_dir / "external_en.json", external_en)
    _write(args.output_dir / "external_tr.json", external_tr)
    _write(
        args.output_dir / "meaning_en.json",
        {"dataset": development["dataset"]["en"], "models": model_meaning["en"]},
    )
    _write(
        args.output_dir / "meaning_tr.json",
        {"dataset": development["dataset"]["tr"], "models": model_meaning["tr"]},
    )
    for language in ("en", "tr"):
        rows = model_meaning[language]
        _write(
            args.output_dir / f"causality_{language}.json",
            {
                model: _threshold(values)["category"]["association_to_causation"]
                for model, values in rows.items()
            },
        )
        _write(
            args.output_dir / f"evidence_strength_{language}.json",
            {
                model: _threshold(values)["category"]["evidence_strengthening"]
                for model, values in rows.items()
            },
        )
    _write(args.output_dir / "architecture_comparison.json", architecture)
    _write(
        args.output_dir / "performance.json",
        {"models": models, "measurement": "Apple MPS; batch 32"},
    )
    _write(
        args.output_dir / "error_analysis.json",
        {
            "status": "NOT_EXPANDED_AFTER_EXTERNAL_GATE_FAILURE",
            "reason": (
                "No candidate passed the prerequisite external English gate; no production "
                "shortlist existed."
            ),
            "development_gate_attribution": {
                language: _threshold(model_meaning[language][PRIMARY])["rejection_gate_counts"]
                for language in ("en", "tr")
            },
        },
    )
    _write(args.output_dir / "selection.json", selection)
    claims = """# NLI v2 claims

- Four local NLI models were evaluated on the same official English SNLI and Turkish SNLI-TR
  development splits.
- The synthetic grouped rewrite suite is an engineering safety signal, not evidence that a model is
  safe for production.
- No architecture is promoted unless the official English budget and every safety gate pass.
- This run does not make a 1% production-safety claim; validation and holdout remain untouched after
  the external gate decision.
"""
    (args.output_dir / "CLAIMS.md").write_text(claims, encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
