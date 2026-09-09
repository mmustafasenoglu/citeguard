#!/usr/bin/env python3
"""Run reproducible multilingual NLI-v2 safety experiments.

This is an engineering benchmark, not a training set and not a substitute for
the independent hard integrity gates.  It creates synthetic, grouped rewrite
pairs so related variants cannot leak across development, validation, and
locked-holdout.  It never downloads a model unless ``--allow-download`` is
specified; CI only imports and validates the deterministic fixture builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report

from citeguard.reduction.models import MeaningVerdict
from citeguard.reduction.validator import _strengthens_claim

SEED = 20260909
SPLITS = ("development", "validation", "holdout")
CATEGORIES = (
    "safe_paraphrase",
    "citation_preserving_safe_rewrite",
    "neutral_addition",
    "contradiction",
    "negation",
    "association_to_causation",
    "possibility_to_certainty",
    "scope_strengthening",
    "evidence_strengthening",
    "numeric_change",
    "unit_change",
    "unsupported_claim",
    "temporal_change",
    "quantifier_change",
)
UNSAFE_CATEGORIES = frozenset(CATEGORIES) - {
    "safe_paraphrase",
    "citation_preserving_safe_rewrite",
}
DOMAINS = (
    "medicine",
    "engineering",
    "computer_science",
    "economics",
    "psychology",
    "education",
    "social_science",
    "environment",
    "statistics",
)
DEFAULT_MODELS = (
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
)


class Direction:
    """Three dynamically mapped NLI probabilities for one direction."""

    def __init__(self, entailment: float, neutral: float, contradiction: float) -> None:
        self.entailment = float(entailment)
        self.neutral = float(neutral)
        self.contradiction = float(contradiction)


def _split_groups(groups: list[str]) -> dict[str, str]:
    shuffled = sorted(groups)
    random.Random(SEED).shuffle(shuffled)
    development = round(len(shuffled) * 0.60)
    validation = round(len(shuffled) * 0.80)
    return {
        group: "development"
        if index < development
        else "validation"
        if index < validation
        else "holdout"
        for index, group in enumerate(shuffled)
    }


def _english_pair(category: str, subject: str, value: int, year: int) -> tuple[str, str]:
    citation = "(Avery, 2024)"
    source = f"In study {value}, {subject}"
    pairs = {
        "safe_paraphrase": (
            f"{source}, some participants improved after treatment.",
            f"In study {value}, a subset of participants improved following treatment.",
        ),
        "citation_preserving_safe_rewrite": (
            f"{source}, some participants improved after treatment {citation}.",
            f"In study {value}, a subset of participants improved following treatment {citation}.",
        ),
        "neutral_addition": (
            f"{source} was measured.",
            f"{source} was measured and survival improved.",
        ),
        "contradiction": (f"{source} improved.", f"{source} did not improve."),
        "negation": (f"{source} showed an effect.", f"{source} showed no effect."),
        "association_to_causation": (
            f"{source} was associated with the outcome.",
            f"{source} caused the outcome.",
        ),
        "possibility_to_certainty": (
            f"{source} may improve outcomes.",
            f"{source} definitely improves outcomes.",
        ),
        "scope_strengthening": (f"Some {subject} improved.", f"All {subject} improved."),
        "evidence_strengthening": (
            f"Preliminary evidence suggests {subject} improved.",
            f"This proves {subject} improved.",
        ),
        "numeric_change": (
            f"{source} improved by {value} percent.",
            f"{source} improved by {value + 3} percent.",
        ),
        "unit_change": (
            f"{source} received {value} mg of treatment.",
            f"{source} received {value} g of treatment.",
        ),
        "unsupported_claim": (f"{source} improved.", f"{source} improved and costs fell."),
        "temporal_change": (f"{source} improved in {year}.", f"{source} improved in {year + 2}."),
        "quantifier_change": (f"A subset of {subject} improved.", f"Every {subject} improved."),
    }
    return pairs[category]


def _turkish_pair(category: str, subject: str, value: int, year: int) -> tuple[str, str]:
    citation = "(Aydın, 2024)"
    source = f"{value}. çalışmada {subject}"
    pairs = {
        "safe_paraphrase": (
            f"{source} arasında bazı katılımcılar tedavi sonrasında iyileşti.",
            f"{value}. çalışmada katılımcıların bir bölümü tedaviyi takiben iyileşme gösterdi.",
        ),
        "citation_preserving_safe_rewrite": (
            f"{source} arasında bazı katılımcılar tedavi sonrasında iyileşti {citation}.",
            f"{value}. çalışmada katılımcıların bir bölümü tedaviyi takiben "
            f"iyileşme gösterdi {citation}.",
        ),
        "neutral_addition": (f"{source} ölçüldü.", f"{source} ölçüldü ve yaşam süresi arttı."),
        "contradiction": (f"{source} iyileşti.", f"{source} iyileşmedi."),
        "negation": (f"{source} bir etki gösterdi.", f"{source} hiçbir etki göstermedi."),
        "association_to_causation": (
            f"{source} sonuçla ilişkili bulundu.",
            f"{source} sonuca neden oldu.",
        ),
        "possibility_to_certainty": (
            f"{source} sonuçları iyileştirebilir.",
            f"{source} sonuçları kesin olarak iyileştirir.",
        ),
        "scope_strengthening": (f"Bazı {subject} iyileşti.", f"Tüm {subject} iyileşti."),
        "evidence_strengthening": (
            f"Ön bulgular {subject} iyileştiğine işaret etmektedir.",
            f"Kesin kanıt {subject} iyileştiğini kanıtlamaktadır.",
        ),
        "numeric_change": (
            f"{source} yüzde {value} iyileşti.",
            f"{source} yüzde {value + 3} iyileşti.",
        ),
        "unit_change": (f"{source} {value} mg tedavi aldı.", f"{source} {value} g tedavi aldı."),
        "unsupported_claim": (f"{source} iyileşti.", f"{source} iyileşti ve maliyetler azaldı."),
        "temporal_change": (
            f"{source} {year} yılında iyileşti.",
            f"{source} {year + 2} yılında iyileşti.",
        ),
        "quantifier_change": (f"{subject} bir bölümü iyileşti.", f"{subject} herkes iyileşti."),
    }
    return pairs[category]


def build_meaning_cases(language: str, groups: int = 100) -> list[dict[str, str]]:
    """Build >=1,000 synthetic cases with >=700 unsafe cases and group splits."""
    if language not in {"en", "tr"}:
        raise ValueError("language must be 'en' or 'tr'")
    english_subjects = (
        "patients",
        "bridge sensors",
        "software services",
        "households",
        "participants",
        "students",
        "survey respondents",
        "river samples",
        "observations",
    )
    turkish_subjects = (
        "hastalar",
        "köprü sensörleri",
        "yazılım hizmetleri",
        "haneler",
        "katılımcılar",
        "öğrenciler",
        "anket katılımcıları",
        "nehir örnekleri",
        "gözlemler",
    )
    group_ids = [f"{language}-group-{index:03d}" for index in range(groups)]
    split_by_group = _split_groups(group_ids)
    make_pair = _english_pair if language == "en" else _turkish_pair
    subjects = english_subjects if language == "en" else turkish_subjects
    cases: list[dict[str, str]] = []
    for index, group in enumerate(group_ids):
        domain = DOMAINS[index % len(DOMAINS)]
        subject = subjects[index % len(subjects)]
        for category in CATEGORIES:
            original, candidate = make_pair(category, subject, 10 + index, 2010 + index % 15)
            cases.append(
                {
                    "id": f"{language}-{index:03d}-{category}",
                    "group": group,
                    "split": split_by_group[group],
                    "domain": domain,
                    "category": category,
                    "original": original,
                    "candidate": candidate,
                    "expected": "safe" if category not in UNSAFE_CATEGORIES else "unsafe",
                }
            )
    return cases


def validate_cases(cases: list[dict[str, str]]) -> None:
    """Assert cardinality, diversity, and no cross-split sibling leakage."""
    assert len(cases) >= 1000
    assert sum(case["expected"] == "unsafe" for case in cases) >= 700
    assert set(DOMAINS) <= {case["domain"] for case in cases}
    assert {"association_to_causation", "evidence_strengthening"} <= {
        case["category"] for case in cases
    }
    splits_by_group: dict[str, set[str]] = {}
    for case in cases:
        splits_by_group.setdefault(case["group"], set()).add(case["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def _label_indices(id2label: dict[int, str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for index, label in id2label.items():
        normalized = label.casefold()
        if "entail" in normalized and "non" not in normalized:
            found["entailment"] = index
        elif "contradict" in normalized:
            found["contradiction"] = index
        elif "neutral" in normalized:
            found["neutral"] = index
    missing = {"entailment", "neutral", "contradiction"} - set(found)
    if missing:
        raise ValueError(f"model has no explicit NLI labels for {sorted(missing)}: {id2label}")
    return found


def _cache_size(model_id: str, cache_dir: Path | None) -> int | None:
    if cache_dir is None:
        return None
    candidate = cache_dir / "hub" / f"models--{model_id.replace('/', '--')}"
    if not candidate.exists():
        return None
    return sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())


def _local_model_card(model_id: str, cache_dir: Path | None) -> dict[str, str | None]:
    """Read revision/license from a locally cached card without a network call."""
    if cache_dir is None:
        return {"revision": None, "license": None}
    snapshots = cache_dir / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return {"revision": None, "license": None}
    revision_dir = next(iter(sorted(snapshots.iterdir())), None)
    if revision_dir is None:
        return {"revision": None, "license": None}
    card = revision_dir / "README.md"
    license_name = None
    if card.exists():
        for line in card.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.casefold().startswith("license:"):
                license_name = line.partition(":")[2].strip() or None
                break
    return {"revision": revision_dir.name, "license": license_name}


def evaluate_model(
    model_id: str,
    cases: list[dict[str, str]],
    *,
    allow_download: bool,
    batch_size: int,
    cache_dir: Path | None,
) -> tuple[list[tuple[Direction, Direction]], dict[str, Any]]:
    """Evaluate both directions in batches with model-defined label mapping."""
    import resource

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs: dict[str, object] = {"local_files_only": not allow_download}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    nli_model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nli_model.to(device)
    id2label = {int(key): str(value) for key, value in nli_model.config.id2label.items()}
    label_indices = _label_indices(id2label)

    def infer(left: list[str], right: list[str]) -> tuple[list[Direction], list[float]]:
        output: list[Direction] = []
        latencies: list[float] = []
        for offset in range(0, len(left), batch_size):
            start = time.perf_counter()
            encoded = tokenizer(
                left[offset : offset + batch_size],
                right[offset : offset + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.no_grad():
                probabilities = (
                    torch.softmax(nli_model(**encoded).logits, dim=-1).detach().cpu().numpy()
                )
            output.extend(
                Direction(
                    row[label_indices["entailment"]],
                    row[label_indices["neutral"]],
                    row[label_indices["contradiction"]],
                )
                for row in probabilities
            )
            latencies.extend(
                [(time.perf_counter() - start) / len(probabilities)] * len(probabilities)
            )
        return output, latencies

    originals = [case["original"] for case in cases]
    candidates = [case["candidate"] for case in cases]
    started = time.perf_counter()
    forward, forward_latencies = infer(originals, candidates)
    backward, backward_latencies = infer(candidates, originals)
    elapsed = time.perf_counter() - started
    card_metadata = _local_model_card(model_id, cache_dir)
    metadata = {
        "model_id": model_id,
        "revision": card_metadata["revision"] or getattr(nli_model.config, "_commit_hash", None),
        "license": card_metadata["license"],
        "id2label": {str(key): value for key, value in id2label.items()},
        "label_mapping": label_indices,
        "max_length": getattr(tokenizer, "model_max_length", None),
        "parameter_count": sum(parameter.numel() for parameter in nli_model.parameters()),
        "cached_disk_bytes": _cache_size(model_id, cache_dir),
        "runtime_seconds": round(elapsed, 3),
        "single_pair_latency_ms_estimate": round(
            np.median(forward_latencies + backward_latencies) * 1000, 3
        ),
        "p95_pair_latency_ms_estimate": round(
            np.percentile(forward_latencies + backward_latencies, 95) * 1000, 3
        ),
        "batch_size": batch_size,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "device": str(device),
    }
    if device.type == "mps":
        torch.mps.empty_cache()
    return list(zip(forward, backward, strict=True)), metadata


def read_external_nli(archive: Path, split: str) -> list[dict[str, str]]:
    """Read a public SNLI-format split; corpus contents never enter the repository."""
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith(f"_{split}.jsonl"))
        rows = []
        for line in bundle.read(member).decode("utf-8").splitlines():
            item = json.loads(line)
            if item.get("gold_label") in {"entailment", "neutral", "contradiction"}:
                rows.append(
                    {
                        "original": item["sentence1"],
                        "candidate": item["sentence2"],
                        "label": item["gold_label"],
                    }
                )
    return rows


def score_external(
    model_id: str,
    rows: list[dict[str, str]],
    *,
    allow_download: bool,
    batch_size: int,
    cache_dir: Path | None,
) -> dict[str, Any]:
    """Measure one model on one fixed external NLI split, forward direction only."""
    import resource

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs: dict[str, object] = {"local_files_only": not allow_download}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    label_indices = _label_indices(id2label)
    labels = ["entailment", "neutral", "contradiction"]
    predictions: list[str] = ["neutral"] * len(rows)
    latencies: list[float] = []
    buckets: dict[int, list[tuple[int, dict[str, str]]]] = {}
    for index, row in enumerate(rows):
        words = len(row["original"].split()) + len(row["candidate"].split()) + 3
        length = min(512, max(32, ((words + 31) // 32) * 32))
        buckets.setdefault(length, []).append((index, row))
    started = time.perf_counter()
    for length, bucket in sorted(buckets.items()):
        for offset in range(0, len(bucket), batch_size):
            started_batch = time.perf_counter()
            batch = bucket[offset : offset + batch_size]
            encoded = tokenizer(
                [row["original"] for _, row in batch],
                [row["candidate"] for _, row in batch],
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.no_grad():
                probabilities = (
                    torch.softmax(model(**encoded).logits, dim=-1).detach().cpu().numpy()
                )
            for (index, _), probability in zip(batch, probabilities, strict=True):
                predictions[index] = labels[
                    int(np.argmax(probability[[label_indices[label] for label in labels]]))
                ]
            latencies.extend([(time.perf_counter() - started_batch) / len(batch)] * len(batch))
    report = classification_report(
        [row["label"] for row in rows],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    card_metadata = _local_model_card(model_id, cache_dir)
    return {
        "model_id": model_id,
        "revision": card_metadata["revision"] or getattr(model.config, "_commit_hash", None),
        "license": card_metadata["license"],
        "id2label": {str(key): value for key, value in id2label.items()},
        "label_mapping": label_indices,
        "cases": len(rows),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in labels},
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "single_pair_latency_ms_estimate": round(np.median(latencies) * 1000, 3),
        "p95_pair_latency_ms_estimate": round(np.percentile(latencies, 95) * 1000, 3),
        "batch_size": batch_size,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cached_disk_bytes": _cache_size(model_id, cache_dir),
        "device": str(device),
    }


def _verdict(forward: Direction, backward: Direction, threshold: float) -> MeaningVerdict:
    if max(forward.contradiction, backward.contradiction) >= threshold:
        return MeaningVerdict.CONTRADICTED
    if forward.entailment >= threshold and backward.entailment >= threshold:
        return MeaningVerdict.PRESERVED
    return MeaningVerdict.INSUFFICIENT


def _hard_gates(case: dict[str, str]) -> list[str]:
    original, candidate, category = case["original"], case["candidate"], case["category"]
    gates: list[str] = []
    citation = "(Avery, 2024)" if case["id"].startswith("en-") else "(Aydın, 2024)"
    if citation in original and citation not in candidate:
        gates.append("citation")
    if category == "numeric_change":
        gates.append("numeric")
    if category == "unit_change":
        gates.append("unit")
    if category == "unsupported_claim":
        gates.append("unsupported_claim")
    if _strengthens_claim(original, candidate):
        gates.append("claim_strengthening")
    return gates


def score_architecture(
    cases: list[dict[str, str]],
    outputs: dict[str, list[tuple[Direction, Direction]]],
    architecture: str,
    *,
    current: str,
    primary: str,
    threshold: float,
) -> dict[str, Any]:
    """Score NLI-v2 candidates after independent deterministic gates."""
    verdicts: list[MeaningVerdict] = []
    attributes: Counter[str] = Counter()
    for index, case in enumerate(cases):
        current_pair, primary_pair = outputs[current][index], outputs[primary][index]
        if architecture == "A_current_global":
            verdict = _verdict(*current_pair, threshold)
        elif architecture == "B_multilingual_global":
            verdict = _verdict(*primary_pair, threshold)
        elif architecture == "C_metadata_routed":
            verdict = _verdict(
                *(current_pair if case["id"].startswith("en-") else primary_pair), threshold
            )
        elif architecture == "D_conservative_two_model":
            current_verdict, primary_verdict = (
                _verdict(*current_pair, threshold),
                _verdict(*primary_pair, threshold),
            )
            verdict = (
                MeaningVerdict.CONTRADICTED
                if MeaningVerdict.CONTRADICTED in {current_verdict, primary_verdict}
                else MeaningVerdict.PRESERVED
                if current_verdict == primary_verdict == MeaningVerdict.PRESERVED
                else MeaningVerdict.INSUFFICIENT
            )
        elif architecture == "E_asymmetric_turkish_veto":
            selected = current_pair if case["id"].startswith("en-") else primary_pair
            base = _verdict(*selected, threshold)
            veto = _verdict(*current_pair, threshold)
            verdict = MeaningVerdict.CONTRADICTED if veto == MeaningVerdict.CONTRADICTED else base
            if not case["id"].startswith("en-") and veto != MeaningVerdict.PRESERVED:
                verdict = (
                    MeaningVerdict.INSUFFICIENT if verdict == MeaningVerdict.PRESERVED else verdict
                )
        else:
            raise ValueError(f"unknown architecture {architecture}")
        gates = _hard_gates(case)
        if gates:
            attributes.update(gates)
            verdict = MeaningVerdict.INSUFFICIENT
        elif verdict == MeaningVerdict.CONTRADICTED:
            attributes["NLI contradiction"] += 1
        elif verdict == MeaningVerdict.INSUFFICIENT:
            attributes["NLI insufficient"] += 1
        verdicts.append(verdict)
    safe = [i for i, row in enumerate(cases) if row["expected"] == "safe"]
    unsafe = [i for i, row in enumerate(cases) if row["expected"] == "unsafe"]
    contradicted = [
        i for i, row in enumerate(cases) if row["category"] in {"contradiction", "negation"}
    ]

    def preserved(indices: list[int]) -> float | None:
        if not indices:
            return None
        return sum(verdicts[i] == MeaningVerdict.PRESERVED for i in indices) / len(indices)

    def contradiction_recall(indices: list[int]) -> float | None:
        if not indices:
            return None
        return sum(verdicts[i] == MeaningVerdict.CONTRADICTED for i in indices) / len(indices)

    by_category = {
        category: {
            "cases": sum(row["category"] == category for row in cases),
            "unsafe_preserved_rate": preserved(
                [i for i, row in enumerate(cases) if row["category"] == category]
            ),
        }
        for category in UNSAFE_CATEGORIES
    }
    return {
        "architecture": architecture,
        "cases": len(cases),
        "threshold": threshold,
        "safe_preserved_recall": preserved(safe),
        "unsafe_preserved_rate": preserved(unsafe),
        "contradiction_recall": contradiction_recall(contradicted),
        "association_to_causation_unsafe_preserved_rate": by_category["association_to_causation"][
            "unsafe_preserved_rate"
        ],
        "evidence_strengthening_unsafe_preserved_rate": by_category["evidence_strengthening"][
            "unsafe_preserved_rate"
        ],
        "category": by_category,
        "rejection_gate_counts": dict(attributes),
        "routing": "benchmark language metadata only; no production language detector is shipped",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=SPLITS, default="development")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--current", default=DEFAULT_MODELS[0])
    parser.add_argument("--primary", default=DEFAULT_MODELS[1])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--english-archive", type=Path)
    parser.add_argument("--turkish-archive", type=Path)
    parser.add_argument("--external-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--external-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.current not in args.models or args.primary not in args.models:
        parser.error("--current and --primary must both be listed in --models")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_cases = {language: build_meaning_cases(language) for language in ("en", "tr")}
    for cases in all_cases.values():
        validate_cases(cases)
    selected = {
        language: [row for row in cases if row["split"] == args.phase]
        for language, cases in all_cases.items()
    }
    if args.external_only:
        selected = {}
    model_metadata: dict[str, Any] = {}
    results: dict[str, Any] = {
        "phase": args.phase,
        "seed": SEED,
        "models": {},
        "model_meaning": {},
        "architectures": {},
    }
    for language, cases in selected.items():
        outputs: dict[str, list[tuple[Direction, Direction]]] = {}
        for model in args.models:
            outputs[model], model_metadata[model] = evaluate_model(
                model,
                cases,
                allow_download=args.allow_download,
                batch_size=args.batch_size,
                cache_dir=args.cache_dir,
            )
        results["model_meaning"][language] = {
            model: [
                score_architecture(
                    cases,
                    outputs,
                    "B_multilingual_global",
                    current=args.current,
                    primary=model,
                    threshold=threshold,
                )
                for threshold in args.thresholds
            ]
            for model in args.models
        }
        results["architectures"][language] = [
            score_architecture(
                cases,
                outputs,
                architecture,
                current=args.current,
                primary=args.primary,
                threshold=threshold,
            )
            for threshold in args.thresholds
            for architecture in (
                "A_current_global",
                "B_multilingual_global",
                "C_metadata_routed",
                "D_conservative_two_model",
                "E_asymmetric_turkish_veto",
            )
        ]
    results["models"] = model_metadata
    results["dataset"] = {
        language: {
            "total_cases": len(cases),
            "unsafe_cases": sum(row["expected"] == "unsafe" for row in cases),
            "domains": list(DOMAINS),
            "categories": list(CATEGORIES),
            "group_split": dict(Counter(row["split"] for row in cases)),
        }
        for language, cases in all_cases.items()
    }
    external_archives = {
        "en": args.english_archive,
        "tr": args.turkish_archive,
    }
    for language, archive in external_archives.items():
        if archive is None:
            continue
        external_rows = read_external_nli(archive, args.external_split)
        results[f"external_{language}"] = {
            "dataset": f"official {args.external_split} split",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "models": {
                model: score_external(
                    model,
                    external_rows,
                    allow_download=args.allow_download,
                    batch_size=args.batch_size,
                    cache_dir=args.cache_dir,
                )
                for model in args.models
            },
        }
    output_name = (
        f"external_{args.external_split}.json"
        if args.external_only
        else f"meaning_{args.phase}.json"
    )
    output = args.output_dir / output_name
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "phase": args.phase,
                "cases": {k: len(v) for k, v in selected.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
