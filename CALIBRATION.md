# citeguard v1.0 Calibration

> **Status:** Calibration complete. All release gate targets met.
> **Date:** 2026-09-08
> **Scope:** Offline deterministic benchmarks only. Not a scientific validation.

---

## 1. Benchmark Suite

citeguard uses a fixed JSON fixture-based benchmark suite with **6 categories** and two dataset splits:

| Category | Calibration | Holdout | Total |
|---|---|---|---|
| citation_parsing | 20 | 15 | 35 |
| bibliography_resolution | 16 | 12 | 28 |
| retrieval_ranking | 12 | 10 | 22 |
| suggestion_threshold | 15 | 10 | 25 |
| health_score | 6 | 4 | 10 |
| entailment | 10 | 8 | 18 |
| **Total** | **79** | **59** | **138** |

### Calibration vs Holdout

- **Calibration set:** Used to tune thresholds and verify implementation correctness.
- **Holdout set:** Used **only** for final evaluation, **never** for threshold tuning.

---

## 2. Chosen Thresholds

| Threshold | Value | Constant |
|---|---|---|
| Suggestion (default) | 60 | `DEFAULT_THRESHOLD` |
| Metadata verification | 70 | `VERIFIED_METADATA_THRESHOLD` |
| Strong retrieval | 80 | `STRONG_MATCH_THRESHOLD` |
| Health pass (exit code) | 80 | `HEALTH_PASS_THRESHOLD` |

---

## 3. Calibration Methodology

- Fixed JSON fixtures define inputs and expected outputs.
- Deterministic evaluation: same input always produces the same result.
- **No network calls** during calibration.
- Mocked providers return canned responses; no real API calls.
- Holdout set is held back and only evaluated after all threshold decisions are final.

---

## 4. Release Gate Targets

| Category | Gate | Target |
|---|---|---|
| citation_parsing | F1 | >= 0.95 |
| bibliography_resolution | VERIFIED precision | >= 0.95 |
| bibliography_resolution | VERIFIED recall | >= 0.85 |
| retrieval_ranking | MRR | >= 0.90 |
| retrieval_ranking | Recall@5 | >= 0.95 |
| suggestion_threshold | precision | >= 0.90 |
| suggestion_threshold | recall | >= 0.70 |
| health_score | monotonicity violations | = 0 |
| entailment | accuracy | >= 0.80 |

---

## 5. Calibration Results

All categories meet or exceed release gate targets on both calibration and holdout sets.

| Category | Metric | Calibration | Holdout | Gate Met |
|---|---|---|---|---|
| citation_parsing | F1 | 1.0 | 1.0 | Yes |
| bibliography_resolution | verified_precision | 1.0 | 1.0 | Yes |
| bibliography_resolution | verified_recall | 0.9 | 1.0 | Yes |
| retrieval_ranking | MRR | 0.917 | 1.0 | Yes |
| retrieval_ranking | recall@5 | 1.0 | 1.0 | Yes |
| suggestion_threshold | precision | 1.0 | 1.0 | Yes |
| suggestion_threshold | recall | 1.0 | 1.0 | Yes |
| health_score | monotonicity violations | 0 | 0 | Yes |
| entailment | accuracy | 1.0 | 1.0 | Yes |

---

## 6. Scoring Formulas

### Health Score Weights

| Component | Weight |
|---|---|
| citation_coverage | 0.25 |
| verification_ratio | 0.25 |
| support_ratio | 0.20 |
| evidence_coverage | 0.20 |
| bibliography_consistency | 0.10 |

### Penalties

| Condition | Penalty |
|---|---|
| uncited_high | -2 |
| contradictions | -4 |
| unresolved_high | -2 |

### Overall Confidence Formula

- **With entailment:** `metadata * 0.20 + support * 0.80`
- **Without entailment:** `metadata * 0.40 + support * 0.60`

---

## 7. Mocked-vs-Real Distinction

This calibration was performed entirely with mocked providers. The benchmark suite verifies:

- **Parser correctness** against known citation and bibliography formats.
- **Ranking logic** against predefined candidate lists.
- **Scoring monotonicity** against synthetic health score inputs.
- **Threshold behavior** against calibrated suggestion confidence values.

**Not calibrated here:**

- Real-world provider search quality (Semantic Scholar, Crossref, arXiv latency, coverage, or ranking).
- LLM output variability across providers, models, or prompt versions.
- Edge-case document structures beyond the fixture set.

Real provider behavior is evaluated separately via `citeguard llm doctor` and manual review.

---

## 8. Known Limitations

- Citation parser does not detect all edge forms (e.g., parenthetical with "see" prefix, colon-chapter format).
- Bibliography resolution benchmark tests metadata matching only, not provider search quality.
- Entailment benchmark uses deterministic scoring, not real LLM output.
- Health score benchmark tests mathematical properties, not real-world classification.
- Suggestion threshold calibration uses `overall_confidence` formula, not full pipeline.

---

## 9. No Scientific Validation

This calibration document records the results of an internal benchmark suite designed to verify implementation correctness of citeguard's deterministic components.

**This is not a scientific validation.** The benchmark fixtures are synthetic, the evaluation is deterministic, and the results do not generalize to arbitrary real-world documents. No claims of academic rigor, statistical significance, or cross-domain applicability are made or implied.

Users should perform their own evaluation on representative documents before relying on citeguard for any consequential purpose.
