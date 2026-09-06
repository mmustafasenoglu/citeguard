# Changelog

All notable changes to citeguard will be documented here.

## [0.2.0] - 2026-09-06

### Added

- `Evidence` data model and `EvidenceType` enum for evidence passage representation.
- Abstract-level evidence extraction with sentence splitting, lexical ranking, and top-N selection.
- Entailment evaluation: offline negation-signal classifier and optional LLM-backed entailment.
- Three-stage matcher pipeline: metadata match → evidence → entailment → aggregate confidence.
- Offline lexical matcher no longer returns `SUPPORTED` verdict; only produces `partially_supported`
  at most. `SUPPORTED` and `CONTRADICTED` require entailment evaluation.
- Evidence-aware scoring: `evidence_coverage` metric, updated health score weights, priority
  scoring considers evidence relevance.
- Evidence rendering in terminal, Markdown, and JSON reports.
- CLI options: `--show-evidence` (display evidence passages in terminal) and `--require-evidence`
  (filter to only claims with evidence in priority review).
- `match_claim_to_source` now returns a 5-tuple: `(metadata_score, support_score, verdict,
  reasoning, evidence_list)`.
- `overall_confidence` accepts optional `has_entailment` keyword arg for weighted aggregation.
- `compute_health_score` requires `evidence_coverage` parameter.
- `AuditMetrics` dataclass includes `evidence_coverage` field.
- `MatchResult` dataclass includes `entailment_score` and `entailment_verdict` fields.
- Multi-paragraph citation linking: `claims.py` now accepts `paragraph_index` parameter.
- Parser authoritative for citation presence: `llm.py` ignores LLM `has_existing_citation` hints.
- LLM claim extraction maps claims back to original sentences via token overlap for citation
  detection, fixing false negatives when LLM strips citation markers from claim text.
- Integration tests for evidence pipeline, `--require-evidence` filter, and sentence-mapping
  citation detection.

### Changed

- Removed legacy `_try_llm_match()` bypass from matcher; LLM is now only invoked inside
  `_evaluate_evidence()` via `evaluate_evidence_with_llm()`, ensuring the full evidence→entailment
  pipeline always runs.
- `support_score` is now a standalone signal (evidence + entailment only, no metadata).
  `overall_confidence` combines metadata and support exactly once.
- Health score formula weights: citation_coverage 25%, verification_ratio 25%, support_ratio 20%,
  evidence_coverage 20%, bibliography_consistency 10% (previously 30/30/25/0/15).
- `overall_confidence` formula with entailment: metadata 20% + support 80%; without entailment:
  metadata 40% + support 60%.  No double-counting of metadata.
- Offline contradiction detector no longer returns `CONTRADICTED` verdict.  Negation signals
  with high topical overlap produce elevated-confidence `INSUFFICIENT_INFORMATION` instead.
  Final `CONTRADICTED` verdict requires LLM-backed entailment.
- Version bumped to 0.2.0 across `pyproject.toml`, `__init__.py`, and `CHANGELOG.md`.

## [0.1.0] - 2026-09-05

### Added

- Initial project specification and open-source repository structure.
- Offline `.md`, `.txt`, and `.docx` parsing foundation.
- Author-year, numbered, and DOI citation detection.
- Basic bibliography parser and consistency checks.
- Deterministic scoring helpers.
- Versioned local JSON cache primitive.
- Development CLI `inspect` command.
- Initial test suite and CI configuration.
- Semantic Scholar, Crossref, and arXiv academic metadata providers.
- Provider HTTP retry handling for rate limits and transient server failures.
- Cached, failure-tolerant retrieval orchestration with deterministic ranking.
- DOI/arXiv normalization and canonical candidate deduplication.
- Document-order extraction of paragraphs inside DOCX tables, including nested and merged cells.
- Language-neutral bibliography appendix detection for structured application forms.
- Narrative and semicolon-separated author-date citation detection.
- Organization-author, page-locator, and Turkish/English no-date citation support.
- Turkish bibliography headings, including appendix-prefixed headings.
- Paragraph-level citation details and bibliography match status in `inspect` output.
- Deterministic author, year, title, and DOI metadata scoring.
- Sentence-level citation context linking.
- Crossref-backed `verify` command with direct DOI lookup and cached provider responses.
- Stable JSON output for `inspect` and `verify`.
- `--format both` option for `inspect`, `check`, `suggest`, and `verify` commands (produces `.citeguard.json` and `.citeguard.md` files).
- Deterministic claim-to-citation linking with sentence-final and mid-sentence confidence scores.
- Markdown inspection report for `--format both` output.
- Expanded test suite for citation linking, cache edge cases, and format-both integration.
- Reproducible example output committed under `examples/` for `inspect` and `check`.
- CI workflow: pip caching, branch-filtered push triggers, short test traceback.
- Improved provider error messages with attempt counts and underlying cause context.
- Empty document validation across all CLI commands with clear error messages.
- Optional LLM integration via Anthropic Messages API for enhanced claim extraction
  and source matching (requires ANTHROPIC_API_KEY; deterministic baseline always
  available as fallback).
- Hybrid claim extraction: LLM when available, deterministic rules as fallback.
- LLM-backed source matcher layered on top of the deterministic lexical baseline.
- Robust JSON extraction from LLM responses, handling markdown fences and surrounding text.
- Parallel provider queries via ThreadPoolExecutor for faster multi-provider searches.
- Per-provider rate limiter (1 s minimum interval) to avoid simultaneous API hammering.
- Rich progress spinners in `check` and `suggest` terminal output for long-running searches.
- Paragraph-final citation targeting: paragraph-final citations now upgrade their link
  confidence to 100 when attached to a claim in the final sentence (SPEC §9 rule 3).
- Ambiguous citation link warnings: citations linked to multiple claims in the same
  paragraph now emit `ambiguous_citation_link` warnings on each affected claim
  (SPEC §9 rule 5).
- `python -m citeguard` support via `__main__.py`.
- Unified `normalize_doi` canonical implementation (was duplicated across modules).
- Dynamic `User-Agent` headers derived from `__version__` across all providers.
