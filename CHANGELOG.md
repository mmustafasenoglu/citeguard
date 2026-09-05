# Changelog

All notable changes to citeguard will be documented here.

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
