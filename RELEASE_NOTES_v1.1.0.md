# CiteGuard v1.1.0 release notes

## Highlights

CiteGuard 1.1.0 adds a source-aware `improve-attribution` workflow. It identifies attribution risk,
generates bounded candidates, validates them, and previews changes before any file is written.
Application is explicit and always targets a separate output path.

## Safety / integrity

- Candidate changes must retain citations and numeric facts, remain grounded in supporting evidence,
  reduce source overlap, and pass semantic and bidirectional-entailment checks.
- English and Turkish validation covers causality, modality, scope, evidence strengthening, and
  clause-local negation.
- Markdown and TXT application is transactional. DOCX application is limited to safe single-run
  text; protected or ambiguous structures fail closed. Bibliographies are not rewritten.
- Semantic-only matches remain separate from textual spans and textual-overlap percentages.

## Turkish capability

Turkish-aware normalization and grouped engineering benchmarks cover textual-overlap detection and
rewrite safety. `intfloat/multilingual-e5-small` is selected for Turkish semantic retrieval. The
locked textual-overlap engineering holdout reports precision 1.000, recall 0.875, and F1 0.933; the
semantic retrieval benchmark reports Recall@1 0.717, Recall@3 0.950, Recall@5 0.983, and MRR 0.836.
These task-specific results are not real-world plagiarism accuracy.

## CLI improvements

`citeguard improve-attribution` supports preview-first analysis and explicit `--apply --output`
operation for Markdown, TXT, and conservatively supported DOCX documents. JSON, Markdown, and
terminal reports describe plans, rejected candidates, validation evidence, and rescan metrics.

## Offline behavior

`--offline` prevents academic-provider calls, external rewrite-provider calls, and model downloads.
Cached local semantic and NLI models may be used. Missing required models fail closed instead of
silently substituting weaker evidence.

## Compatibility

Python 3.10, 3.11, and 3.12 remain supported. Core report schema version 4 is unchanged; the
attribution-reduction report schema remains version 2. Semantic and NLI dependencies remain optional.

## Known limitations

- CiteGuard assists human review; it does not prove claims true, detect plagiarism, or establish
  scientific correctness.
- Production NLI remains much stronger in English than Turkish. Turkish semantic retrieval is
  strong, but raw Turkish NLI remains only partially calibrated.
- Abstract-level evidence may be insufficient, and provider coverage can be incomplete.
- Engineering safety suites do not establish a population-level rewrite-safety rate.

## Verification

The release gate covers the offline test suite, Python 3.10-3.12 CI, deterministic benchmarks,
fresh wheel and sdist installation, CLI and DOCX smoke tests, zero-network enforcement, dependency
and secret audits, package-content inspection, and artifact hashing. Exact final artifact hashes are
recorded in the release-gate report after the final source commit is validated.
