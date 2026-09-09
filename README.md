# CiteGuard

### Static analysis for academic writing.

Find unsupported claims, verify citations, inspect source overlap, and safely improve
attribution — from your terminal.

[![Release](https://img.shields.io/github/v/release/mmustafasenoglu/citeguard?display_name=tag)](https://github.com/mmustafasenoglu/citeguard/releases/latest)
[![CI](https://github.com/mmustafasenoglu/citeguard/actions/workflows/tests.yml/badge.svg)](https://github.com/mmustafasenoglu/citeguard/actions/workflows/tests.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

```bash
citeguard check paper.md
```

Real report excerpt generated from [`examples/example-paper.md`](examples/example-paper.md)
([full output](examples/example-check.md)):

```text
Citation Health Score: 30/100

Total claims detected:       2
Claims requiring citations:  1
Cited claims:                 1
Unresolved citations:        1

Priority review
LOW · Large language models can generate references that appear plausible
      but require independent verification.
```

[Install the current release](#install) · [Try the example](#quick-start) ·
[See the evidence](#measured-not-marketed)

## Follow the evidence, not just the reference

CiteGuard connects the full review path instead of stopping when it finds a bibliography entry:

```text
Document → Claims → Citations → Sources → Evidence
         → Attribution risk → Safe revision → Rescan
```

- **Find the review surface.** Extract citation-worthy claims and connect citations to the
  sentences they are meant to support.
- **Separate existence from support.** Resolve source metadata, extract relevant abstract
  passages, and assess whether those passages support, contradict, or do not establish a claim.
- **Make risk inspectable.** Prioritize uncited claims, weak matches, bibliography problems, and
  source overlap in terminal, Markdown, or JSON reports.
- **Improve attribution conservatively.** Preview bounded revisions, preserve citations and factual
  constraints, write to a new file, and rescan the actual result.

CiteGuard supports Markdown, plain text, and DOCX. Its deterministic baseline works without an LLM;
academic metadata providers and supported LLM backends add retrieval and entailment signals when
configured.

## Install

Python 3.10–3.12 is supported. CiteGuard is not yet published on PyPI, so install v1.1.0 from the
GitHub release:

```bash
python -m pip install \
  https://github.com/mmustafasenoglu/citeguard/releases/download/v1.1.0/citeguard-1.1.0-py3-none-any.whl
```

Or install the current source checkout:

```bash
git clone https://github.com/mmustafasenoglu/citeguard.git
cd citeguard
python -m pip install .
```

Optional local models remain separate so the core installation stays lightweight:

```bash
python -m pip install "citeguard[semantic]"  # semantic similarity
python -m pip install "citeguard[nli]"       # local NLI
```

## Quick start

```bash
# Run the complete citation audit
citeguard check examples/example-paper.md

# Run a deterministic scan with zero network calls
citeguard check examples/example-paper.md --offline

# Save machine-readable and human-readable reports
citeguard check paper.md --format both --output report

# Review exact, lexical, and optional semantic overlap against known sources
citeguard plagiarism paper.md --corpus licensed-sources/ --format both --output review

# Inspect the evidence attached to source matches
citeguard check paper.md --show-evidence
```

The core commands follow the questions reviewers ask:

| Question | Command |
|---|---|
| What claims and citations are in this document? | `citeguard inspect FILE` |
| Which findings need review? | `citeguard check FILE` |
| Can an uncited claim be matched to a source? | `citeguard suggest FILE` |
| Does bibliography metadata match a published record? | `citeguard verify FILE` |
| Which passages overlap available sources, and is attribution adequate? | `citeguard plagiarism FILE --corpus PATH` |
| Where does this document overlap a licensed corpus? | `citeguard similarity FILE --corpus PATH` |
| Can attribution be improved without overwriting the source? | `citeguard improve-attribution FILE --corpus PATH` |

See the [CLI and integration reference](docs/CLI.md) for options, providers, LLM configuration,
the Python API, and exit codes.

## One result, several distinct signals

Finding a source is not the same as showing that it supports a claim. CiteGuard keeps these stages
separate:

| Stage | What it asks |
|---|---|
| Source resolution | Can the referenced work be found? |
| Metadata match | Do title, authors, year, DOI, and related fields align? |
| Evidence extraction | Which abstract passages are relevant to the claim? |
| Entailment | Does available evidence support, contradict, or leave the claim unresolved? |
| Review priority | Which findings deserve attention first? |

The Citation Health Score combines deterministic audit signals into a review queue. It is a
heuristic, not a grade for scientific correctness or writing quality.

## Safe attribution workflow

Start with an auditable plagiarism review. Local comparison is the default and makes no network
calls:

```bash
citeguard plagiarism paper.md \
  --corpus licensed-sources/ --offline \
  --format both --output paper.plagiarism
```

The raw score covers all matched document words. The review-relevant score excludes generic
phrases and, by default, quoted material and bibliography entries. Source contributions allocate
each covered word once, so duplicate sources do not multiply the score. Use `--source` for
individual files, `--source-url ... --online` for explicit web sources, and `--semantic` for the
optional local embedding pass.

Preview is the default:

```bash
citeguard improve-attribution paper.md \
  --corpus licensed-corpus.txt --corpus-license CC-BY --dry-run

# Or reuse the local sources recorded by the plagiarism JSON report
citeguard improve-attribution paper.md \
  --from-report paper.plagiarism.json --dry-run
```

Applying a validated candidate always requires a different output path:

```bash
citeguard improve-attribution paper.md \
  --corpus licensed-corpus.txt --corpus-license CC-BY \
  --apply --output paper.revised.md
```

CiteGuard never overwrites the source document. Candidates must preserve citations, numbers, and
factual constraints; reduce measured overlap; pass semantic and bidirectional-entailment gates; and
survive a rescan of the written output. Ambiguous or unsupported edits fail closed for manual
review.

## Measured, not marketed

The repository includes reproducible results and the fixtures or dataset manifests needed to
interpret them.

| Evaluation | Recorded result | Scope |
|---|---:|---|
| Frozen core release gate | Passed all 6 metric gates | Synthetic, deterministic fixtures with mocked providers |
| Turkish textual-overlap detection v2 | Precision 1.000 · Recall 0.875 · F1 0.933 | 168-case group-isolated engineering holdout |
| Turkish semantic candidate retrieval | Recall@1 0.717 · Recall@3 0.950 · Recall@5 0.983 | 60 close-paraphrase queries; retrieval only |
| Offline reproducibility | 100% on identical inputs, cache, and options | Deterministic benchmark environment |

Read [CALIBRATION.md](CALIBRATION.md), the [external benchmark protocol](benchmarks/BENCHMARKS.md),
and the [versioned benchmark results](benchmarks/results/) before interpreting these numbers.
They are engineering evidence, not scientific validation or real-world plagiarism accuracy.

## Privacy and control

- `--offline` permits zero network calls and does not read API keys.
- Default provider searches send derived queries and citation metadata, not the full document.
- LLM-backed features are optional and only run with a configured provider.
- CiteGuard has no telemetry and does not put secrets in reports or caches.
- Rewrite suggestions never change verification results; `--apply` writes only to an explicit,
  different output path and never overwrites the source.

Exact network behavior and provider configuration are documented in the
[CLI reference](docs/CLI.md#network-and-privacy).

## Boundaries

CiteGuard assists human review. It does **not**:

- prove that a claim is true or scientifically valid;
- treat an unresolved citation as fake;
- replace a reference manager;
- issue a definitive plagiarism verdict or estimate a proprietary detector score;
- guarantee that an LLM classification or rewrite is correct.

CiteGuard provides source-aware document similarity and plagiarism-review signals over the
corpora and sources available to it. It does not reproduce Turnitin's proprietary corpus or
guarantee equivalence to commercial plagiarism scores.

Evidence is currently abstract-level, provider coverage varies, and Turkish production NLI is less
capable than English. Human review remains required.

## Project

- [Release notes](CHANGELOG.md)
- [Calibration and benchmark scope](CALIBRATION.md)
- [CLI and integration reference](docs/CLI.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Issue tracker](https://github.com/mmustafasenoglu/citeguard/issues)

Licensed under the [MIT License](LICENSE).
