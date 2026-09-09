# citeguard v1.0.1

**Audit citations in academic documents. Detect uncited claims, verify bibliography metadata, and estimate source support — from your terminal.**

[![CI](https://github.com/mmustafasenoglu/citeguard/actions/workflows/tests.yml/badge.svg)](https://github.com/mmustafasenoglu/citeguard/actions/workflows/tests.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`citeguard` is an open-source CLI and Python library for auditing citations in academic documents
(.md, .txt, .docx). It parses citations and bibliographies, searches academic providers for
metadata and source suggestions, and produces structured reports with deterministic health scores.

> **Important:** citeguard does **not** prove that a claim is true, detect plagiarism, or verify
> scientific correctness. It helps determine whether a real, relevant source can be found and
> whether that source appears to support the claim. All results require human review.

## Features

- Detect citation-worthy claims with type and severity classification
- Parse parenthetical, narrative, multi-source, numbered, and DOI citations
- Support `.md`, `.txt`, and `.docx` files including DOCX table cells
- Parse English and Turkish bibliography headings
- Search Semantic Scholar, Crossref, OpenAlex, and arXiv for source suggestions
- Verify bibliography metadata against Crossref (direct DOI lookup + bibliographic fallback)
- Compute a deterministic Citation Health Score
- Produce terminal, Markdown, and JSON reports (`--format both` for JSON + Markdown)
- Link claims to their citing references with sentence-position confidence scores
- Optional LLM integration for enhanced claim extraction and source matching
- **Multi-provider LLM support** — Anthropic, OpenAI, xAI/Grok, Groq, OpenRouter, NVIDIA NIM, and custom OpenAI-compatible endpoints
- Parallel provider queries for faster multi-provider searches
- Per-provider rate limiting to avoid API throttling
- Rich progress spinners during long-running searches
- **Evidence extraction** — sentence-level evidence passages from source abstracts, ranked by relevance
- **Entailment evaluation** — lexical contradiction signals and LLM-backed support classification
- **Evidence-aware scoring** — three-stage pipeline: metadata match → evidence → entailment → aggregate
- **CLI evidence controls** — `--show-evidence` to display evidence passages, `--require-evidence` to filter
- **Document similarity** — sentence-level overlap, lexical, and semantic similarity analysis against a corpus
- **Optional rewrite suggestions** — explicit, evidence-grounded rewording proposals via remote LLM (`citeguard rewrite`); never automatic, never verification

## Installation

```bash
pip install citeguard
```

Or install from source:

```bash
git clone https://github.com/mmustafasenoglu/citeguard.git
cd citeguard
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# Inspect document parsing (offline, no network calls)
citeguard inspect examples/example-paper.md

# Full citation audit
citeguard check examples/example-paper.md

# Suggest sources for uncited claims
citeguard suggest examples/example-paper.md

# Verify bibliography metadata against Crossref
citeguard verify examples/example-paper.md

# Offline mode — zero network calls, deterministic baseline only
citeguard check examples/example-paper.md --offline

# JSON output
citeguard check examples/example-paper.md --format json

# Markdown report to file
citeguard check examples/example-paper.md --format md --output report.md

# Both JSON and Markdown at once
citeguard check examples/example-paper.md --format both --output report

# Verbose mode with progress spinners
citeguard check examples/example-paper.md --verbose

# Show evidence passages for matched claims
citeguard check examples/example-paper.md --show-evidence

# Filter to only claims with evidence
citeguard check examples/example-paper.md --require-evidence

# Filter by severity
citeguard check examples/example-paper.md --severity high

# Limit extracted claims
citeguard check examples/example-paper.md --max-claims 5

# Document similarity analysis
citeguard similarity examples/example-paper.md --corpus path/to/corpus
```

### LLM-enhanced mode (optional)

citeguard supports multiple LLM providers. Set one API key to enable LLM-backed claim extraction
and entailment:

```bash
# Create .env template
citeguard init

# Pick your provider — just set one API key:

# Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
OPENAI_API_KEY=sk-...

# xAI / Grok
XAI_API_KEY=xai-...

# Groq
GROQ_API_KEY=gsk_...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# NVIDIA NIM
NVIDIA_API_KEY=nvapi-...

# Or use a local/custom endpoint (Ollama, LM Studio, vLLM, etc.)
CITEGUARD_LLM_PROVIDER=custom
CITEGUARD_LLM_BASE_URL=http://localhost:11434/v1
CITEGUARD_LLM_API_KEY=local
CITEGUARD_LLM_MODEL=qwen3
```

Provider is auto-detected from available API keys, or set explicitly:

```bash
CITEGUARD_LLM_PROVIDER=groq
CITEGUARD_LLM_MODEL=openai/gpt-oss-20b
```

When no API key is available, citeguard falls back to the deterministic offline baseline automatically.

## Commands

| Command | Description |
|---------|-------------|
| `citeguard check FILE` | Full audit: claims, verification, suggestions, health score |
| `citeguard verify FILE` | Verify bibliography metadata via Crossref |
| `citeguard suggest FILE` | Search academic providers for uncited claims |
| `citeguard inspect FILE` | Offline parsing and bibliography detection |
| `citeguard similarity FILE` | Document similarity analysis against a corpus |
| `citeguard init` | Create a local `.env` template for API keys |
| `citeguard llm doctor` | Test LLM provider connectivity |
| `citeguard llm list` | Display current LLM configuration (no secrets) |
| `citeguard rewrite FILE` | Propose evidence-grounded rewordings (explicit opt-in, remote LLM) |

### Common options

| Option | Description |
|--------|-------------|
| `--format terminal\|json\|md\|both` | Output format (default: terminal) |
| `--output FILE` | Write report to file |
| `--offline` | Skip all remote provider network calls |
| `--max-results INT` | Maximum provider results per query (1–20) |
| `--threshold INT` | Minimum confidence to report suggestions (0–100) |
| `--max-claims INT` | Limit number of extracted claims |
| `--severity high\|medium\|low` | Filter claims by severity |
| `--no-cache` | Skip provider response cache |
| `--verbose` | Show progress information with spinners |
| `--show-evidence` | Display evidence passages for matched claims |
| `--require-evidence` | Only show claims with evidence in priority review |

## Python API

citeguard exposes a stable Python API for programmatic use:

```python
from citeguard.api import (
    AuditOptions,
    AuditResult,
    audit_document,
    suggest_document,
    verify_document,
)

# Full audit
result: AuditResult = audit_document("paper.md", AuditOptions(offline=True))

# Suggest sources for uncited claims only
result: AuditResult = suggest_document("paper.md")

# Verify bibliography entries only
result: AuditResult = verify_document("paper.md")
```

Internal modules are not part of the public API. Only symbols re-exported from `citeguard.api`
are supported.

## Providers

citeguard queries academic metadata providers to resolve citations and suggest sources:

| Provider | Purpose |
|----------|---------|
| **Semantic Scholar** | Primary source search and metadata retrieval |
| **Crossref** | Bibliography verification (DOI lookup + bibliographic search) |
| **OpenAlex** | Open-access source search |
| **arXiv** | Preprint search |

Provider queries run in parallel with per-provider rate limiting (1 s minimum interval).
HTTP 429 and 5xx responses trigger exponential backoff (max 3 attempts). Provider failures do not
terminate the scan; citeguard warns and continues with remaining providers.

## Citation verification vs. claim support

citeguard treats these as separate questions:

1. **Source resolution** — can the referenced work be found?
2. **Metadata match** — do author, year, DOI, title, and other metadata align?
3. **Evidence extraction** — can relevant passages be found in source abstracts?
4. **Entailment** — do the evidence passages support, contradict, or leave the claim unsupported?
5. **Overall confidence** — a deterministic score derived from the previous signals.

A citation can therefore be successfully resolved while receiving a `contradicted` or `unrelated`
verdict. Verification confirms that metadata exists; support assessment evaluates whether the source
actually backs the claim.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Pass — no findings, or all checks above threshold |
| `1` | Findings — claims below health score threshold, or audit issues detected |
| `2` | Usage error — invalid arguments or missing file |
| `3` | Provider failure — a required online provider phase failed |

## Citation Health Score

The Citation Health Score is a **deterministic heuristic** for prioritizing review. It combines
citation coverage, verification ratio, support ratio, and bibliography consistency into a single
0–100 score. It is **not** a measure of scientific correctness or writing quality. A high score
does not mean claims are true; a low score does not mean they are false.

Calibration details and benchmark results are available in
[CALIBRATION.md](CALIBRATION.md) and the [`benchmarks/`](benchmarks/) directory.

## Rewrite suggestions (optional, remote LLM)

`citeguard rewrite FILE` proposes rewordings for verified claims — it never modifies
your document and never reinterprets verification results.

```bash
# Rewrite with the default clarify goal
citeguard rewrite paper.md --mode clarify --format json

# Soften wording to match partial evidence
citeguard rewrite paper.md --mode hedge --claim-index 0
```

What rewrite does and does not do:

- **Does**: propose a rewording grounded strictly in already-verified evidence
  (claim text, existing citation, matched source metadata, evidence passages).
- **Does not**: verify citations, repair citations, create bibliography records,
  change evidence scores, or edit any file.
- **Rewrite suggestions are generated text and do not constitute citation
  verification. Verification results remain authoritative.**

Modes: `clarify`, `hedge`, `align_with_evidence`, `remove_unsupported_detail`,
`citation_safe`. A rewrite is only attempted for claims with an evaluated
source, a supporting verdict, and non-empty evidence; otherwise the result is
`insufficient_evidence` with zero network calls.

`--offline` also disables the rewrite provider itself, even when credentials
are configured, so rewrite execution makes zero remote LLM calls. A provider's
successful response is only a proposal, not verification or acceptance. When
used in attribution reduction, proposals remain unvalidated candidates until
they pass citation, numeric, factual, semantic, and bidirectional-entailment
gates and deterministic ranking.

Configuration (all optional, `CITEGUARD_REWRITE_*` namespace):

- Credentials come only from the standard LLM environment variables
  (e.g. `ANTHROPIC_API_KEY`); nothing is hardcoded. Without credentials every
  request reports `unavailable` and the audit still completes.
- `CITEGUARD_REWRITE_PROVIDER` / `CITEGUARD_REWRITE_MODEL`: provider/model override.
- `CITEGUARD_REWRITE_TIMEOUT`: network timeout in seconds (default: 30).
- `CITEGUARD_REWRITE_MAX_CONTEXT`: max evidence characters per request (default: 2000).
- `CITEGUARD_REWRITE_ENABLED=0`: disables the rewrite command.

Privacy implications: only the minimum rewrite context is sent to the remote
provider — the single claim, its citation token, the matched source metadata
(title, authors, year, DOI), and bounded evidence passages. The full document,
bibliography, and unrelated claims are never sent. Document and evidence text
is treated as data, never as instructions.

## Privacy

citeguard is a local CLI. Network behavior depends on mode:

- **`--offline`**: zero network calls. All analysis uses the deterministic baseline with cached
  data only. No API keys are read.
- **Default mode**: limited content is sent to configured third-party APIs:
  - **Academic providers** (Semantic Scholar, Crossref, OpenAlex, arXiv): generated search
    queries derived from claims and bibliography metadata. The full document is not sent.
  - **LLM provider** (when configured): paragraphs are sent for claim extraction; claims plus
    source titles, authors, years, and abstracts are sent for entailment classification. The full
    document, bibliography, and API key are not included in prompts.
- **API keys** are never written to reports, logs, screenshots, or cache files.
- citeguard has no telemetry or usage analytics.

## Limitations

- LLM-based claim detection can miss or misclassify claims.
- Abstract-level evidence may be insufficient to establish claim support.
- Search providers can fail to index legitimate publications.
- `unresolved` does **not** mean `fake`.
- citeguard is **not** plagiarism detection.
- citeguard does **not** evaluate scientific validity or correctness.
- Human review is required for all results.
- Full-text evidence verification is not supported; only source abstracts are used.
- Citation Health Score is a review heuristic, not a measure of writing quality.

## Post-1.0 roadmap

_planned improvements will be documented here as they are scoped._

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test
expectations, and guidelines.

## Security

See [SECURITY.md](SECURITY.md). Never disclose API keys in GitHub issues or logs.

## License

MIT. See [LICENSE](LICENSE).
