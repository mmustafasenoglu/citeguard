# citeguard

**Find missing citations. Verify existing references. Audit academic writing from your terminal.**

[![CI](https://github.com/mmustafasenoglu/citeguard/actions/workflows/tests.yml/badge.svg)](https://github.com/mmustafasenoglu/citeguard/actions/workflows/tests.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`citeguard` is an open-source CLI for detecting citation-worthy claims, resolving existing academic references, and estimating whether a retrieved source actually supports the surrounding claim.

> **Important:** citeguard does **not** prove that a claim is true. It helps determine whether a real, relevant source can be found and whether that source appears to support the claim. All results require human review.

## Features

- Detect citation-worthy claims with type and severity classification
- Parse parenthetical, narrative, multi-source, numbered, and DOI citations
- Support `.md`, `.txt`, and `.docx` files including DOCX table cells
- Parse English and Turkish bibliography headings
- Search Semantic Scholar, Crossref, and arXiv for source suggestions
- Verify bibliography metadata against Crossref (direct DOI lookup + bibliographic fallback)
- Compute a deterministic Citation Health Score
- Produce terminal, Markdown, and JSON reports (`--format both` for JSON + Markdown)
- Link claims to their citing references with sentence-position confidence scores
- Optional LLM integration for enhanced claim extraction and source matching
- **Multi-provider LLM support** — Anthropic, OpenAI, xAI/Grok, Groq, OpenRouter, NVIDIA NIM, and custom OpenAI-compatible endpoints
- Parallel provider queries for faster multi-provider searches
- Per-provider rate limiting to avoid API throttling
- Rich progress spinners during long-running searches
- Full offline baseline — no API keys required for default operation
- **Evidence extraction** — sentence-level evidence passages from source abstracts, ranked by relevance
- **Entailment evaluation** — lexical contradiction signals and LLM-backed support classification
- **Evidence-aware scoring** — three-stage pipeline: metadata match → evidence → entailment → aggregate
- **CLI evidence controls** — `--show-evidence` to display evidence passages, `--require-evidence` to filter

## Installation

```bash
git clone https://github.com/mmustafasenoglu/citeguard.git
cd citeguard
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

For development:

```bash
git clone https://github.com/mmustafasenoglu/citeguard.git
cd citeguard
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
# Inspect document parsing (offline, no API keys needed)
citeguard inspect examples/example-paper.md

# Full citation audit
citeguard check examples/example-paper.md

# Suggest sources for uncited claims
citeguard suggest examples/example-paper.md

# Verify bibliography metadata against Crossref
citeguard verify examples/example-paper.md

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
```

### LLM-enhanced mode (optional)

citeguard supports multiple LLM providers. Set one API key to enable LLM-backed claim extraction and entailment:

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
| `citeguard inspect FILE` | Offline parsing and bibliography detection |
| `citeguard check FILE` | Full audit: claims, verification, health score |
| `citeguard suggest FILE` | Search academic providers for uncited claims |
| `citeguard verify FILE` | Verify bibliography metadata via Crossref |
| `citeguard init` | Create a local `.env` template for API keys |

### Common options

| Option | Description |
|--------|-------------|
| `--format terminal\|json\|md\|both` | Output format (default: terminal) |
| `--output FILE` | Write report to file |
| `--max-results INT` | Maximum provider results per query (1–20) |
| `--threshold INT` | Minimum confidence to report suggestions (0–100) |
| `--max-claims INT` | Limit number of extracted claims |
| `--severity high\|medium\|low` | Filter claims by severity |
| `--no-cache` | Skip provider response cache |
| `--verbose` | Show progress information with spinners |
| `--show-evidence` | Display evidence passages for matched claims |
| `--require-evidence` | Only show claims with evidence in priority review |

## Example: check output

```
Citation Health Score: 30/100

┌──────────────────────────────────────────┐
│           Audit Summary                  │
├─────────────────────────────┬────────────┤
│ Total claims                │            2 │
│ Claims requiring citations  │            1 │
│ Cited claims                │            1 │
│ Verified citations          │            0 │
│ Weak matches                │            1 │
│ Uncited high-severity claims│            0 │
│ Contradictions              │            0 │
│ Bibliography issues         │            0 │
│ Citation coverage           │       50.0%  │
│ Verification ratio          │        0.0%  │
│ Support ratio               │        0.0%  │
│ Bibliography consistency    │      100.0%  │
└─────────────────────────────┴────────────┘

Priority review list
  LOW (confidence: 43) - Large language models can generate references that
  appear plausible but require independent verification. [verdict: insufficient_information]
```

Example output files are committed under `examples/`:

- `example-inspect.json` / `example-inspect.md` — offline inspection
- `example-check.json` / `example-check.md` — full audit

Reproduce them with:

```bash
citeguard inspect examples/example-paper.md --format both --output examples/example-inspect
citeguard check examples/example-paper.md --format both --output examples/example-check
```

## Reliability model

citeguard treats these as separate questions:

1. **Source resolution** — can the referenced work be found?
2. **Metadata match** — do author, year, DOI, title, and other metadata align?
3. **Evidence extraction** — can relevant passages be found in source abstracts?
4. **Entailment** — do the evidence passages support, contradict, or leave the claim unsupported?
5. **Overall confidence** — a deterministic score derived from the previous signals.

A citation can therefore be successfully resolved while receiving a `contradicted` or `unrelated` verdict.

## Privacy

citeguard is a local CLI. The analysis pipeline sends limited content to configured third-party APIs:

- **Academic providers** (Semantic Scholar, Crossref, arXiv): generated search queries derived
  from claims and bibliography metadata. The full document is not sent.
- **LLM provider** (when configured): paragraphs are sent for claim extraction; claims plus
  source titles, authors, years, and abstracts are sent for entailment classification. Supported
  providers: Anthropic, OpenAI, xAI/Grok, Groq, OpenRouter, NVIDIA NIM, and custom endpoints.
  The full document, bibliography, and API key are not included in prompts.
- **API keys** are never written to reports, logs, screenshots, or cache files.

citeguard has no telemetry or usage analytics.

## Limitations

- LLM-based claim detection can miss or misclassify claims.
- Abstract-level evidence may be insufficient to establish claim support.
- Search providers can fail to index legitimate publications.
- `unresolved` does **not** mean `fake`.
- Numbered citation-to-bibliography resolution is deferred beyond v0.1.
- Full-text evidence verification is not part of v0.1.
- Citation Health Score is a heuristic prioritization metric, not a measure of scientific correctness.

## Roadmap

See [SPEC.md](SPEC.md) for the full technical specification.

**v0.2 in progress:**
- Evidence extraction from source abstracts (shipped)
- Entailment evaluation with lexical + LLM classifier (shipped)
- Three-stage evidence-aware matching pipeline (shipped)
- `--show-evidence` / `--require-evidence` CLI options (shipped)
- OpenAlex / PubMed provider support
- Retraction metadata
- Full-text evidence verification

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test expectations, and guidelines.

## Security

See [SECURITY.md](SECURITY.md). Never disclose API keys in GitHub issues or logs.

## License

MIT. See [LICENSE](LICENSE).
