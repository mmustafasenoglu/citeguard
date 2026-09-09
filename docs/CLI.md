# CiteGuard CLI and integration reference

This page keeps operational detail out of the project landing page. Run `citeguard COMMAND --help`
for the authoritative option list installed with your version.

## Commands

| Command | Purpose |
|---|---|
| `citeguard check FILE` | Run the full claim, citation, verification, and scoring audit |
| `citeguard inspect FILE` | Inspect parsing and bibliography detection without network calls |
| `citeguard suggest FILE` | Search academic providers for sources for uncited claims |
| `citeguard verify FILE` | Verify bibliography metadata with Crossref |
| `citeguard plagiarism FILE` | Run source-aware similarity and attribution review |
| `citeguard similarity FILE` | Analyze document similarity against a licensed corpus |
| `citeguard rewrite FILE` | Propose evidence-grounded wording for a verified claim |
| `citeguard improve-attribution FILE` | Preview or apply validated attribution-risk reductions |
| `citeguard init` | Create a local `.env` template |
| `citeguard llm doctor` | Test configured LLM connectivity |
| `citeguard llm list` | Show LLM configuration without secrets |
| `citeguard corpus index PATH` | Build a versioned local fingerprint/TF-IDF index |

Common `check` options:

| Option | Purpose |
|---|---|
| `--offline` | Forbid remote provider calls |
| `--format terminal\|json\|md\|both` | Choose report format |
| `--output FILE` | Write a report to disk |
| `--max-claims INT` | Limit extracted claims |
| `--severity high\|medium\|low` | Filter findings by severity |
| `--show-evidence` | Include evidence passages |
| `--require-evidence` | Show only claims with evidence |
| `--show-similarity --corpus PATH` | Include similarity analysis |
| `--threshold INT` | Set the minimum suggestion confidence |
| `--no-cache` | Skip provider cache reads and writes |
| `--verbose` | Show progress details |

## Optional LLM configuration

CiteGuard uses its deterministic baseline when no LLM is configured. To enable supported remote
LLM features, create a template and set one provider key:

```bash
citeguard init

# Choose one, for example:
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
XAI_API_KEY=...
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
NVIDIA_API_KEY=...
```

Provider selection can be explicit:

```bash
CITEGUARD_LLM_PROVIDER=groq
CITEGUARD_LLM_MODEL=openai/gpt-oss-20b
```

Custom OpenAI-compatible endpoints such as local Ollama, LM Studio, or vLLM instances are also
supported:

```bash
CITEGUARD_LLM_PROVIDER=custom
CITEGUARD_LLM_BASE_URL=http://localhost:11434/v1
CITEGUARD_LLM_API_KEY=local
CITEGUARD_LLM_MODEL=qwen3
```

Use `citeguard llm list` to inspect the selected configuration and `citeguard llm doctor` to test
connectivity. These commands do not print API keys.

## Academic providers

| Provider | Role |
|---|---|
| Semantic Scholar | Primary source search and metadata retrieval |
| Crossref | DOI lookup, bibliographic search, and metadata verification |
| OpenAlex | Academic source search |
| arXiv | Preprint search |

Provider failures do not end the whole scan. Requests use per-provider rate limiting; HTTP 429 and
5xx responses are retried with exponential backoff.

## Plagiarism review

`citeguard plagiarism` compares Markdown, TXT, or DOCX prose with actual text in supplied local
corpora, explicit files, or explicit URLs. Academic metadata alone never contributes to a
similarity score.

```bash
citeguard plagiarism thesis.docx \
  --corpus sources/ \
  --source appendix-source.txt \
  --offline \
  --format both --output thesis.plagiarism
```

Important options:

| Option | Purpose |
|---|---|
| `--corpus PATH` | Compare every supported file in a corpus directory; repeatable |
| `--source PATH` | Add a distinct explicit source; repeatable |
| `--source-url URL --online` | Fetch a bounded textual web source with redirect/type/size checks |
| `--discover-sources --online` | Search with bibliography metadata; compare returned abstracts only |
| `--include-quotes` | Include quoted overlap in the review-relevant score |
| `--include-bibliography` | Include bibliography overlap in the review-relevant score |
| `--exclude-small-matches N` | Ignore matches covering fewer than N words |
| `--exact-threshold`, `--lexical-threshold`, `--semantic-threshold` | Override centralized thresholds |
| `--semantic` | Enable the optional sentence-transformer candidate and classification pass |
| `--max-sources`, `--max-matches` | Bound source ingestion and report size |
| `--no-cache` | Disable URL and corpus-index cache reads and writes |

JSON reports use schema `1.0` and preserve passage/source offsets, separate exact/lexical/semantic
signals, quote/citation/bibliography state, attribution status, severity, source contribution, and
the exact configuration. Percentages derive from unique covered document words. A semantic match
covers a detected passage but is never represented as word-for-word alignment.

Build an explicit reusable index with:

```bash
citeguard corpus index sources/ --output sources.ctac
```

See [Plagiarism review methodology](PLAGIARISM.md) for scoring, calibration, safety, and corpus
limitations.

## Rewrite and attribution improvement

`citeguard rewrite FILE` returns a proposal only when a claim already has evaluated supporting
evidence. It does not verify citations, create bibliography entries, change evidence scores, or
edit the document.

```bash
citeguard rewrite paper.md --mode clarify --format json
citeguard rewrite paper.md --mode hedge --claim-index 0
```

Available modes are `clarify`, `hedge`, `align_with_evidence`, `remove_unsupported_detail`, and
`citation_safe`. Missing evidence produces `insufficient_evidence` without a provider call.

`citeguard improve-attribution` adds corpus comparison, validation gates, a separate output path,
and a rescan:

```bash
citeguard improve-attribution paper.md \
  --corpus corpus.txt --corpus-license CC0 --dry-run

citeguard improve-attribution paper.md \
  --from-report paper.plagiarism.json --dry-run

citeguard improve-attribution paper.md \
  --corpus corpus.txt --corpus-license CC0 \
  --apply --output paper.revised.md
```

Markdown and TXT replacements are exact and auditable. DOCX updates are limited to uniquely mapped
text contained in one ordinary run. Hyperlinks, fields, tracked changes, embedded objects,
ambiguous text, and unsafe multi-run spans are left unchanged for manual review.

## Python API

Only symbols exported from `citeguard.api` are part of the supported public API:

```python
from citeguard.api import AuditOptions, AuditResult, audit_document

result: AuditResult = audit_document(
    "paper.md",
    AuditOptions(offline=True),
)
```

The module also exports `suggest_document` and `verify_document` for focused workflows. Internal
modules can change without compatibility guarantees. Source-aware review is also available as
`scan_document(path, sources=[...], config=PlagiarismConfig(...))`; it returns a
`PlagiarismResult` and performs no network access with the default configuration.

## Network and privacy

| Mode | Behavior |
|---|---|
| `--offline` | Zero network calls; API keys are not read; required uncached local models fail closed |
| Default | Academic providers receive derived claim queries and bibliography metadata |
| LLM configured | Relevant paragraphs or bounded claim/evidence context may be sent to that provider |
| Plagiarism with local sources | Document text remains local |
| `plagiarism --source-url --online` | Only the explicit URL is fetched; the document is not uploaded |

The full document and bibliography are not included in LLM entailment prompts. Rewrite requests
send one claim, its citation token, matched source metadata, and bounded evidence passages. CiteGuard
has no telemetry, and API keys are not written to reports, logs, or cache files.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | No findings, or all checks are above the configured threshold |
| `1` | Audit findings require review |
| `2` | Invalid usage or missing input |
| `3` | A required online provider phase failed |

The Citation Health Score and process exit code prioritize review. Neither is a judgment of
scientific correctness.
