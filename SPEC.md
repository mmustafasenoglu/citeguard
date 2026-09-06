# citeguard — Technical Specification

## 1. Product definition

`citeguard` is an open-source command-line citation auditing tool for academic writing. It is designed to identify citation-worthy claims, suggest real academic sources for uncited claims, resolve existing citations, and estimate whether resolved sources support the claims they are attached to.

The system must never present source retrieval as proof that a claim is true. It is a research-assistance tool, not an automated academic approval mechanism.

## 2. Core problems

Academic writing workflows repeatedly encounter two failure modes:

1. **Missing citation** — a concrete factual, statistical, historical, comparative, causal, or quoted claim has no supporting source.
2. **Questionable citation** — a citation is present, but it is unclear whether the referenced work exists, whether the metadata is correct, or whether the work supports the surrounding claim.

LLM-assisted writing increases the second risk because plausible-looking but nonexistent or mismatched references can be generated.

## 3. v0.1 scope

v0.1 is scope-frozen around three user outcomes: **find**, **verify**, and **audit**.

### 3.1 Find

- Read `.md`, `.txt`, and `.docx` files.
- Extract citation-worthy verifiable claims.
- Classify each claim by type and severity.
- Search real academic metadata providers for uncited claims.
- Suggest only candidates above the configured reporting threshold.

### 3.2 Verify

- Detect author-year citations, numbered citations, and DOIs.
- Parse a simple bibliography section.
- Resolve existing author-year and DOI citations where possible.
- Separate source existence from metadata match and semantic claim support.
- Classify the claim-source relationship.

### 3.3 Audit

- Citation coverage.
- Verified / unresolved citation counts.
- Weak matches.
- Contradictions.
- Uncited high-severity claims.
- Duplicate DOI detection.
- Cited-but-not-listed and listed-but-not-cited checks.
- Deterministic Citation Health Score.
- Priority review list.

## 4. Non-goals for v0.1

The following are explicitly deferred:

- Full-text verification.
- Automatic citation insertion or document rewriting.
- Zotero/BibTeX synchronization.
- LaTeX `\\cite{}` parsing.
- OpenAlex and PubMed providers.
- Retraction checking.
- VS Code extension.
- GitHub Action / pre-commit integration.
- Multi-LLM support.
- Automatic numbered citation-to-bibliography resolution.

## 5. Architecture

```text
citeguard/
├── citeguard/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── extractor.py
│   ├── bibliography.py
│   ├── claims.py
│   ├── linking.py
│   ├── retrieval.py
│   ├── matcher.py
│   ├── evidence.py
│   ├── entailment.py
│   ├── verification.py
│   ├── scoring.py
│   ├── cache.py
│   ├── report.py
│   └── providers/
│       ├── base.py
│       ├── semantic_scholar.py
│       ├── crossref.py
│       └── arxiv.py
├── tests/
├── examples/
├── .github/
├── pyproject.toml
├── README.md
├── SPEC.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## 6. Data model

### Severity

- `high`
- `medium`
- `low`

### ClaimType

- `statistic`
- `causal`
- `comparative`
- `historical`
- `definition`
- `prior_work`
- `quotation`
- `general_fact`

### VerificationStatus

Verification status describes citation resolution, not semantic support:

- `verified`
- `unresolved`
- `not_found`
- `suggested`

### Verdict

Verdict describes the semantic relationship between a claim and a candidate source:

- `supported`
- `partially_supported`
- `contradicted`
- `unrelated`
- `insufficient_information`

A result such as `VERIFIED + CONTRADICTED` is valid and important: the reference exists, but does not support the attached claim.

### ExistingCitation

Fields:

- raw citation text
- author string when available
- year when available
- DOI when available
- numbered reference when available
- paragraph index
- character offset

### Claim

Fields:

- exact claim excerpt, capped at 25 words for model-generated excerpts
- optimized search query
- claim type
- severity
- paragraph index
- existing citation state
- linked citation(s)
- link confidence

### SourceCandidate

Fields:

- title
- authors
- year
- venue
- DOI
- URL
- abstract
- provider name
- arXiv ID when available

### MatchResult

Fields:

- candidate
- `source_exists: bool`
- `metadata_match_score: int`
- `claim_support_score: int`
- `evidence_list: list[Evidence]`
- deterministic `overall_confidence: int`
- verdict
- concise reasoning
- warnings

### Evidence

Fields:

- `text: str` — extracted sentence from source abstract
- `relevance_score: int` — 0–100 lexical relevance score
- `evidence_type: EvidenceType` — sentence type classification

### EvidenceType

- `supporting` — evidence that supports the claim
- `contradicting` — evidence that contradicts the claim
- `contextual` — relevant but not directly supporting or contradicting

### VerificationResult

Fields:

- claim
- verification status
- linked citation
- matched result
- suggestions
- warnings

## 7. Extraction

### Supported files

- Markdown: UTF-8 text.
- Plain text: UTF-8 text.
- DOCX: `python-docx` extraction of body and table-cell paragraphs in document order; nested
  tables are traversed and merged cells are emitted once.

### Citation detection

v0.1 supports three broad citation signals:

1. Parenthetical and narrative author-date citations, including semicolon-separated sources,
   organization authors, page locators, and `t.y.` / `n.d.` no-date markers.
2. Numbered citations such as `[12]` and `[12, 13]`.
3. DOI strings.

Citation detection is deliberately treated as a parser with false-positive/false-negative risk rather than a perfect citation-style engine.

## 8. Bibliography parsing

The parser searches for headings such as:

- `References`
- `Bibliography`
- `Works Cited`

The output attempts to recover author text, year, title, DOI, and an optional numeric reference label.

v0.1 consistency checks:

- duplicate DOI
- cited-not-listed
- listed-not-cited

Automatic numbered citation resolution is not enabled in v0.1.

## 9. Claim-citation linking

The linker must not assume that every citation in a paragraph supports every claim in that paragraph.

Deterministic proximity rules run first:

1. A citation inside the same sentence is linked to claims in that sentence.
2. A sentence-final citation can link to multiple claims in that sentence.
3. A paragraph-final citation first targets claims in the final sentence.
4. One citation may link to multiple claims.
5. Ambiguous cases produce an `ambiguous_citation_link` warning.

LLM-assisted structured linking may be used as a fallback when deterministic rules are inconclusive. The linker returns citation indexes and link confidence rather than free-form prose.

## 10. Claim extraction

Each non-bibliography paragraph is processed independently. The structured model output must include:

- claim text
- search query
- claim type
- severity
- citation requirement / existing-citation hint

The model must focus on externally verifiable statements such as statistics, research findings, direct quotations, historical facts, causal claims, and concrete comparative claims.

LLM-extracted claims are mapped back to the original paragraph via token overlap to detect citations that the LLM may have stripped from the claim text.  The sentence with the highest overlap is checked for citation markers; if found, the claim is tagged as already cited.

A single malformed model response must not abort the document run. The paragraph is skipped with a warning.

## 11. Providers

All academic providers implement a common protocol:

```python
class SourceProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[SourceCandidate]:
        ...
```

v0.1 providers:

- Semantic Scholar
- Crossref
- arXiv

Provider exceptions are handled internally or by the retrieval layer. A provider failure must not terminate the whole scan.

## 12. Retrieval strategy

For each claim:

1. Search Semantic Scholar.
2. If Semantic Scholar returns an identifier match or an exact normalized-title match,
   early-stop is allowed. Broader lexical similarity alone does not trigger early-stop before
   the semantic matcher exists.
3. Otherwise query Crossref and arXiv.
4. Normalize provider results.
5. Deduplicate candidates.
6. Match and score candidates.
7. If no adequate academic candidate is found, use model web-search fallback.

Recommended thresholds:

- `STRONG_MATCH_THRESHOLD = 80`
- default report threshold `= 60`

A score between 60 and 79 must not prevent additional providers from being searched.

Before the semantic matcher is implemented, retrieval order is deterministic. It combines query
token overlap with normalized-title sequence similarity, then uses provider priority, publication
year, normalized title, and DOI as stable tie-breakers. This rank is a retrieval heuristic and is
not a claim-support score.

## 13. Source deduplication

Canonical source identity uses this priority:

1. normalized DOI
2. normalized arXiv ID
3. normalized title + publication year

A source returned by several providers is reported once, while provider provenance may be retained internally.

Provider search results are cached after normalization. Cache identity includes the cache schema,
provider name, normalized query, and requested result limit. Higher layers can bypass both cache
reads and writes.

## 14. Matching and scoring

### Three-stage matching pipeline

The matcher implements a three-stage pipeline: metadata match → evidence → entailment → aggregate.

1. **Metadata match**: deterministic scoring of title, author, year, DOI overlap between claim candidate and source metadata.
2. **Evidence extraction**: extract candidate sentences from source abstracts and rank by lexical relevance to the claim.
3. **Entailment evaluation**: classify each evidence passage as supporting, contradicting, or contextual using lexical negation signals and optional LLM classifier.
4. **Aggregate**: combine metadata, evidence relevance, and entailment verdicts into a final `overall_confidence` score.

The offline lexical matcher can produce `partially_supported` at most. `supported` and `contradicted` verdicts require entailment evaluation.  The offline negation detector flags contradiction *risk* with elevated confidence but returns `insufficient_information`; only the LLM-backed entailment classifier can produce a final `contradicted` verdict.

### Metadata verification scoring

The deterministic metadata verifier scores bibliography entries against provider candidates before
semantic claim matching. Without a DOI, the available title, first-author, and year fields receive
weights of 40%, 35%, and 25%. When both records provide a DOI, DOI receives 60%, title 20%, author
15%, and year 5%. An explicit DOI mismatch caps the metadata score below the verification threshold.
Missing fields are excluded from the denominator. A score of 70 or higher is `verified`; a lower
best-candidate score is `unresolved`; and no provider candidate is `not_found`.

Crossref verification uses direct DOI lookup when a DOI is present and falls back to bibliographic
search when Crossref has no record for that DOI. These statuses describe metadata resolution only
and must not be presented as evidence that the source supports a claim.

The semantic matcher receives the claim plus available source metadata, especially title and abstract.

The model produces:

- metadata match score: 0–100
- claim support score: 0–100
- evidence list (ranked passages)
- verdict
- concise reasoning

`overall_confidence` is computed by application code. ``support_score`` is a
standalone signal (evidence + entailment, no metadata).  Metadata is combined
exactly once:

```python
# With entailment available
# support_score = evidence_relevance * 0.25 + entailment_score * 0.75
overall_confidence = round(
    metadata_match_score * 0.20
    + support_score * 0.80
)

# Without entailment (offline baseline)
# support_score = evidence_relevance (standalone)
overall_confidence = round(
    metadata_match_score * 0.40
    + support_score * 0.60
)
```

This prevents the model from inventing an opaque aggregate score and weights semantic support more heavily than bibliographic similarity.

## 15. Audit scoring

Citation Health Score is deterministic and heuristic. It must not be described as academic correctness.

Base score:

```python
health = (
    citation_coverage * 100 * 0.25
    + verification_ratio * 100 * 0.25
    + support_ratio * 100 * 0.20
    + evidence_coverage * 100 * 0.20
    + bibliography_consistency * 100 * 0.10
)
```

Penalties:

```python
health -= uncited_high_count * 2
health -= contradiction_count * 4
health -= unresolved_high_count * 2
health = max(0, min(round(health), 100))
```

The implementation must document the exact metric definitions used for each ratio.

## 16. Priority score

Risky items are displayed first:

```python
SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}

priority = severity_weight * 100 + (100 - confidence)
```

Higher scores are reviewed first.

## 17. CLI

Target commands:

```text
citeguard check FILE
citeguard verify FILE
citeguard suggest FILE
citeguard init
```

Target options:

```text
--output FILE
--format md|json|both
--threshold INT
--severity high|medium|low
--max-claims INT
--no-cache
--verbose
--show-evidence
--require-evidence
```

During development, `citeguard inspect FILE` is allowed as an offline parser diagnostic command.

## 18. Cache

Cache directory:

```text
.citeguard_cache/
```

Cache identity must include at least:

- cache schema version
- operation
- provider
- query/input identity
- model
- prompt version

This prevents stale LLM outputs from surviving prompt or model changes unnoticed.

## 19. Error handling

- HTTP 429 and transient 5xx responses: exponential backoff, maximum three total attempts.
- Provider network failure: warn and continue with the next provider.
- Invalid structured LLM output: skip that paragraph or candidate, warn, and continue.
- Unsupported document type: fail fast with a clear CLI error.
- API keys must never appear in logs or generated reports.

## 20. Reporting

### Terminal

Rich summary table with concise claim text, status, verdict, severity, and confidence.

### Markdown

Detailed report including candidate metadata, source links, scores, warnings, and citation formatting where metadata permits.

### JSON

Stable structured representation suitable for future editor and CI integrations.

The report begins with a Citation Health summary and a priority review list.

## 21. Testing

Offline unit tests:

- extractor
- bibliography parser
- linker rules
- cache
- scoring
- normalization/deduplication
- evidence extraction and ranking
- entailment evaluation (lexical negation signals)
- evidence-aware matcher pipeline

Mocked network tests:

- providers
- retrieval fallback
- verification
- full check/suggest pipeline with evidence integration

LLM calls must be mocked in the default test suite. Test runs must not spend API credits or depend on network availability.

Fixtures should cover:

- APA-like author-year citations
- IEEE-like numbered citations
- multiple citations
- DOI forms
- one citation for multiple claims
- ambiguous citation placement
- duplicate bibliography entries
- cited-not-listed
- listed-not-cited
- malformed citations

## 22. Privacy and security

v0.1 has no telemetry.

Users must be told which text may leave the local machine. API keys must never be embedded in reports or cached request headers.

## 23. Open-source project policy

- License: MIT for v0.1.
- Semantic versioning.
- Public test workflow on supported Python versions.
- Bug, feature, parser, and provider issue templates.
- `CONTRIBUTING.md`, `SECURITY.md`, and changelog maintained from the first public release.

## 24. v0.2 roadmap

- Numbered citation-to-bibliography resolution.
- Evidence quote extraction from source abstracts (shipped).
- Entailment evaluation with lexical negation signals and LLM classifier (shipped).
- Three-stage evidence-aware matching pipeline (shipped).
- `--show-evidence` / `--require-evidence` CLI options (shipped).
- Evidence-aware scoring with `evidence_coverage` metric (shipped).
- Retraction metadata.
- OpenAlex and PubMed / Europe PMC providers.
- Domain-aware provider routing.
- Source recency warnings.
- Primary-source preference.
- Full-text evidence verification.
- Zotero/BibTeX/LaTeX integrations.
- CI / pre-commit integration.
- Multi-LLM provider support.

## 25. Release definition for v0.1.0

v0.1.0 is releasable when:

- all three target commands execute end-to-end;
- the three academic providers are functional and fail gracefully;
- claim extraction and matcher outputs are structured;
- result deduplication is implemented;
- audit and priority scoring are deterministic;
- terminal, Markdown, and JSON reports work;
- no default test performs a live network or paid LLM request;
- README privacy and limitation language matches actual behavior;
- example output is reproducible from a committed example document.
