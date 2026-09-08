# citeguard — Technical Specification (v1.0.0)

## 1. Product definition

`citeguard` is an open-source command-line citation auditing tool for academic writing. It
identifies citation-worthy claims, suggests real academic sources for uncited claims, resolves
existing citations, estimates whether resolved sources support the claims they are attached to,
and detects textual similarity against a reference corpus.

citeguard also exposes a stable Python API (`citeguard.api`) for programmatic use.

The system must never present source retrieval as proof that a claim is true. It is a
research-assistance tool, not an automated academic approval mechanism.

## 2. Core problems

Academic writing workflows repeatedly encounter two failure modes:

1. **Missing citation** — a concrete factual, statistical, historical, comparative, causal, or
   quoted claim has no supporting source.
2. **Questionable citation** — a citation is present, but it is unclear whether the referenced
   work exists, whether the metadata is correct, or whether the work supports the surrounding
   claim.

LLM-assisted writing increases the second risk because plausible-looking but nonexistent or
mismatched references can be generated.

A third failure mode emerges in automated writing pipelines:

3. **Textual overlap** — a sentence is too close to a source text without proper attribution,
   whether direct quotation or paraphrase.

## 3. User outcomes

citeguard is built around four user outcomes: **find**, **verify**, **audit**, and **similar**.

### 3.1 Find

- Read `.md`, `.txt`, and `.docx` files.
- Extract citation-worthy verifiable claims (LLM-assisted when available, deterministic
  fallback when offline).
- Classify each claim by type and severity.
- Search real academic metadata providers for uncited claims.
- Suggest only candidates above the configured reporting threshold.

### 3.2 Verify

- Detect author-year citations, numbered citations, and DOIs.
- Parse a bibliography section (References, Bibliography, Works Cited).
- Resolve existing author-year and DOI citations where possible.
- Separate source existence from metadata match and semantic claim support.
- Classify the claim-source relationship.

### 3.3 Audit

- Citation coverage.
- Verified / unresolved / partially verified citation counts.
- Weak matches.
- Contradictions.
- Uncited high-severity claims.
- Duplicate DOI detection.
- Cited-but-not-listed and listed-but-not-cited checks.
- Bibliography metadata verification via Crossref and OpenAlex.
- Deterministic Citation Health Score.
- Priority review list.

### 3.4 Similar

- Compare document sentences against a reference corpus.
- Detect exact overlap, near-duplicate, lexical similarity, and semantic similarity.
- Compute per-sentence attribution risk (citation-aware, semantic-aware).
- Aggregate character-span-based overall similarity percentage.

## 4. Non-goals

The following are explicitly out of scope:

- Full-text verification (only abstracts are used for evidence).
- Automatic citation insertion or document rewriting.
- Zotero/BibTeX synchronization.
- LaTeX `\cite{}` parsing.
- Retraction checking.
- VS Code extension.
- GitHub Action / pre-commit integration.
- PubMed / Europe PMC providers.
- Domain-aware provider routing.
- Primary-source preference heuristics.

## 5. Architecture

```text
citeguard/
├── citeguard/
│   ├── __init__.py              # version string
│   ├── __main__.py              # python -m citeguard entry point
│   ├── api.py                   # stable Python API (AuditOptions, AuditResult)
│   ├── audit.py                 # product-level audit orchestration
│   ├── cli.py                   # Click CLI commands
│   ├── config.py                # global defaults, Settings, LLMProviderSettings
│   ├── models.py                # shared data models and enums
│   ├── extractor.py             # document parsing, citation detection
│   ├── bibliography.py          # bibliography detection, entry parsing
│   ├── claims.py                # deterministic claim extraction
│   ├── linking.py               # citation-to-sentence linking
│   ├── matcher.py               # deterministic + LLM-assisted matching
│   ├── evidence.py              # evidence extraction and ranking
│   ├── entailment.py            # entailment evaluation (lexical + LLM)
│   ├── verification.py          # bibliography verification
│   ├── retrieval.py             # provider orchestration, ranking, deduplication
│   ├── scoring.py               # AuditMetrics, ProductMetrics, health score
│   ├── cache.py                 # JSON response cache
│   ├── report.py                # terminal, Markdown, JSON formatters
│   ├── llm.py                   # multi-provider LLM integration
│   ├── llm_backends.py          # LLMBackend protocol, provider implementations
│   ├── llm_router.py            # resilient router (retry, circuit breaker, failover)
│   ├── llm_cache.py             # LLM response cache with prompt-version awareness
│   ├── sentence_splitter.py     # sentence boundary detection
│   ├── providers/
│   │   ├── base.py              # SourceProvider protocol, HTTP helpers
│   │   ├── semantic_scholar.py  # Semantic Scholar adapter
│   │   ├── crossref.py          # Crossref adapter
│   │   ├── openalex.py          # OpenAlex adapter
│   │   └── arxiv.py             # arXiv adapter
│   ├── corpus/
│   │   ├── loader.py            # load_similarity_corpus (library-level)
│   │   ├── ingest.py            # document ingestion
│   │   ├── deduplicate.py       # entry deduplication
│   │   ├── normalize.py         # text normalization
│   │   ├── licenses.py          # corpus license metadata
│   │   ├── synthetic.py         # synthetic corpus generation for tests
│   │   └── models.py            # CorpusMetadata, IndexedPassage
│   └── similarity/
│       ├── engine.py            # SimilarityEngine orchestration
│       ├── index.py             # SimilarityIndex (build, retrieve)
│       ├── fingerprint.py       # BLAKE2b fingerprinting, winnowing
│       ├── lexical.py           # TF-IDF, char-ngram Jaccard
│       ├── normalize.py         # offset remapping
│       ├── reranker.py          # ranking score computation
│       ├── fusion.py            # RRF candidate fusion
│       ├── models.py            # MatchType, RiskLevel, SimilarityMatch
│       └── embeddings/          # semantic embedding backend (optional)
├── tests/
├── examples/
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

- `verified` — metadata match score >= 70 (or DOI match).
- `partially_verified` — metadata match score >= 50 but < 70.
- `unresolved` — best candidate below partial threshold, or no provider candidate.
- `not_found` — no provider returned a candidate.
- `suggested` — uncited claim with a candidate above the suggestion threshold.
- `provider_error` — all providers failed for this entry.
- `metadata_mismatch` — DOI present but DOI mismatch between bibliography and provider.

### Verdict

Verdict describes the semantic relationship between a claim and a candidate source:

- `supported`
- `partially_supported`
- `contradicted`
- `unrelated`
- `insufficient_information`

A result such as `VERIFIED + CONTRADICTED` is valid and important: the reference exists, but
does not support the attached claim.

### SourceType

- `primary` — research article, original research, clinical trial, etc.
- `secondary` — review, editorial, letter, commentary, retraction, etc.
- `unknown` — work type not recognized.

### MatchType (similarity)

- `exact` — exact character-level overlap above threshold.
- `near_duplicate` — very high similarity but not exact.
- `lexical_overlap` — significant lexical similarity.
- `semantic_overlap` — high semantic cosine but low lexical overlap.
- `unmatched` — below all thresholds (excluded from results).

### RiskLevel (similarity)

- `high` — high overlap without citation, or wrong citation.
- `medium` — semantic overlap without citation, or quoted text with matching citation.
- `low` — matching citation present, or quoted text with matching citation.
- `none` — no match or bibliography sentence excluded.

### ExistingCitation

Fields:

- raw citation text
- author string when available
- year when available
- DOI when available
- numbered reference when available
- paragraph index
- character offset
- `no_date` flag

### Claim

Fields:

- exact claim excerpt, capped at 25 words for model-generated excerpts
- optimized search query
- claim type
- severity
- paragraph index
- existing citation state
- linked citation(s) — single and list
- link confidence
- warnings

### SourceCandidate

Fields:

- title
- authors
- year
- venue
- DOI
- URL
- abstract
- provider name (`source_api`)
- arXiv ID when available
- work type
- source APIs list (after deduplication merge)
- provider records (per-provider metadata for conflict detection)

### Evidence

Fields:

- `text` — extracted sentence from source abstract
- `source_title` — title of the source
- `source_api` — provider that supplied the source
- `evidence_type` — `ABSTRACT` or `FULL_TEXT`
- `section` — section name when available
- `page` — page number when available
- `lexical_score` — 0–100 lexical relevance score
- `semantic_score` — optional semantic cosine score
- `entailment_score` — optional entailment confidence
- `verdict` — preliminary verdict from evidence stage

### MatchResult

Fields:

- candidate
- `source_exists: bool`
- `metadata_match_score: int`
- `claim_support_score: int`
- `overall_confidence: int`
- `verdict`
- `reasoning`
- `warnings`
- `evidence: list[Evidence]`
- `entailment_score: int | None`
- `entailment_verdict: Verdict | None`

### VerificationResult

Fields:

- claim
- verification status
- linked citation
- matched result
- suggestions
- warnings

### ReferenceVerification

Fields:

- `entry_index`
- `entry: BibliographyEntry`
- `query: str`
- `status: VerificationStatus`
- `candidate: SourceCandidate | None`
- `scores: MetadataScores | None`
- `warnings`
- `recency_warning: str | None`
- `source_type: SourceType`
- `provider_conflicts: list[str]`

### MetadataScores

Fields:

- `author: int | None`
- `year: int | None`
- `title: int | None`
- `doi: int | None`
- `overall: int`

### Sentence

Fields:

- `text` — original sentence text
- `normalized_text` — lowercased, whitespace-normalized
- `paragraph_index`
- `sentence_index`
- `start_offset` — paragraph-relative
- `end_offset` — paragraph-relative
- `citations` — linked ExistingCitation objects
- `is_bibliography` — True if inside bibliography section

### EnrichedDocument

Fields:

- `path`
- `paragraphs: list[Paragraph]`
- `sentences: list[Sentence]`
- `citations`
- `bibliography_entries`
- `bibliography_start_index`

## 7. Extraction

### Supported files

- Markdown: UTF-8 text.
- Plain text: UTF-8 text.
- DOCX: `python-docx` extraction of body and table-cell paragraphs in document order; nested
  tables are traversed and merged cells are emitted once.

### Citation detection

citeguard supports three broad citation signals:

1. Parenthetical and narrative author-date citations, including semicolon-separated sources,
   organization authors, page locators, and `t.y.` / `n.d.` no-date markers.
2. Numbered citations such as `[12]` and `[12, 13]`.
3. DOI strings.

Citation detection is deliberately treated as a parser with false-positive/false-negative risk
rather than a perfect citation-style engine.

## 8. Bibliography parsing

The parser searches for headings such as:

- `References`
- `Bibliography`
- `Works Cited`

The output attempts to recover author text, year, title, DOI, and an optional numeric reference
label.

Consistency checks:

- duplicate DOI
- cited-not-listed
- listed-not-cited

Numbered citation-to-bibliography resolution is supported via `resolve_numbered_citations`.

## 9. Claim-citation linking

The linker must not assume that every citation in a paragraph supports every claim in that
paragraph.

Deterministic proximity rules run first:

1. A citation inside the same sentence is linked to claims in that sentence.
2. A sentence-final citation can link to multiple claims in that sentence.
3. A paragraph-final citation first targets claims in the final sentence.
4. One citation may link to multiple claims.
5. Ambiguous cases produce an `ambiguous_citation_link` warning.

Link confidence: 100 for sentence-final, 80 for mid-sentence, upgraded to 100 for
paragraph-final.

LLM-assisted structured linking may be used as a fallback when deterministic rules are
inconclusive. The linker returns citation indexes and link confidence rather than free-form
prose.

## 10. Claim extraction

Each non-bibliography paragraph is processed independently. The structured model output must
include:

- claim text
- search query
- claim type
- severity
- citation requirement / existing-citation hint

The model must focus on externally verifiable statements such as statistics, research findings,
direct quotations, historical facts, causal claims, and concrete comparative claims.

LLM-extracted claims are mapped back to the original paragraph via token overlap to detect
citations that the LLM may have stripped from the claim text. The sentence with the highest
overlap is checked for citation markers; if found, the claim is tagged as already cited.

A single malformed model response must not abort the document run. The paragraph is skipped
with a warning.

When no LLM provider is configured or offline mode is active, deterministic claim extraction
rules are used as a fallback.

## 11. Providers

All academic providers implement a common protocol:

```python
class SourceProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int = 5) -> list[SourceCandidate]:
        ...
```

Providers:

- Semantic Scholar
- Crossref
- OpenAlex
- arXiv

Provider exceptions are handled internally or by the retrieval layer. A provider failure must
not terminate the whole scan.

HTTP errors: exponential backoff with up to 3 total attempts for 429 and 5xx responses.
Per-provider rate limiting: 1 second minimum interval between calls.

## 12. Retrieval strategy

For each claim or bibliography entry:

1. Search Semantic Scholar (first provider, sequential).
2. If Semantic Scholar returns a strong match (DOI, arXiv ID, or exact title above threshold),
   early-stop.
3. Otherwise query Crossref, OpenAlex, and arXiv in parallel via `ThreadPoolExecutor`.
4. Normalize provider results.
5. Deduplicate candidates by canonical identity (DOI > arXiv ID > title+year).
6. Rank candidates by relevance score with provider priority, year, title, and DOI as stable
   tie-breakers.

Thresholds:

- `STRONG_MATCH_THRESHOLD = 80`
- `DEFAULT_THRESHOLD = 60` (suggestion reporting)
- `WEAK_MATCH_THRESHOLD = 60`

## 13. Source deduplication

Canonical source identity uses this priority:

1. normalized DOI
2. normalized arXiv ID
3. normalized title + publication year

A source returned by several providers is reported once, with provider provenance preserved in
`source_apis` and `provider_records` fields.

Provider search results are cached after normalization. Cache identity includes the cache
schema version, provider name, normalized query, and requested result limit. Higher layers can
bypass both cache reads and writes.

## 14. Matching and scoring

### Three-stage matching pipeline

The matcher implements a three-stage pipeline: metadata match -> evidence -> entailment ->
aggregate.

1. **Metadata match**: deterministic scoring of query-token overlap with candidate title.
2. **Evidence extraction**: extract candidate sentences from source abstracts, rank by lexical
   relevance to the claim (coverage 60%, sequence similarity 20%, overlap 20%).
3. **Entailment evaluation**: classify each evidence passage as supporting, contradicting, or
   contextual using lexical negation signals and optional LLM classifier.
4. **Aggregate**: combine metadata, evidence relevance, and entailment verdicts into a final
   `overall_confidence` score.

The offline lexical matcher can produce `partially_supported` at most. `supported` and
`contradicted` verdicts require entailment evaluation. The offline negation detector flags
contradiction *risk* with elevated confidence but returns `insufficient_information`; only the
LLM-backed entailment classifier can produce a final `contradicted` verdict.

### Metadata verification scoring

The deterministic metadata verifier scores bibliography entries against provider candidates.

Without a DOI, available title, first-author, and year fields receive weights of 40%, 35%,
and 25%. When both records provide a DOI, DOI receives 60%, title 20%, author 15%, and year
5%. An explicit DOI mismatch caps the metadata score at 49. Missing fields are excluded from
the denominator.

Thresholds:

- `VERIFIED_METADATA_THRESHOLD = 70` -> `verified`
- `_PARTIAL_METADATA_THRESHOLD = 50` -> `partially_verified`
- Below 50 -> `unresolved`

### Support score computation

```python
# With entailment available
support_score = evidence_relevance * 0.25 + entailment_score * 0.75
overall_confidence = round(
    metadata_match_score * 0.20
    + support_score * 0.80
)

# Without entailment (offline baseline)
support_score = evidence_relevance (standalone)
overall_confidence = round(
    metadata_match_score * 0.40
    + support_score * 0.60
)
```

This prevents the model from inventing an opaque aggregate score and weights semantic support
more heavily than bibliographic similarity.

## 15. Audit scoring

### ProductMetrics

The audit produces `ProductMetrics` with explicit availability semantics. Ratio fields are
`None` when the underlying evaluation could not be performed (e.g. offline cache miss, total
provider failure), which is distinct from a measured perfect score.

Metric definitions:

- `citation_coverage` = cited_claims / total_claims (1.0 when no claims).
- `verification_ratio` = VERIFIED bib entries / evaluated bib entries; `None` when entries
  exist but none could be evaluated. PARTIALLY_VERIFIED never counts.
- `support_ratio` = cited claims with >= 1 supported resolved source / cited claims
  evaluated for support; `None` when none were evaluated.
- `evidence_coverage` = evaluated cited claims with usable evidence / evaluated cited claims;
  `None` when none were evaluated.
- `bibliography_consistency` = 1 - issues / max(entries + distinct_citations, 1), clamped
  to 0..1.

### Citation Health Score

The health score is deterministic and heuristic. It must not be described as academic
correctness.

```python
health = (
    citation_coverage * 100 * 0.25
    + verification_ratio * 100 * 0.25
    + support_ratio * 100 * 0.20
    + bibliography_consistency * 100 * 0.10
    + evidence_coverage * 100 * 0.20
)
```

Penalties:

```python
health -= uncited_high_count * 2
health -= contradiction_count * 4
health -= unresolved_high_count * 2
health = max(0, min(round(health), 100))
```

When any required component is unavailable (`None`), `health_score` is `None` and
`health_score_complete` is `False`.

## 16. Priority review queue

Items are sorted by a deterministic priority ordering:

1. Contradiction on a cited claim (group 0).
2. Unresolved/mismatched cited claim, high/medium severity (group 1).
3. Uncited high-severity claim (group 2).
4. Uncited medium-severity claim (group 3).
5. Low-severity findings (group 4).
6. Informational: metadata conflicts, bibliography issues, similarity risks (group 5).

Within problem groups (0-4), lower confidence sorts first (higher risk reviewed first).

## 17. Similarity engine

The similarity engine compares document sentences against a reference corpus using multiple
signal types:

### Match classification

- **EXACT**: exact overlap >= `exact_threshold` (default 0.7).
- **NEAR_DUPLICATE**: exact overlap >= `near_duplicate_threshold` (default 0.5) but below
  exact threshold.
- **LEXICAL_OVERLAP**: lexical similarity >= `lexical_overlap_threshold` (default 0.3) but
  below near-duplicate.
- **SEMANTIC_OVERLAP**: semantic cosine >= `semantic_threshold` (default 0.7) but below
  lexical thresholds.
- **UNMATCHED**: below all thresholds (excluded from results).

### Attribution risk

Risk is citation-aware and semantic-aware:

- Semantic overlap + no citation -> MEDIUM.
- Semantic overlap + matching citation -> LOW.
- High exact overlap + no citation -> HIGH.
- High exact overlap + matching citation + quotation markers -> LOW.
- High exact overlap + matching citation + no quotation -> MEDIUM.
- High exact overlap + mismatched citation -> HIGH.

### Overall similarity

Computed as the percentage of eligible (non-bibliography) characters that are covered by
merged character spans from matched sentences. SEMANTIC_OVERLAP matches are excluded from the
character-span percentage since they have no character-level span evidence.

### Corpus infrastructure

- `corpus/loader.py`: library-level `load_similarity_corpus` entry point.
- Corpus ingestion: file or directory, with optional license metadata.
- Fingerprinting: BLAKE2b + winnowing with configurable shingle size and window.
- TF-IDF indexing for lexical similarity.
- Optional semantic embeddings via `sentence-transformers` (requires `[semantic]` extra).

## 18. LLM integration

### Supported providers

- Anthropic (Messages API)
- OpenAI (Responses API)
- xAI / Grok (Responses API)
- Groq (Chat Completions)
- OpenRouter (Chat Completions)
- NVIDIA NIM (Chat Completions)
- Custom OpenAI-compatible endpoints (Ollama, LM Studio, vLLM, LiteLLM)

### Provider selection

Resolution order:

1. `CITEGUARD_LLM_PROVIDER` environment variable.
2. Auto-detect from available API keys (Anthropic > OpenAI > xAI > Groq > OpenRouter > NVIDIA).

### Resilience

`LLMRouter` wraps backends with:

- **Retry with exponential backoff**: up to 3 attempts on 429, 5xx, and timeout. Respects
  `Retry-After` headers.
- **Circuit breaker**: per-provider, opens after 5 consecutive failures for 60 seconds.
- **Failover chain**: primary provider -> fallbacks from `CITEGUARD_LLM_FALLBACKS`.
- **Health tracking**: per-provider success/failure counts and average latency.

### Task-specific models

Tasks (`claim`, `entailment`) can each have an independent provider and model override via
`CITEGUARD_CLAIM_PROVIDER`, `CITEGUARD_CLAIM_MODEL`, `CITEGUARD_ENTAILMENT_PROVIDER`,
`CITEGUARD_ENTAILMENT_MODEL`.

### LLM caching

Responses are cached in-process with prompt-version-aware invalidation. Cache key includes
provider, model, prompt version, and content hash. Stale entries are automatically invalidated
when prompt versions change.

## 19. Python API

The stable public Python API lives in `citeguard.api`:

```python
from citeguard.api import (
    AuditOptions,
    AuditResult,
    audit_document,
    suggest_document,
    verify_document,
)
```

### AuditOptions

```python
@dataclass(slots=True)
class AuditOptions:
    threshold: int = 60
    max_results: int = 5
    max_claims: int | None = None
    severity: str | None = None
    offline: bool = False
    no_cache: bool = False
    cache_dir: Path | None = None
    recency_max_age: int = 25
    require_evidence: bool = False
    show_similarity: bool = False
    corpus: Path | None = None
    corpus_license: str | None = None
    enable_semantic: bool = False
    bib_providers: Sequence[Any] | None = None
    search_providers: Sequence[Any] | None = None
    on_progress: Callable[[int, int, str], None] | None = None
```

### Functions

- `audit_document(source: Path | ParsedDocument, options: AuditOptions | None) -> AuditResult`
  — full audit pipeline: claims, bibliography verification, cited-claim support, uncited
  suggestions, optional similarity, metrics, review queue.

- `suggest_document(source: Path | ParsedDocument, options: AuditOptions | None) -> AuditResult`
  — claim extraction plus uncited suggestions only (no bibliography verification).

- `verify_document(source: Path | ParsedDocument, options: AuditOptions | None) -> AuditResult`
  — bibliography verification only (no claim extraction).

### AuditResult

```python
@dataclass(slots=True)
class AuditResult:
    document: str
    claims: list[Claim]
    bibliography_verification: list[ReferenceVerification]
    claim_assessments: list[ClaimAssessment]
    suggestions: list[SuggestionAssessment]
    bibliography_issues: list[BibliographyIssue]
    similarity: SimilarityEngineResult | None
    metrics: ProductMetrics
    review_queue: list[dict[str, Any]]
    diagnostics: list[ProviderDiagnostic]
    execution: ExecutionContext
    provider_phase_failed: bool
```

Internal modules remain internal. Only the symbols re-exported in `citeguard.api` are part of
the supported programmatic interface.

## 20. CLI

### Commands

```text
citeguard check FILE       # full audit
citeguard verify FILE      # bibliography verification only
citeguard suggest FILE     # source suggestions only
citeguard inspect FILE     # offline parsing diagnostic
citeguard similarity FILE  # similarity analysis
citeguard init             # create .env template
citeguard llm doctor       # test LLM provider connectivity
citeguard llm list         # show LLM configuration (no secrets)
```

### Common options

```text
--format terminal|json|md|both
--output FILE
--threshold INT
--severity high|medium|low
--max-claims INT
--max-results INT
--no-cache
--verbose
--offline
```

### Command-specific options

`check`:

```text
--show-evidence            # show evidence passages
--require-evidence         # only claims with evidence
--show-similarity          # run similarity engine
--corpus PATH              # similarity corpus
--corpus-license TEXT       # explicit corpus license
--recency-max-age INT      # years (default 25)
```

`verify`:

```text
--recency-max-age INT      # years (default 25)
```

`similarity`:

```text
--corpus PATH
--corpus-license TEXT
--threshold FLOAT          # match threshold (0.0-1.0, default 0.7)
--show-sentences           # sentence-level details
```

### Exit codes

- `0` — success, health score >= 80 (for `check`).
- `1` — findings present, health score < 80 or incomplete.
- `2` — invalid input.
- `3` — provider phase failed online with no usable results.

## 21. Cache

Cache directory:

```text
.citeguard_cache/
```

Cache identity includes:

- cache schema version
- operation
- provider
- query/input identity
- model (for LLM cache)
- prompt version (for LLM cache)

This prevents stale outputs from surviving prompt or model changes unnoticed.

## 22. Error handling

- HTTP 429 and transient 5xx responses: exponential backoff, maximum three total attempts.
- Provider network failure: warn and continue with the next provider.
- LLM circuit breaker: opens after 5 consecutive failures, recovers after 60 seconds.
- Invalid structured LLM output: skip that paragraph or candidate, warn, and continue.
- Unsupported document type: fail fast with a clear CLI error.
- API keys must never appear in logs, reports, or cached request headers.

## 23. Reporting

### Terminal

Rich summary table with concise claim text, status, verdict, severity, and confidence.
Priority review list with severity-colored entries. Evidence passages shown with
`--show-evidence` or `--verbose`.

### Markdown

Detailed report including candidate metadata, source links, scores, warnings, evidence
passages, and citation formatting where metadata permits.

### JSON

Stable structured representation suitable for editor and CI integrations. Ratio fields use
`None` (not 100%) when evaluation was impossible.

The report begins with a Citation Health summary and a priority review list.

## 24. Testing

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
- similarity engine (fingerprinting, TF-IDF, risk classification)
- corpus ingestion and deduplication

Mocked network tests:

- providers
- retrieval fallback
- verification
- full check/suggest/audit pipeline with evidence integration
- LLM backend auto-detection, resolution, and text extraction for all providers
- LLM router retry, circuit breaker, and failover
- similarity engine end-to-end

LLM calls must be mocked in the default test suite. Test runs must not spend API credits or
depend on network availability.

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
- corpus ingestion (synthetic documents)

## 25. Privacy and security

citeguard has no telemetry.

Users must be told which text may leave the local machine. API keys must never be embedded in
reports or cached request headers.

When an LLM provider is configured, paragraph text may be sent to the configured provider for
claim extraction and entailment classification. The full document and bibliography are not
included in prompts.

Academic provider queries send only bibliographic metadata (title, author, year, DOI) — never
document prose.

The `privacy_notice` function in `audit.py` produces an accurate, execution-aware privacy
statement describing exactly what data was sent where during a run.

## 26. Benchmark and calibration

The similarity engine's thresholds and risk classification are calibrated against synthetic
corpora and committed example documents. Calibration changes should be accompanied by
regression tests that verify expected risk levels for known overlap patterns.

The audit health score weights are calibrated to produce meaningful differentiation between
well-cited and poorly-cited documents. Changes to the scoring formula must update the
corresponding tests.

## 27. Open-source project policy

- License: MIT.
- Semantic versioning.
- Public test workflow on supported Python versions (3.10-3.12).
- Bug, feature, parser, and provider issue templates.
- `CONTRIBUTING.md`, `SECURITY.md`, and changelog maintained from the first public release.

## 28. Release definition for v1.0.0

v1.0.0 is releasable when:

- all four target commands (`check`, `verify`, `suggest`, `similarity`) execute end-to-end;
- all four academic providers are functional and fail gracefully;
- claim extraction and matcher outputs are structured;
- result deduplication is implemented;
- audit and priority scoring are deterministic;
- terminal, Markdown, and JSON reports work;
- the Python API (`citeguard.api`) is stable and documented;
- similarity engine produces reproducible results against committed example corpora;
- no default test performs a live network or paid LLM request;
- README privacy and limitation language matches actual behavior;
- example output is reproducible from a committed example document.
