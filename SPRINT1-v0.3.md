# v0.3.0 Implementation Plan — Citeguard Academic Integrity Platform Foundation

## Executive Summary

Bu sprint mevcut Citation Engine'i (v0.2) **dokunmadan** yanına Similarity Engine'i (exact + lexical), sentence-level document modeli, ve CTAC corpus ingest v1'ini ekler. AI detector, semantic matching, web search, rewrite — hepsi v0.4+.

**Target**: ~1,540 LOC yeni kod, ~109 yeni test, mevcut 260 testin hepsi yeşil.

## Dependency Updates

```toml
# pyproject.toml - runtime deps'e eklenecek
nltk >= 3.8
scikit-learn >= 1.4
numpy >= 1.24
```

## Architecture Layers (Execution Order)

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 | `citeguard/normalization.py` | Turkish-aware text normalization |
| 2 | `citeguard/models.py` (extend) | `Sentence`, `Paragraph`, `EnrichedDocument` |
| 3 | `citeguard/sentence_splitter.py` | Deterministic Turkish sentence splitting |
| 4 | `citeguard/similarity/__init__.py` + `models.py` | Similarity data models |
| 5 | `citeguard/similarity/fingerprint.py` | BLAKE2 hash, winnowing, segment detection |
| 6 | `citeguard/similarity/lexical.py` | TF-IDF, cosine, Jaccard, char n-grams |
| 7 | `citeguard/extractor.py` (extend) | `parse_enriched_document()` |
| 8 | `citeguard/similarity/engine.py` | Orchestration + attribution risk |
| 9 | `citeguard/corpus/` (6 modules) | CTAC v1: models, ingest, licenses, normalize, dedup, synthetic |
| 10 | `citeguard/cli.py` (extend) | `similarity` command, `--sentences`, `--show-similarity` |
| 11 | `citeguard/report.py` (extend) | Schema v2, similarity output |

---

## Layer 1: Text Normalization (`citeguard/normalization.py`)

```python
def normalize_turkish(text: str) -> str:
    """Turkish-aware lowercase + whitespace normalize.
    Critical: İ→i, I→ı. Does NOT destroy ç,ğ,ı,İ,ö,ş,ü."""
```

**Test cases**:
- `İSTANBUL` → `istanbul`
- `IŞIK` → `ışık`
- `ÖĞRENCİ` → `öğrenci`
- `MİLLİ` → `milli`
- `Işık` → `ışık`
- `çalışma` → `çalışma`
- Whitespace collapse

## Layer 2: Document Model (`citeguard/models.py`)

**Additive only — `ParsedDocument` korunur.**

```python
@dataclass(slots=True)
class Sentence:
    text: str
    normalized_text: str
    paragraph_index: int
    sentence_index: int
    start_offset: int
    end_offset: int
    citations: list[ExistingCitation] = field(default_factory=list)

@dataclass(slots=True)
class Paragraph:
    text: str
    index: int
    sentences: list[Sentence] = field(default_factory=list)
    is_bibliography: bool = False

@dataclass(slots=True)
class EnrichedDocument:
    path: str
    paragraphs: list[Paragraph]
    sentences: list[Sentence]
    citations: list[ExistingCitation]
    bibliography_entries: list[BibliographyEntry]
    bibliography_start_index: int | None
```

## Layer 3: Turkish Sentence Splitter (`citeguard/sentence_splitter.py`)

**Protected abbreviations**:
`Dr.`, `Prof.`, `Doç.`, `Müh.`, `Yrd.`, `Tek.`, `Mim.`, `ibid.`, `bkz.`, `vb.`, `vd.`, `vs.`, `örn.`, `T.C.`

**Protected patterns**:
- `Prof. Dr.` combinations
- Decimal: `3.5`, `1,5`
- DOI: `10.1000/xyz.123`
- URLs
- Numbered headings: `1. Giriş`
- Ellipsis: `...`

**Output**: `list[Sentence]` with accurate `start_offset`/`end_offset` in paragraph.

**Test matrix** (16 test):
| Case | Expected |
|------|----------|
| `Prof. Dr. Mehmet Yılmaz çalışmayı yürüttü.` | 1 sentence |
| `Oran %3.5 olarak ölçüldü.` | 1 sentence |
| `Bkz. Şekil 2.` | 1 sentence |
| `DOI: 10.1000/xyz.123` | 1 sentence |
| Multi-sentence paragraph | Correct split |
| Offset accuracy | Per-sentence verified |

## Layer 4: Similarity Models (`citeguard/similarity/models.py`)

```python
@dataclass(frozen=True, slots=True)
class Shingle:
    hash: int
    start: int
    end: int
    position: int

@dataclass(frozen=True, slots=True)
class FingerprintPoint:
    hash: int
    position: int
    start: int
    end: int

@dataclass(slots=True)
class Fingerprint:
    points: list[FingerprintPoint]
    doc_id: str | None = None

class MatchType(str, Enum):
    EXACT = "exact"
    NEAR_DUPLICATE = "near_duplicate"
    LEXICAL_OVERLAP = "lexical_overlap"
    UNMATCHED = "unmatched"

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

@dataclass(slots=True)
class SimilarityMatch:
    source_text: str
    source_title: str | None
    source_id: str | None
    source_url: str | None = None
    exact_overlap: float
    lexical_similarity: float
    combined_score: float
    matched_source_spans: list[tuple[int, int]] = field(default_factory=list)
    matched_document_spans: list[tuple[int, int]] = field(default_factory=list)
    match_type: MatchType = MatchType.UNMATCHED

@dataclass(slots=True)
class SimilarityResult:
    sentence: "Sentence"
    matches: list[SimilarityMatch] = field(default_factory=list)
    best_match: SimilarityMatch | None = None
    attribution_risk: RiskLevel = RiskLevel.NONE
    attribution_reason: str = ""

@dataclass(slots=True)
class SimilarityEngineResult:
    results: list[SimilarityResult]
    overall_similarity_pct: float
    high_risk_count: int
    medium_risk_count: int
    total_sentences: int
    matched_sentences: int
    unique_matched_chars: int
    eligible_chars: int

@dataclass(frozen=True, slots=True)
class SimilarityConfig:
    shingle_size: int = 5
    winnow_window: int = 4
    exact_threshold: float = 0.95
    near_duplicate_threshold: float = 0.70
    lexical_overlap_threshold: float = 0.40
    max_results_per_sentence: int = 5
    min_match_length: int = 20
    combined_weight_exact: float = 0.6
    combined_weight_lexical: float = 0.4

@dataclass(frozen=True, slots=True)
class SimilarityMetrics:
    overall_pct: float
    high_risk_count: int
    medium_risk_count: int
    matched_sentences: int
    total_sentences: int
    avg_overlap: float
    unique_matched_chars: int
    eligible_chars: int
```

## Layer 5: Fingerprinting (`citeguard/similarity/fingerprint.py`)

```python
def stable_hash(text: str) -> int:
    """Deterministic 64-bit via BLAKE2b. Python hash() KULLANMA."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")

def generate_shingles(text: str, k: int = 5) -> list[Shingle]:
    """Ordered k-grams with positions. Returns list[Shingle], NOT set."""

def winnow(shingles: list[Shingle], window: int = 4) -> list[FingerprintPoint]:
    """Sliding window minimum hash selection. Preserves positions."""

def exact_overlap(fp1: Fingerprint, fp2: Fingerprint) -> float:
    """|hash intersection| / |hash union|"""

def find_matching_segments(fp1, fp2, shingles1, shingles2, k=5) -> list[tuple[int,int]]:
    """Contiguous character spans from matching fingerprint points."""
```

## Layer 6: Lexical Similarity (`citeguard/similarity/lexical.py`)

```python
def build_tfidf_index(documents: list[str]) -> tuple:
    vectorizer = TfidfVectorizer(analyzer=turkish_analyzer)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix

def compute_cosine_similarity(query, vectorizer, matrix) -> list[tuple[int, float]]:
    return sorted(enumerate(scores), key=lambda x: -x[1])

def char_ngram_jaccard(text1, text2, n=3) -> float:
    return len(set(ngrams1) & set(ngrams2)) / len(set(ngrams1) | set(ngrams2))

def turkish_analyzer(text):
    return normalize_turkish(text).split()

def classify_match_type(exact, lexical, config) -> MatchType: ...
```

**Thresholds** (configurable via `SimilarityConfig`):
- EXACT: `exact_overlap >= 0.95`
- NEAR_DUPLICATE: `exact_overlap >= 0.70`
- LEXICAL_OVERLAP: `lexical_similarity >= 0.40`
- UNMATCHED: else

## Layer 7: Enriched Parse (`citeguard/extractor.py`)

```python
def parse_enriched_document(path: str | Path) -> EnrichedDocument:
    doc = parse_document(path)
    for pidx, para_text in enumerate(doc.paragraphs):
        is_bib = doc.bibliography_start_index is not None and pidx >= doc.bibliography_start_index
        sentences = split_sentences(para_text)
        for sidx, sent in enumerate(sentences):
            sent.paragraph_index = pidx
            sent.sentence_index = sidx
            sent.citations = [
                c for c in doc.citations 
                if c.paragraph_index == pidx and sent.start_offset <= c.char_offset < sent.end_offset
            ]
            all_sentences.append(sent)
        enriched_paragraphs.append(Paragraph(...))
    return EnrichedDocument(...)
```

## Layer 8: Similarity Engine (`citeguard/similarity/engine.py`)

```python
class SimilarityEngine:
    def analyze_document(self, doc: EnrichedDocument, corpus) -> SimilarityEngineResult: ...
    def compare_sentence_to_corpus(self, sentence, corpus_fps) -> list[SimilarityMatch]: ...
    def compute_attribution_risk(
        self, match, sentence_citations, bibliography_entries
    ) -> tuple[RiskLevel, str]: ...
    def compute_overall_similarity(self, results, doc) -> float: ...
```

**Attribution Risk Logic**:
```
match exists?
├── NO → NONE
└── YES
    ├── citation present?
    │   ├── NO → HIGH
    │   └── YES
    │       ├── citation matches detected source?
    │       │   ├── YES → LOW
    │       │   └── NO → HIGH (wrong source)
    │       └── quotation markers? → MEDIUM
    └── severity adjustment:
        exact >= 0.95 → +1
        exact < 0.70 → -1
```

**Overall Similarity** (character-span union):
```python
def compute_overall_similarity(self, results, doc):
    eligible_chars = sum(len(s.text) for s in doc.sentences if not s.is_bibliography)
    matched_spans = [span for result in results for match in result.matches for span in match.matched_document_spans]
    merged = merge_intervals(matched_spans)
    unique_matched = sum(end - start for start, end in merged)
    return (unique_matched / eligible_chars * 100) if eligible_chars > 0 else 0.0
```

## Layer 9-13: Corpus Infrastructure

### `corpus/models.py`
### `corpus/licenses.py`
### `corpus/ingest.py`
### `corpus/normalize.py`
### `corpus/deduplicate.py`
### `corpus/synthetic.py` (test fixtures only)

## Layer 10: CLI Commands

```bash
citeguard similarity FILE [--corpus PATH] [--threshold 0.7] [--format json|md|both] [--output PATH]
citeguard inspect FILE --sentences
citeguard check FILE --show-similarity [--corpus PATH]
```

## Layer 11: Report Integration

- `SCHEMA_VERSION = "2"`
- Similarity output JSON + markdown
- Citation Health + Similarity ayrı gösterilir

## Implementation Sıralamı

1. normalization.py + test → pytest + ruff
2. models.py extend + test → pytest + ruff
3. sentence_splitter.py + test → pytest + ruff
4. similarity/models.py + test → pytest + ruff
5. similarity/fingerprint.py + test → pytest + ruff
6. similarity/lexical.py + test → pytest + ruff
7. extractor.py extend + test → pytest + ruff
8. similarity/engine.py + test → pytest + ruff
9. corpus/ (all modules + tests) → pytest + ruff
10. CLI integration + test → pytest + ruff
11. Report integration + test → pytest + ruff
12. pyproject.toml update + build + full pytest

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| BLAKE2b hash | Python hash() randomized per-process |
| Ordered shingles + positions | set loses position → no highlight |
| LEXICAL_OVERLAP not PARAPHRASE | TF-IDF can't prove paraphrase |
| Character-span union for % | sentence-count % unfair |
| Citation Health ≠ Similarity | Composite later (v0.8) |
| CTAC primary = real corpus | synthetic only for tests |
| Unknown license → conservative | Must opt-in |

## What v0.3 Does NOT Include

- AI-writing detector (v0.5)
- sentence-transformers / embeddings (v0.4)
- Paraphrase detection (v0.4)
- Web search (v0.4)
- Automatic rewriting (v0.6)
- Composite Integrity Score (v0.8+)
- Next.js/FastAPI UI (v0.8+)

## Backward Compatibility Guarantees

- `ParsedDocument` + `parse_document()` → unchanged
- `Citation Health Score` → unchanged
- LLM infrastructure → dokunulmaz
- `matcher.py`, `entailment.py`, `verification.py` → unchanged
- All 260 existing tests → pass without modification

## File Structure After v0.3

citeguard/
├── __init__.py
├── __main__.py
├── bibliography.py
├── cache.py
├── claims.py
├── cli.py              ← extended
├── config.py
├── corpus/             ← NEW
│   ├── __init__.py
│   ├── models.py
│   ├── ingest.py
│   ├── licenses.py
│   ├── normalize.py
│   ├── deduplicate.py
│   └── synthetic.py
├── entailment.py
├── evidence.py
├── extractor.py        ← extended
├── fingerprint.py      ← NEW
├── lexical.py          ← NEW
├── linking.py
├── matcher.py
├── models.py           ← extended
├── normalization.py    ← NEW
├── report.py           ← extended
├── retrieval.py
├── scoring.py
├── sentence_splitter.py← NEW
├── similarity/         ← NEW
│   ├── __init__.py
│   ├── models.py
│   ├── fingerprint.py
│   ├── lexical.py
│   └── engine.py
├── verification.py
├── llm.py
├── llm_backends.py
├── llm_router.py
└── llm_cache.py

## Final Validation

1. `ruff check .`
2. `pytest -q`
3. `python -m build`
4. Tüm existing citation/LLM/tests pass edilmeli
5. 360+ test expected (260 existing + ~109 new)

Kodlamaya başlayalım. İlk katmandan (normalization + models extend) başlayıp, test + lint koşup sonraki katmana geçeceğiz.