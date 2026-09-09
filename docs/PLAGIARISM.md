# Plagiarism review methodology

CiteGuard provides source-aware document similarity and plagiarism-review signals over the text
available to it. It does not issue a definitive plagiarism verdict, reproduce Turnitin's
proprietary corpus, or predict a commercial score.

## Pipeline

```text
Document → distinct textual sources → candidate retrieval → exact/lexical/semantic evidence
         → quote/citation/bibliography context → unique coverage scores → safe improvement
         → complete revised document → same-corpus rescan
```

Markdown, TXT, and DOCX use the existing document parser. Sentence and paragraph offsets are kept
through normalization. Each corpus file remains a distinct source. URL input is opt-in and accepts
only HTTP(S) textual content after hostname, redirect, content-type, response-size, and timeout
checks. Downloaded content is never executed.

`--discover-sources --online` sends bounded bibliography title/author/year queries through the
existing Semantic Scholar, Crossref, OpenAlex, and arXiv orchestration. A discovered metadata
record is marked separately from textual comparison. Only an abstract actually returned by a
provider can enter similarity analysis; title, author, year, and DOI matches never affect overlap
percentages.

## Scores and classifications

The raw and review-relevant percentages come from actual document word coverage, not an arbitrary
classifier output. Overlapping matches and duplicate sources count each document word once.
Review-relevant coverage excludes common phrases and, unless explicitly included, quotations and
bibliography regions. Exact, lexical, and semantic signals remain separately inspectable.

A citation and textual independence are separate questions. Long verbatim text can therefore be
`cited_but_verbatim` and high severity. Quoted text remains visible, while quoted-and-attributed
text is excluded from the default review score. Semantic similarity alone is conservative review
evidence; topical similarity must not be interpreted as plagiarism.

## Source and corpus limits

Local files and explicit URLs provide the compared text. Academic APIs can resolve metadata and
discover candidate works elsewhere in CiteGuard, but metadata similarity is never counted as
textual overlap. Coverage is necessarily limited to supplied or legitimately retrieved text.

Corpus indexes are content-addressed and include normalized passages, fingerprints, TF-IDF
artifacts, source hashes, algorithm configuration, and schema versions. `--no-cache` disables cache
reads and writes. An explicit reusable index can be built with `citeguard corpus index`.

## Improvement safety

`improve-attribution` defaults to preview and can reuse local sources from a plagiarism JSON report
with `--from-report`. Applied changes always target a new file. Candidate rewrites must preserve
citations, numbers, factual constraints, and meaning; reduce source overlap; pass semantic and
bidirectional entailment checks where available; survive document reconstruction; and improve an
actual same-corpus rescan. Unsafe actions remain unchanged for manual review. No source or citation
is fabricated.

When a remote rewrite provider is configured, CiteGuard sends only a bounded risky passage, its
citation token, matched source metadata, and bounded evidence—not the complete document. Offline
mode forbids academic, URL, LLM, and model-download network activity.
