"""Similarity engine orchestration for Citeguard v0.5.

Responsibilities:
- Compare sentences to corpus fingerprints
- Classify match types (exact, near-duplicate, lexical, semantic)
- Compute attribution risk (citation-aware, semantic-aware)
- Aggregate overall similarity metrics
- Merge character-spans for percentage calculation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from citeguard.models import TextSpan
from citeguard.similarity.fingerprint import (
    exact_overlap,
    find_matching_segments_pair,
    generate_shingles,
    winnow,
)
from citeguard.similarity.lexical import (
    build_tfidf_index,
    char_ngram_jaccard,
    classify_match_type,
    compute_cosine_similarity,
)
from citeguard.similarity.models import (
    CorpusEntryTuple,
    Fingerprint,
    MatchType,
    RiskLevel,
    SimilarityConfig,
    SimilarityEngineResult,
    SimilarityMatch,
    SimilarityResult,
)
from citeguard.similarity.normalize import build_offset_map, remap_span
from citeguard.similarity.reranker import compute_ranking_score

if TYPE_CHECKING:
    from citeguard.models import (
        BibliographyEntry,
        ExistingCitation,
        Sentence,
    )
    from citeguard.similarity.index import SimilarityIndex

# ---------------------------------------------------------------------------
# Helper: merge overlapping/adjacent spans (paragraph-relative)
# ---------------------------------------------------------------------------


def _merge_spans(spans: list[TextSpan]) -> list[TextSpan]:
    """Merge overlapping or touching TextSpan objects.

    Spans are paragraph-relative: ``(paragraph_index, start, end)``.
    Adjacent spans (cur_start <= last_end) are merged.
    """
    if not spans:
        return []

    # Sort by paragraph then start offset
    sorted_spans = sorted(spans, key=lambda s: (s.paragraph_index, s.start))

    merged: list[TextSpan] = [sorted_spans[0]]
    for current in sorted_spans[1:]:
        last = merged[-1]
        # Same paragraph and overlapping/touching
        if (
            current.paragraph_index == last.paragraph_index
            and current.start <= last.end
        ):
            merged[-1] = TextSpan(
                paragraph_index=last.paragraph_index,
                start=last.start,
                end=max(last.end, current.end),
            )
        else:
            merged.append(current)
    return merged


def _tokenize_author(value: str | None) -> set[str]:
    """Extract coarse author tokens for metadata matching."""
    if not value:
        return set()
    cleaned = value.lower().replace("&", " ").replace(",", " ")
    return {token for token in cleaned.split() if len(token) > 2 and token != "et"}


def _bibliography_match_for_citation(
    citation: ExistingCitation, bibliography_entries: list[BibliographyEntry]
) -> BibliographyEntry | None:
    """Return the bibliography entry linked to a citation, if obvious."""
    for entry in bibliography_entries:
        if citation.doi and entry.doi and citation.doi.lower() == entry.doi.lower():
            return entry
        if citation.numbered_ref is not None and citation.numbered_ref == entry.numbered_ref:
            return entry
        citation_tokens = _tokenize_author(citation.authors)
        entry_tokens = _tokenize_author(entry.authors)
        if citation.year and entry.year == citation.year and citation_tokens & entry_tokens:
            return entry
    return None


def _citation_matches_source_metadata(
    citation: ExistingCitation,
    *,
    source_authors: list[str],
    source_year: int | None,
    source_doi: str,
    bibliography_entries: list[BibliographyEntry],
) -> bool:
    """Match citation metadata against corpus source metadata."""
    if citation.doi and source_doi and citation.doi.lower() == source_doi:
        return True

    candidate_authors = citation.authors
    candidate_year = citation.year
    bib_entry = _bibliography_match_for_citation(citation, bibliography_entries)
    if bib_entry is not None:
        if bib_entry.doi and source_doi and bib_entry.doi.lower() == source_doi:
            return True
        candidate_authors = bib_entry.authors or candidate_authors
        candidate_year = bib_entry.year or candidate_year

    if source_year is not None and candidate_year != source_year:
        return False

    citation_tokens = _tokenize_author(candidate_authors)
    source_tokens: set[str] = set()
    for author in source_authors:
        source_tokens.update(_tokenize_author(author))
    return bool(citation_tokens and source_tokens and citation_tokens & source_tokens)


def _has_quotation_markers(text: str) -> bool:
    """Return True when the sentence contains quotation punctuation."""
    return any(marker in text for marker in ('"', "\u201c", "\u201d", "\u00ab", "\u00bb"))


def _find_quote_pairs(text: str) -> list[tuple[int, int]]:
    """Find all quotation mark pairs in text, returning (start, end) indices.

    Supports standard quotes ("), curly quotes ("").and guillemets (<<>>).
    Returns list of (start, end) tuples where start is the opening quote index
    and end is the closing quote index (exclusive).
    """
    quote_pairs: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []

    matching = {'"': '"', "\u201c": "\u201d", "\u00ab": "\u00bb"}
    curly_opening = {'\u201c', '\u00ab'}
    curly_closing = {'\u201d', '\u00bb'}

    for i, char in enumerate(text):
        if char in curly_opening:
            stack.append((char, i))
        elif char in curly_closing:
            if stack and matching.get(stack[-1][0]) == char:
                _open_char, open_pos = stack.pop()
                quote_pairs.append((open_pos, i + 1))
        elif char == '"':
            if stack and stack[-1][0] == '"':
                _open_char, open_pos = stack.pop()
                quote_pairs.append((open_pos, i + 1))
            else:
                stack.append((char, i))

    return quote_pairs


def _span_within_quotes(text: str, span_start: int, span_end: int) -> bool:
    """Check if a span is fully contained within quotation marks."""
    quote_pairs = _find_quote_pairs(text)
    return any(
        q_start <= span_start and span_end <= q_end
        for q_start, q_end in quote_pairs
    )


def _compute_quote_coverage(
    sentence_text: str,
    matched_spans: list[TextSpan],
    sentence_start_offset: int,
) -> float:
    """Compute the fraction of matched chars that lie inside quotation marks."""
    if not matched_spans:
        return 0.0

    quote_pairs = _find_quote_pairs(sentence_text)
    if not quote_pairs:
        return 0.0

    sorted_quotes = sorted(quote_pairs, key=lambda q: (q[0], q[1]))
    merged_quotes: list[tuple[int, int]] = [sorted_quotes[0]]
    for q_start, q_end in sorted_quotes[1:]:
        last_start, last_end = merged_quotes[-1]
        if q_start < last_end:
            merged_quotes[-1] = (last_start, max(last_end, q_end))
        else:
            merged_quotes.append((q_start, q_end))

    merged = _merge_spans(matched_spans)

    total_chars = 0
    quoted_chars = 0
    for span in merged:
        sent_start = span.start - sentence_start_offset
        sent_end = span.end - sentence_start_offset
        span_len = sent_end - sent_start
        if span_len <= 0:
            continue
        total_chars += span_len

        for q_start, q_end in merged_quotes:
            inter_start = max(sent_start, q_start)
            inter_end = min(sent_end, q_end)
            if inter_start < inter_end:
                quoted_chars += inter_end - inter_start

    if total_chars == 0:
        return 0.0
    return quoted_chars / total_chars


# ---------------------------------------------------------------------------
# SimilarityEngine class
# ---------------------------------------------------------------------------


class SimilarityEngine:
    """Orchestrate sentence-against-corpus similarity analysis."""

    def __init__(self, config: SimilarityConfig | None = None):
        self.config = config or SimilarityConfig()

    # ------------------------------------------------------------------
    # Corpus indexing (offline, once per document or batch)
    # ------------------------------------------------------------------

    def build_fingerprint(self, text: str) -> Fingerprint:
        """Create a BLAKE2b fingerprint for *text*."""
        shingles = generate_shingles(text, k=self.config.shingle_size)
        points = winnow(shingles, window=self.config.winnow_window)
        return Fingerprint(points=points)

    def build_tfidf_index(self, documents: list[str]) -> tuple:
        """Fit TF-IDF over *documents* and return (vectorizer, matrix)."""
        return build_tfidf_index(documents)

    # ------------------------------------------------------------------
    # Sentence-against-corpus comparison
    # ------------------------------------------------------------------

    def compare_sentence_to_corpus(
        self,
        sentence: Sentence,
        corpus_entries: list[CorpusEntryTuple] | None = None,
        tfidf_info: tuple | None = None,
        index: SimilarityIndex | None = None,
        candidate_indices: list[int] | None = None,
        semantic_scores: dict[int, float] | None = None,
    ) -> list[SimilarityMatch]:
        """Return similarity matches for one sentence against a corpus.

        Parameters
        ----------
        sentence:
            The sentence to match.
        corpus_entries:
            List of (doc_id, text, Fingerprint, metadata, entry_index) tuples.
            Ignored when *index* is provided.
        tfidf_info:
            Optional (vectorizer, matrix) from corpus documents.
            Ignored when *index* is provided.
        index:
            Pre-built ``SimilarityIndex``.
        candidate_indices:
            When provided, only evaluate entries at these indices.
            When *None*, evaluate all entries (backward-compatible full scan).
        semantic_scores:
            Pre-computed semantic cosine per entry_index.  When provided
            the engine reuses these values instead of recomputing.
        """
        # -- Resolve data source ------------------------------------------------
        if index is not None:
            resolved_entries: list[CorpusEntryTuple] = index.entries
            resolved_tfidf = (
                (index.tfidf_vectorizer, index.tfidf_matrix)
                if index.tfidf_vectorizer is not None
                else None
            )
        elif corpus_entries is not None:
            resolved_entries = corpus_entries
            resolved_tfidf = tfidf_info
        else:
            return []

        semantic_enabled = (
            semantic_scores is not None
            or (index is not None and index.has_embeddings)
        )

        matches: list[SimilarityMatch] = []

        # 1) Exact/fingerprint overlap
        sentence_fp = self.build_fingerprint(sentence.normalized_text)
        sentence_offset_map = build_offset_map(
            sentence.text, sentence.normalized_text,
        )
        tfidf_scores: dict[int, float] = {}
        if resolved_tfidf is not None:
            vectorizer, matrix = resolved_tfidf
            tfidf_scores = dict(
                compute_cosine_similarity(sentence.normalized_text, vectorizer, matrix)
            )

        # Determine which entries to evaluate
        if candidate_indices is not None:
            entry_iter = (
                (i, resolved_entries[i])
                for i in candidate_indices
                if i < len(resolved_entries)
            )
        else:
            entry_iter = enumerate(resolved_entries)

        for entry_index, entry in entry_iter:
            (
                doc_id, doc_text, doc_fp, metadata,
                source_entry_index, corpus_entry_obj,
            ) = entry
            eo = exact_overlap(sentence_fp, doc_fp)

            # 2) Lexical similarity
            if resolved_tfidf is not None:
                ls = tfidf_scores.get(entry_index, 0.0)
            else:
                ls = char_ngram_jaccard(sentence.normalized_text, doc_text)

            # 3) Semantic score (reuse from CandidateSet, don't recompute)
            sem_raw = 0.0
            if semantic_scores and entry_index in semantic_scores:
                sem_raw = semantic_scores[entry_index]

            # 4) Classify match type
            mt = classify_match_type(
                exact_overlap=eo,
                lexical_similarity=ls,
                exact_threshold=self.config.exact_threshold,
                near_duplicate_threshold=self.config.near_duplicate_threshold,
                lexical_overlap_threshold=self.config.lexical_overlap_threshold,
                semantic_score=sem_raw if semantic_enabled else None,
                semantic_threshold=self.config.semantic_threshold,
            )

            # 5) Ranking score (hybrid)
            semantic_component, ranking_score = compute_ranking_score(
                eo, ls, sem_raw, self.config,
            )

            # 6) Combined score (legacy, kept for backward compat)
            combined = (
                self.config.combined_weight_exact * eo
                + self.config.combined_weight_lexical * ls
            )

            # 7) SEMANTIC_OVERLAP invariant: no fake spans from cosine similarity
            if mt == MatchType.SEMANTIC_OVERLAP:
                matched_doc_spans = []
                matched_src_spans = []
            else:
                # Find matching spans using fingerprint
                segments = find_matching_segments_pair(
                    sentence_fp, doc_fp, k=self.config.shingle_size
                )
                # Remap document spans
                matched_doc_spans = [
                    TextSpan(
                        paragraph_index=sentence.paragraph_index,
                        start=sentence.start_offset + remap_span(
                            start, end, sentence_offset_map
                        )[0],
                        end=sentence.start_offset + remap_span(
                            start, end, sentence_offset_map
                        )[1],
                    )
                    for start, end in segments.document_spans
                ]
                # Remap source spans: use passage-level offset_map
                corpus_offset_map = None
                if (
                    corpus_entry_obj
                    and hasattr(corpus_entry_obj, "offset_map")
                    and corpus_entry_obj.offset_map
                ):
                    corpus_offset_map = corpus_entry_obj.offset_map
                matched_src_spans = [
                    TextSpan(
                        paragraph_index=source_entry_index,
                        start=(
                            remap_span(start, end, corpus_offset_map)[0]
                            if corpus_offset_map else start
                        ),
                        end=(
                            remap_span(start, end, corpus_offset_map)[1]
                            if corpus_offset_map else end
                        ),
                    )
                    for start, end in segments.source_spans
                ]

            authors = metadata.authors if metadata is not None else []
            year = metadata.year if metadata is not None else None
            doi = metadata.doi if metadata is not None else None
            title = metadata.title if metadata is not None else doc_id
            url = metadata.url if metadata is not None else None

            # 8) Build match
            match = SimilarityMatch(
                source_text=(
                    corpus_entry_obj.text
                    if corpus_entry_obj and hasattr(corpus_entry_obj, "text")
                    else doc_text
                ),
                source_title=title,
                source_id=doc_id,
                exact_overlap=eo,
                lexical_similarity=ls,
                combined_score=combined,
                source_url=url,
                source_authors=authors,
                source_year=year,
                source_doi=doi,
                match_type=mt,
                matched_document_spans=matched_doc_spans,
                matched_source_spans=matched_src_spans,
                semantic_similarity_raw=sem_raw,
                semantic_rerank_score=semantic_component,
                ranking_score=ranking_score,
            )
            matches.append(match)

        # Filter out UNMATCHED, sort by ranking_score descending, then
        # cap results per sentence when max_results_per_sentence > 0.
        matches = [m for m in matches if m.match_type != MatchType.UNMATCHED]
        matches.sort(key=lambda m: -m.ranking_score)
        if self.config.max_results_per_sentence > 0:
            matches = matches[:self.config.max_results_per_sentence]
        return matches

    # ------------------------------------------------------------------
    # Attribution risk (citation-aware, semantic-aware)
    # ------------------------------------------------------------------

    def compute_attribution_risk(
        self,
        match: SimilarityMatch,
        sentence_citations: list[ExistingCitation],
        bibliography_entries: list[BibliographyEntry],
        sentence_text: str = "",
        sentence_start_offset: int = 0,
    ) -> tuple[RiskLevel, str]:
        """Determine attribution risk for a match given the sentence's citations.

        Risk logic:
        - semantic overlap + no citation → MEDIUM
        - semantic overlap + matching citation → LOW
        - semantic overlap + mismatched citation → MEDIUM
        - semantic overlap never → HIGH
        - high overlap + no citation → HIGH
        - high overlap + wrong citation → HIGH
        - high overlap + matching citation + no quotation → MEDIUM
        - high overlap + matching citation + quotation markers → LOW
        """
        if not match:
            return RiskLevel.NONE, "no match found"

        source_authors = [author.lower() for author in match.source_authors]
        source_year = match.source_year
        source_doi = (match.source_doi or "").lower()

        # SEMANTIC_OVERLAP-specific branch: risk capped at MEDIUM
        if match.match_type == MatchType.SEMANTIC_OVERLAP:
            if not sentence_citations:
                return RiskLevel.MEDIUM, "semantic overlap without citation"
            for cit in sentence_citations:
                citation_matches = _citation_matches_source_metadata(
                    cit,
                    source_authors=source_authors,
                    source_year=source_year,
                    source_doi=source_doi,
                    bibliography_entries=bibliography_entries,
                )
                if citation_matches:
                    return RiskLevel.LOW, "semantic overlap with matching citation"
            return RiskLevel.MEDIUM, "semantic overlap without matching citation"

        # Standard path (exact/lexical/near-duplicate)
        if not sentence_citations:
            if match.exact_overlap >= 0.95:
                return RiskLevel.HIGH, "high exact overlap without citation"
            return RiskLevel.MEDIUM, "similar text without citation"

        # Sentence has citations — check if ANY citation matches
        at_least_one_matches = False
        citation_issues: list[str] = []

        for cit in sentence_citations:
            citation_matches_source = _citation_matches_source_metadata(
                cit,
                source_authors=source_authors,
                source_year=source_year,
                source_doi=source_doi,
                bibliography_entries=bibliography_entries,
            )

            if citation_matches_source:
                at_least_one_matches = True
            else:
                citation_issues.append(
                    f"citation from {cit.authors or 'unknown'} "
                    f"({cit.year or 'n.d.'}) "
                    f"does not match detected source"
                )

        if not at_least_one_matches:
            return RiskLevel.HIGH, "; ".join(citation_issues)

        # At least one citation matches
        if match.exact_overlap >= 0.95:
            quoted = False
            if match.matched_document_spans:
                coverage = _compute_quote_coverage(
                    sentence_text,
                    match.matched_document_spans,
                    sentence_start_offset,
                )
                quoted = (
                    coverage >= self.config.quotation_coverage_threshold
                )
            else:
                quoted = _has_quotation_markers(sentence_text)

            if quoted:
                return RiskLevel.LOW, (
                    "quoted high overlap with matching citation"
                )
            return RiskLevel.MEDIUM, (
                "high exact overlap with matching citation"
            )
        if match.exact_overlap < 0.70:
            return RiskLevel.MEDIUM, "citations match but low exact overlap"
        return RiskLevel.LOW, "citations match detected source"

    # ------------------------------------------------------------------
    # Overall similarity aggregation
    # ------------------------------------------------------------------

    def compute_overall_similarity(
        self,
        results: list[SimilarityResult],
        doc_sentences: list[Sentence],
    ) -> float:
        """Compute overall document similarity as character-span percentage.

        SEMANTIC_OVERLAP matches are excluded because they have no
        character-level span evidence.
        """
        eligible_chars = sum(
            s.length for s in doc_sentences if not s.is_bibliography
        )

        if eligible_chars == 0:
            return 0.0

        all_spans: list[TextSpan] = []
        for result in results:
            for match in result.matches:
                for span in match.matched_document_spans:
                    all_spans.append(
                        TextSpan(
                            paragraph_index=result.sentence.paragraph_index,
                            start=span.start,
                            end=span.end,
                        )
                    )

        merged = _merge_spans(all_spans)
        unique_matched = sum(span.length() for span in merged)

        return (unique_matched / eligible_chars) * 100.0

    def analyze_document(
        self,
        sentences: list[Sentence],
        corpus_entries: list[CorpusEntryTuple] | None = None,
        bibliography_entries: list[BibliographyEntry] | None = None,
        index: SimilarityIndex | None = None,
        embedding_backend: Any | None = None,
    ) -> SimilarityEngineResult:
        """Run full similarity analysis over a document's sentences.

        When *index* has embeddings and *embedding_backend* is provided,
        the engine computes semantic scores, generates RRF candidates,
        and passes them to ``compare_sentence_to_corpus()``.  Otherwise
        the full-scan lexical-only path is used (backward-compatible).
        """
        bibliography_entries = bibliography_entries or []

        # Build index if not supplied (backward-compatible path)
        if index is None and corpus_entries:
            from citeguard.similarity.index import SimilarityIndex

            index = SimilarityIndex.build(corpus_entries)

        # Determine semantic capability
        semantic_enabled = (
            index is not None
            and index.has_embeddings
            and self.config.enable_semantic
            and embedding_backend is not None
        )

        results: list[SimilarityResult] = []
        high_risk = 0
        medium_risk = 0

        for sent in sentences:
            if sent.is_bibliography:
                results.append(
                    SimilarityResult(
                        sentence=sent,
                        matches=[],
                        best_match=None,
                        attribution_risk=RiskLevel.NONE,
                        attribution_reason="bibliography sentence excluded",
                    )
                )
                continue

            candidate_indices = None
            semantic_scores = None

            if semantic_enabled and index is not None:
                # Encode query sentence
                try:
                    query_emb = embedding_backend.encode(
                        [sent.normalized_text]
                    )[0]
                except Exception:
                    query_emb = None

                if query_emb is not None:
                    sentence_fp = self.build_fingerprint(sent.normalized_text)
                    try:
                        candidates = index.retrieve_candidates(
                            query_text=sent.normalized_text,
                            query_fp=sentence_fp,
                            query_embedding=query_emb,
                            top_k=self.config.max_candidates,
                        )
                        candidate_indices = candidates.indices
                        semantic_scores = candidates.semantic_scores
                    except ValueError:
                        # Dimension mismatch or near-zero norm — skip semantic
                        pass

            sent_matches = self.compare_sentence_to_corpus(
                sent,
                index=index,
                candidate_indices=candidate_indices,
                semantic_scores=semantic_scores,
            )

            # Compute attribution risk for the best match
            if sent_matches:
                best = sent_matches[0]
                risk, reason = self.compute_attribution_risk(
                    best,
                    sent.citations,
                    bibliography_entries,
                    sentence_text=sent.text,
                    sentence_start_offset=sent.start_offset,
                )
            else:
                best = None
                risk = RiskLevel.NONE
                reason = ""

            result = SimilarityResult(
                sentence=sent,
                matches=sent_matches,
                best_match=best,
                attribution_risk=risk,
                attribution_reason=reason,
            )
            results.append(result)

            if risk == RiskLevel.HIGH:
                high_risk += 1
            elif risk == RiskLevel.MEDIUM:
                medium_risk += 1

        # Overall similarity
        overall_pct = self.compute_overall_similarity(results, sentences)

        total = len(sentences)
        matched = sum(1 for r in results if r.matches)

        eligible = sum(s.length for s in sentences if not s.is_bibliography)
        unique_matched = 0
        all_spans: list[TextSpan] = []
        for r in results:
            for m in r.matches:
                for sp in m.matched_document_spans:
                    all_spans.append(
                        TextSpan(
                            paragraph_index=r.sentence.paragraph_index,
                            start=sp.start,
                            end=sp.end,
                        )
                    )
        if all_spans:
            merged = _merge_spans(all_spans)
            unique_matched = sum(s.length() for s in merged)

        return SimilarityEngineResult(
            results=results,
            overall_similarity_pct=overall_pct,
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            total_sentences=total,
            matched_sentences=matched,
            unique_matched_chars=unique_matched,
            eligible_chars=eligible,
        )
