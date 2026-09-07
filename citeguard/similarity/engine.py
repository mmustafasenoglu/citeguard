"""Similarity engine orchestration for Citeguard v0.3.

Responsibilities:
- Compare sentences to corpus fingerprints
- Classify match types
- Compute attribution risk (citation-aware)
- Aggregate overall similarity metrics
- Merge character-spans for percentage calculation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    Fingerprint,
    MatchType,
    RiskLevel,
    SimilarityConfig,
    SimilarityEngineResult,
    SimilarityMatch,
    SimilarityResult,
)
from citeguard.similarity.normalize import build_offset_map, remap_span

if TYPE_CHECKING:
    from citeguard.corpus.models import CorpusMetadata
    from citeguard.models import (
        BibliographyEntry,
        ExistingCitation,
        Sentence,
    )

CorpusEntryTuple = tuple[str, str, Fingerprint, "CorpusMetadata | None", int, object]

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
    return any(marker in text for marker in ('"', "“", "”", "«", "»"))


def _find_quote_pairs(text: str) -> list[tuple[int, int]]:
    """Find all quotation mark pairs in text, returning (start, end) indices.

    Supports standard quotes ("), curly quotes (""), and guillemets («»).
    Returns list of (start, end) tuples where start is the opening quote index
    and end is the closing quote index (exclusive).
    """
    quote_pairs: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []  # (quote_char, position)

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
    """Check if a span is fully contained within quotation marks.

    Args:
        text: The original sentence text.
        span_start: Start index of span (inclusive) in text coordinates.
        span_end: End index of span (exclusive) in text coordinates.

    Returns:
        True if the span is fully contained within any quote pair.
    """
    quote_pairs = _find_quote_pairs(text)
    return any(
        q_start <= span_start and span_end <= q_end
        for q_start, q_end in quote_pairs
    )


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
        """Create a BLAKE2b fingerprint for *text*.

        Parameters
        ----------
        text:
            Document paragraph or sentence text.

        Returns
        -------
        Fingerprint
        """
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
        corpus_entries: list[CorpusEntryTuple],
        tfidf_info: tuple | None = None,
    ) -> list[SimilarityMatch]:
        """Return similarity matches for one sentence against a corpus.

        Parameters
        ----------
        sentence:
            The sentence to match.
        corpus_entries:
            List of (doc_id, normalized_text, Fingerprint, metadata, entry_index) tuples.
        tfidf_info:
            Optional (vectorizer, matrix) from corpus documents.

        Returns
        -------
        list[SimilarityMatch]
        """
        matches: list[SimilarityMatch] = []

        # 1) Exact/fingerprint overlap
        sentence_fp = self.build_fingerprint(sentence.normalized_text)
        tfidf_scores: dict[int, float] = {}
        if tfidf_info is not None:
            vectorizer, matrix = tfidf_info
            tfidf_scores = dict(
                compute_cosine_similarity(sentence.normalized_text, vectorizer, matrix)
            )

        for entry_index, entry in enumerate(corpus_entries):
            (
                doc_id, doc_text, doc_fp, metadata,
                source_entry_index, corpus_entry_obj,
            ) = entry
            eo = exact_overlap(sentence_fp, doc_fp)

            # 2) Lexical similarity
            if tfidf_info is not None:
                ls = tfidf_scores.get(entry_index, 0.0)
            else:
                # Fallback to char n-gram Jaccard
                ls = char_ngram_jaccard(sentence.normalized_text, doc_text)

            # 3) Classify match type
            mt = classify_match_type(
                exact_overlap=eo,
                lexical_similarity=ls,
                exact_threshold=self.config.exact_threshold,
                near_duplicate_threshold=self.config.near_duplicate_threshold,
                lexical_overlap_threshold=self.config.lexical_overlap_threshold,
            )

            # 4) Combined score (weighted)
            combined = (
                self.config.combined_weight_exact * eo
                + self.config.combined_weight_lexical * ls
            )

            # 5) Find matching spans using fingerprint
            segments = find_matching_segments_pair(
                sentence_fp, doc_fp, k=self.config.shingle_size
            )
            # Remap document spans from normalized sentence coords to original paragraph coords
            sentence_offset_map = build_offset_map(sentence.text, sentence.normalized_text)
            matched_doc_spans = [
                TextSpan(
                    paragraph_index=sentence.paragraph_index,
                    start=sentence.start_offset + remap_span(start, end, sentence_offset_map)[0],
                    end=sentence.start_offset + remap_span(start, end, sentence_offset_map)[1],
                )
                for start, end in segments.document_spans
            ]
            # Remap source spans from normalized corpus coords to original corpus coords
            # corpus_entry_obj is already unpacked from the tuple
            corpus_offset_map = None
            if (
                corpus_entry_obj
                and hasattr(corpus_entry_obj, 'offset_map')
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

            # 6) Build match
            match = SimilarityMatch(
                source_text=doc_text[:200],  # Truncate for storage
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
            )
            matches.append(match)

        # Filter out UNMATCHED, sort by combined score descending, then
        # cap results per sentence when max_results_per_sentence > 0.
        matches = [m for m in matches if m.match_type != MatchType.UNMATCHED]
        matches.sort(key=lambda m: -m.combined_score)
        if self.config.max_results_per_sentence > 0:
            matches = matches[:self.config.max_results_per_sentence]
        return matches

    # ------------------------------------------------------------------
    # Attribution risk (citation-aware)
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

        # Check if sentence has citations
        if not sentence_citations:
            if match.exact_overlap >= 0.95:
                return RiskLevel.HIGH, "high exact overlap without citation"
            return RiskLevel.MEDIUM, "similar text without citation"

        # Sentence has citations — check if ANY citation matches the detected source
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
                    f"citation from {cit.authors or 'unknown'} ({cit.year or 'n.d.'}) "
                    f"does not match detected source"
                )

        # Decision: if at least one citation matches, attribution is not "wrong source"
        if not at_least_one_matches:
            return RiskLevel.HIGH, "; ".join(citation_issues)
        
        # At least one citation matches the detected source
        if match.exact_overlap >= 0.95:
            # Span-aware quote detection: check if matched spans fall
            # within quotation marks.  Fallback to sentence-level check
            # when matched_document_spans are not populated.
            quoted = False
            if match.matched_document_spans:
                for span in match.matched_document_spans:
                    sent_rel_start = span.start - sentence_start_offset
                    sent_rel_end = span.end - sentence_start_offset
                    if (
                        0 <= sent_rel_start < sent_rel_end <= len(sentence_text)
                        and _span_within_quotes(
                            sentence_text, sent_rel_start, sent_rel_end
                        )
                    ):
                        quoted = True
                        break
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

        Uses paragraph-relative TextSpan objects so spans from different
        paragraphs do not incorrectly collide or merge.

        Parameters
        ----------
        results:
            SimilarityResult list from full-document analysis.
        doc_sentences:
            All sentences in the document (for eligible-char count).

        Returns
        -------
        float
        """
        eligible_chars = sum(
            s.length for s in doc_sentences if not s.is_bibliography
        )

        if eligible_chars == 0:
            return 0.0

        # Collect all matched document spans from all results + matches
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

        # Also add spans from best matches only for a lighter metric
        # (current implementation uses all matches)

        merged = _merge_spans(all_spans)
        unique_matched = sum(span.length() for span in merged)

        return (unique_matched / eligible_chars) * 100.0

    def analyze_document(
        self,
        sentences: list[Sentence],
        corpus_entries: list[CorpusEntryTuple],
        bibliography_entries: list[BibliographyEntry] | None = None,
    ) -> SimilarityEngineResult:
        """Run full similarity analysis over a document's sentences.

        Parameters
        ----------
        sentences:
            Document sentences (EnrichedDocument output).
        corpus_entries:
            Corpus entries as (doc_id, text, Fingerprint, metadata, entry_index).
        bibliography_entries:
            Bibliography entries used for citation-to-source matching.

        Returns
        -------
        SimilarityEngineResult
        """
        bibliography_entries = bibliography_entries or []
        # Build TF-IDF index from corpus if we have entries
        tfidf_info = None
        if corpus_entries:
            corpus_texts = [entry[1] for entry in corpus_entries]
            tfidf_info = build_tfidf_index(corpus_texts)

        results: list[SimilarityResult] = []
        high_risk = 0
        medium_risk = 0

        for sent in sentences:
            # Skip bibliography sentences for overall % but still analyze
            sent_matches = self.compare_sentence_to_corpus(
                sent, corpus_entries, tfidf_info
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

        # Count unique matched chars (from document spans)
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
