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
    find_matching_segments,
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

if TYPE_CHECKING:
    from citeguard.models import (
        BibliographyEntry,
        ExistingCitation,
        Sentence,
    )

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
        corpus_entries: list[tuple[str, str, Fingerprint]],  # (doc_id, text, Fingerprint)
        tfidf_info: tuple | None = None,
    ) -> list[SimilarityMatch]:
        """Return similarity matches for one sentence against a corpus.

        Parameters
        ----------
        sentence:
            The sentence to match.
        corpus_entries:
            List of (doc_id, normalized_text, Fingerprint) tuples from the corpus.
        tfidf_info:
            Optional (vectorizer, matrix) from corpus documents.

        Returns
        -------
        list[SimilarityMatch]
        """
        matches: list[SimilarityMatch] = []

        # 1) Exact/fingerprint overlap
        sentence_fp = self.build_fingerprint(sentence.normalized_text)

        for doc_id, doc_text, doc_fp in corpus_entries:
            eo = exact_overlap(sentence_fp, doc_fp)

            # 2) Lexical similarity
            ls = 0.0
            if tfidf_info is not None:
                vectorizer, matrix = tfidf_info
                # Use TF-IDF cosine similarity
                scores = compute_cosine_similarity(sentence.normalized_text, vectorizer, matrix)
                # Find the score for this document
                for idx, score in scores:
                    if idx < len(corpus_entries) and corpus_entries[idx][0] == doc_id:
                        ls = score
                        break
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
            matched_doc_spans = find_matching_segments(
                sentence_fp, doc_fp, k=self.config.shingle_size
            )
            # For source spans, we'd need the original corpus document's fingerprint
            # For now, use the same spans as approximation
            matched_src_spans = matched_doc_spans

            # 6) Build match
            match = SimilarityMatch(
                source_text=doc_text[:200],  # Truncate for storage
                source_title=doc_id,
                source_id=doc_id,
                exact_overlap=eo,
                lexical_similarity=ls,
                combined_score=combined,
                match_type=mt,
                matched_document_spans=matched_doc_spans,
                matched_source_spans=matched_src_spans,
            )
            matches.append(match)

        # Filter out UNMATCHED and sort by combined score descending
        matches = [m for m in matches if m.match_type != MatchType.UNMATCHED]
        matches.sort(key=lambda m: -m.combined_score)
        return matches

    # ------------------------------------------------------------------
    # Attribution risk (citation-aware)
    # ------------------------------------------------------------------

    def compute_attribution_risk(
        self,
        match: SimilarityMatch,
        sentence_citations: list[ExistingCitation],
        bibliography_entries: list[BibliographyEntry],
    ) -> tuple[RiskLevel, str]:
        """Determine attribution risk for a match given the sentence's citations.

        Risk logic (from v0.3 design):
        - No match → NONE (should not happen if match list non-empty)
        - Match exists:
          - No citation in sentence → HIGH (claim unsupported)
          - Citation present:
            - Citation matches detected source → LOW
            - Citation does NOT match detected source → HIGH (wrong source)
            - Quotation markers present → MEDIUM
        - Severity adjustment:
          - exact >= 0.95 → +1 (boost confidence)
          - exact < 0.70 → -1 (penalize)
        """
        if not match:
            return RiskLevel.NONE, "no match found"

        detected_source = match.source_text or match.source_id or ""

        # Check if sentence has citations
        if not sentence_citations:
            # Upgrade based on exact overlap
            if match.exact_overlap >= 0.95:
                return RiskLevel.LOW, "high exact overlap, no citation"
            if match.exact_overlap < 0.70:
                return RiskLevel.HIGH, "low exact overlap, no citation"
            return RiskLevel.MEDIUM, "no citation, moderate overlap"

        # Sentence has citations — check each
        citation_issues: list[str] = []

        for cit in sentence_citations:
            # Check if citation author/year matches detected source
            # Use author and year for matching
            cit_identifier = ""
            if cit.authors:
                cit_identifier += cit.authors.lower()
            if cit.year:
                cit_identifier += f" {cit.year}"
            
            citation_matches_source = (
                detected_source
                and cit_identifier
                and cit_identifier in detected_source.lower()
            )

            if not citation_matches_source:
                citation_issues.append(
                    f"citation from {cit.authors or 'unknown'} ({cit.year or 'n.d.'}) "
                    f"does not match detected source"
                )

        # Decision
        if citation_issues:
            base_risk = RiskLevel.HIGH
            reason = "; ".join(citation_issues)
        else:
            # All citations match — downgrade based on exact strength
            if match.exact_overlap >= 0.95:
                return RiskLevel.LOW, "citations match detected source"
            if match.exact_overlap < 0.70:
                return RiskLevel.MEDIUM, "citations match but low exact overlap"
            return RiskLevel.LOW, "citations match detected source"

        # Severity adjustment based on exact overlap
        if match.exact_overlap >= 0.95:
            if base_risk == RiskLevel.HIGH:
                base_risk = RiskLevel.MEDIUM
        elif match.exact_overlap < 0.70:
            if base_risk == RiskLevel.LOW:
                base_risk = RiskLevel.MEDIUM
            elif base_risk == RiskLevel.MEDIUM:
                base_risk = RiskLevel.HIGH

        return base_risk, reason

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
                            start=span[0],
                            end=span[1],
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
        corpus_entries: list[tuple[str, str, Fingerprint]],  # (doc_id, text, Fingerprint)
    ) -> SimilarityEngineResult:
        """Run full similarity analysis over a document's sentences.

        Parameters
        ----------
        sentences:
            Document sentences (EnrichedDocument output).
        corpus_entries:
            Corpus entries as (doc_id, normalized_text, Fingerprint) tuples.

        Returns
        -------
        SimilarityEngineResult
        """
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
                    getattr(sent, "bibliography_entries", []),
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
                            start=sp[0],
                            end=sp[1],
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