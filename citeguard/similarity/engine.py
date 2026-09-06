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
from citeguard.similarity.fingerprint import exact_overlap, generate_shingles, winnow
from citeguard.similarity.lexical import (
    build_tfidf_index,
    char_ngram_jaccard,
    classify_match_type,
)
from citeguard.similarity.models import (
    Fingerprint,
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
        corpus_fingerprints: list[tuple[str, Fingerprint]],
        tfidf_info: tuple | None = None,
    ) -> list[SimilarityMatch]:
        """Return similarity matches for one sentence against a corpus.

        Parameters
        ----------
        sentence:
            The sentence to match.
        corpus_fingerprints:
            List of (doc_id, Fingerprint) tuples from the corpus.
        tfidf_info:
            Optional (vectorizer, matrix) from corpus documents.

        Returns
        -------
        list[SimilarityMatch]
        """
        matches: list[SimilarityMatch] = []

        # 1) Exact/fingerprint overlap
        sentence_fp = self.build_fingerprint(sentence.normalized_text)

        for doc_id, doc_fp in corpus_fingerprints:
            eo = exact_overlap(sentence_fp, doc_fp)

            # 2) Lexical (TF-IDF) similarity if we have matrix
            ls = 0.0
            if tfidf_info is not None:
                vectorizer, matrix = tfidf_info
                # Simple: compare sentence against each doc in matrix
                # This is a per-document query; in production would use
                # the vectorizer's transform more efficiently.
                # For now, fall back to char-n-gram Jaccard as proxy.
                ls = char_ngram_jaccard(
                    sentence.normalized_text,
                    doc_id,  # doc_id is actually text; should fix
                )

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

            # 5) Build match
            match = SimilarityMatch(
                source_text=doc_id,
                source_title="",  # would be populated from corpus metadata
                source_id=doc_id,
                exact_overlap=eo,
                lexical_similarity=ls,
                combined_score=combined,
                match_type=mt,
            )
            matches.append(match)

        # Sort by combined score descending
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
        if not match.matches:
            return RiskLevel.NONE, "no match found"

        best = match.matches[0]
        detected_source = best.source_text

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
        has_quotation = False

        for cit in sentence_citations:
            # Check if citation source matches best detected source
            # This is a simplified check; real implementation would compare
            # citation metadata (author, year, etc.) against match source
            citation_matches_source = (
                detected_source
                and detected_source.lower() in cit.source.lower()
            )

            if not citation_matches_source:
                citation_issues.append(
                    f"citation from {cit.author} ({cit.year}) "
                    f"does not match detected source"
                )

            # Check for quotation markers in original text
            if cit.is_quotation:
                has_quotation = True

        # Decision
        if citation_issues and has_quotation:
            base_risk = RiskLevel.MEDIUM
            reason = (
                f"quotation markers present; {'; '.join(citation_issues)}"
            )
        elif citation_issues:
            base_risk = RiskLevel.HIGH
            reason = "; ".join(citation_issues)
        else:
            # All citations match — downgngrade based on exact strength
            if match.exact_overlap >= 0.95:
                return RiskLevel.LOW, "citations match detected source"
            if match.exact_overlap < 0.70:
                return RiskLevel.HIGH, "citations present but weak exact overlap"
            return RiskLevel.MEDIUM, "citations match source, moderate overlap"

        # Apply severity adjustment
        risk = base_risk
        reason = f"{reason}; exact={match.exact_overlap:.2f}"
        if match.exact_overlap >= 0.95:
            risk = RiskLevel(risk.value.rstrip("0123456789") or "low")
            # Simple promotion
            if risk == RiskLevel.MEDIUM:
                risk = RiskLevel.LOW
        elif match.exact_overlap < 0.70 and risk != RiskLevel.HIGH:
            risk = RiskLevel.HIGH

        return risk, reason

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
        unique_matched = sum(span.length for span in merged)

        return (unique_matched / eligible_chars) * 100.0

    # ------------------------------------------------------------------
    # Full-document analysis
    # ------------------------------------------------------------------

    def analyze_document(
        self,
        sentences: list[Sentence],
        corpus_fps: list[tuple[str, Fingerprint]],
        tfidf_info: tuple | None = None,
    ) -> SimilarityEngineResult:
        """Run full similarity analysis over a document's sentences.

        Parameters
        ----------
        sentences:
            Document sentences (EnrichedDocument output).
        corpus_fps:
            Corpus fingerprints (doc_id, Fingerprint).
        tfidf_info:
            Optional TF-IDF index for lexical comparisons.

        Returns
        -------
        SimilarityEngineResult
        """
        results: list[SimilarityResult] = []
        high_risk = 0
        medium_risk = 0

        for sent in sentences:
            # Skip bibliography sentences for overall % but still analyze
            sent_matches = self.compare_sentence_to_corpus(
                sent, corpus_fps, tfidf_info
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

        # Count unique matched chars
        eligible = sum(s.length for s in sentences if not s.is_bibliography)
        unique_matched = 0
        # Recompute from results spans
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
            unique_matched = sum(s.length for s in merged)

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