"""End-to-end source-aware plagiarism review pipeline."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from citeguard.extractor import parse_enriched_document
from citeguard.models import Paragraph, Sentence, TextSpan
from citeguard.similarity.embeddings import SemanticBackendError
from citeguard.similarity.embeddings.base import EmbeddingBackend
from citeguard.similarity.engine import SimilarityEngine
from citeguard.similarity.index import SimilarityIndex
from citeguard.similarity.models import MatchType, SimilarityConfig

from .models import (
    AttributionStatus,
    MatchSeverity,
    PlagiarismConfig,
    PlagiarismMatch,
    PlagiarismResult,
    PlagiarismSource,
    PlagiarismSummary,
    SourceContribution,
)
from .sources import build_sources

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_COMMON = {
    "further research is needed",
    "the results of this study show that",
    "according to previous studies",
    "bu çalışmanın sonuçları göstermektedir ki",
    "daha fazla araştırmaya ihtiyaç vardır",
}
_SEVERITY_ORDER = {
    MatchSeverity.CRITICAL: 0,
    MatchSeverity.HIGH: 1,
    MatchSeverity.MEDIUM: 2,
    MatchSeverity.LOW: 3,
}


def _word_spans(text: str, start: int = 0) -> list[tuple[int, int]]:
    return [(start + match.start(), start + match.end()) for match in _WORD_RE.finditer(text)]


def _covered_words(text: str, start: int, spans: list[TextSpan]) -> set[int]:
    words = _word_spans(text, start)
    return {
        word_start
        for word_start, word_end in words
        if any(span.start < word_end and word_start < span.end for span in spans)
    }


def _quote_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for pattern in (r'"[^"\n]+"', r"“[^”\n]+”", r"‘[^’\n]+’", r"«[^»\n]+»"):
        ranges.extend(match.span() for match in re.finditer(pattern, text))
    return ranges


def _quoted(sentence: Sentence, spans: list[TextSpan], paragraph_text: str) -> bool:
    if paragraph_text.lstrip().startswith(">"):
        return True
    ranges = _quote_ranges(paragraph_text)
    if not ranges:
        return False
    relevant = spans or [
        TextSpan(sentence.paragraph_index, sentence.start_offset, sentence.end_offset)
    ]
    for span in relevant:
        local_start = span.start
        local_end = span.end
        if any(q_start <= local_start and local_end <= q_end for q_start, q_end in ranges):
            return True
    return False


def _verbatim_span(sentence: Sentence, source_text: str) -> TextSpan | None:
    """Locate an actual source substring despite surrounding citation/quote markup."""
    start = sentence.text.casefold().find(source_text.casefold())
    if start < 0:
        return None
    return TextSpan(
        sentence.paragraph_index,
        sentence.start_offset + start,
        sentence.start_offset + start + len(source_text),
    )


def _common_phrase(text: str, source_frequency: int, config: PlagiarismConfig) -> bool:
    normalized = " ".join(word.lower() for word in _WORD_RE.findall(text))
    return normalized in _COMMON or (
        source_frequency >= config.common_phrase_source_count
        and len(normalized.split()) <= config.common_phrase_max_words
    )


def _classify(
    match_type: MatchType,
    *,
    lexical_score: float,
    quoted: bool,
    cited: bool,
    common: bool,
    bibliography: bool,
    matched_words: int,
    critical_exact_words: int,
) -> tuple[AttributionStatus, MatchSeverity, str, str]:
    if bibliography:
        return (
            AttributionStatus.REVIEW_REQUIRED,
            MatchSeverity.LOW,
            "The overlap occurs in the bibliography and is excluded from review by default.",
            "No action is normally required unless bibliography comparison was requested.",
        )
    if common:
        return (
            AttributionStatus.POSSIBLE_COMMON_PHRASE,
            MatchSeverity.LOW,
            "The wording appears across several sources or is generic academic language.",
            "Review only if the phrase is distinctive in context.",
        )
    if quoted and cited:
        return (
            AttributionStatus.QUOTED_AND_ATTRIBUTED,
            MatchSeverity.LOW,
            "The overlapping wording is quoted and the passage contains a citation.",
            "Confirm the quotation and source locator are accurate.",
        )
    if quoted:
        return (
            AttributionStatus.QUOTED_NOT_ATTRIBUTED,
            MatchSeverity.HIGH,
            "The overlapping wording is quoted but no citation was detected.",
            "Add the correct citation; do not invent a source.",
        )
    if cited and (
        match_type == MatchType.EXACT
        or (
            match_type in {MatchType.NEAR_DUPLICATE, MatchType.LEXICAL_OVERLAP}
            and lexical_score >= 0.85
        )
    ):
        return (
            AttributionStatus.CITED_BUT_VERBATIM,
            MatchSeverity.HIGH,
            "The source is cited, but the passage reproduces source wording without quotation.",
            "Quote and cite when wording is necessary, or rewrite independently with citation.",
        )
    if match_type == MatchType.EXACT:
        severity = (
            MatchSeverity.CRITICAL if matched_words >= critical_exact_words else MatchSeverity.HIGH
        )
        return (
            AttributionStatus.UNATTRIBUTED_EXACT,
            severity,
            "Substantial exact source wording appears without a detected citation.",
            "Rewrite independently with the correct citation, or quote and cite.",
        )
    if match_type in {MatchType.NEAR_DUPLICATE, MatchType.LEXICAL_OVERLAP}:
        if cited:
            return (
                AttributionStatus.ATTRIBUTED,
                MatchSeverity.LOW,
                "Close lexical overlap appears in a passage containing a citation.",
                "Review whether the wording is sufficiently independent.",
            )
        return (
            AttributionStatus.UNATTRIBUTED_NEAR_EXACT,
            MatchSeverity.HIGH,
            "Close lexical overlap appears without a detected citation.",
            "Paraphrase independently and add the correct citation.",
        )
    if cited:
        return (
            AttributionStatus.PARAPHRASE_ATTRIBUTED,
            MatchSeverity.LOW,
            "Semantic dependence is accompanied by a citation.",
            "Confirm that the citation supports the passage.",
        )
    return (
        AttributionStatus.UNATTRIBUTED_SEMANTIC,
        MatchSeverity.MEDIUM,
        "A conservative semantic match was found without a detected citation.",
        "Review the source and add attribution if the idea derives from it.",
    )


def _with_embeddings(
    index: SimilarityIndex,
    config: PlagiarismConfig,
    sources: list[PlagiarismSource],
) -> tuple[SimilarityIndex, EmbeddingBackend]:
    from citeguard.similarity.embeddings.sentence_transformers import SentenceTransformerBackend

    from .indexing import cache_key

    backend = SentenceTransformerBackend(config.semantic_model, allow_download=not config.offline)
    embedding_key = hashlib.sha256(
        f"{cache_key(sources, config)}:{backend.model_name}:{backend.dimension}".encode()
    ).hexdigest()
    cache_path = Path(".local/citeguard/plagiarism-embeddings") / f"{embedding_key}.npy"
    embeddings = None
    if not config.no_cache and cache_path.exists():
        import numpy as np

        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape == (index.size, backend.dimension):
            embeddings = cached
    if embeddings is None:
        embeddings = backend.encode([entry[1] for entry in index.entries])
        if not config.no_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{cache_path.stem}-", suffix=".npy", dir=cache_path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                import numpy as np

                np.save(temporary, embeddings, allow_pickle=False)
                os.replace(temporary, cache_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
    enriched = SimilarityIndex(
        entries=index.entries,
        fingerprints=index.fingerprints,
        tfidf_vectorizer=index.tfidf_vectorizer,
        tfidf_matrix=index.tfidf_matrix,
        embeddings=embeddings,
        embedding_model=backend.model_name,
        embedding_dim=backend.dimension,
    )
    return enriched, backend


def _coverage_percent(keys: set[tuple[int, int]], denominator: int) -> float:
    return round(100.0 * len(keys) / denominator, 2) if denominator else 0.0


def _source_base_offset(index: SimilarityIndex, source_id: str | None, text: str) -> int:
    for doc_id, _normalized, _fingerprint, _metadata, _entry_index, source in index.entries:
        source_text = getattr(source, "original_text", None) or getattr(source, "text", "")
        if doc_id == source_id and source_text == text:
            return int(
                getattr(source, "offset_in_parent", None) or getattr(source, "char_offset", 0)
            )
    return 0


def _paragraph_bases(path: Path, paragraphs: list[Paragraph]) -> tuple[dict[int, int], bool]:
    """Map parser paragraphs into the original text stream where one exists."""
    if path.suffix.lower() not in {".md", ".txt"}:
        cursor = 0
        bases: dict[int, int] = {}
        for paragraph in paragraphs:
            bases[paragraph.index] = cursor
            cursor += len(paragraph.text) + 2
        return bases, False
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    cursor = 0
    exact = True
    bases = {}
    for paragraph in paragraphs:
        location = raw.find(paragraph.text, cursor)
        if location < 0:
            exact = False
            location = cursor
        bases[paragraph.index] = location
        cursor = location + len(paragraph.text)
    return bases, exact


def scan_document(
    document: str | Path,
    *,
    corpora: list[Path] | None = None,
    sources: list[Path] | None = None,
    source_urls: list[str] | None = None,
    config: PlagiarismConfig | None = None,
) -> PlagiarismResult:
    """Scan a document against explicit textual sources and local corpora."""
    path = Path(document)
    config = config or PlagiarismConfig()
    parsed = parse_enriched_document(path)
    if not parsed.sentences:
        raise ValueError("The document is empty or contains only whitespace")
    discovery_queries = [
        " ".join(part for part in (entry.title, entry.authors, str(entry.year or "")) if part)
        for entry in parsed.bibliography_entries
    ]
    built = build_sources(
        corpora or [],
        sources or [],
        source_urls or [],
        config,
        discovery_queries=discovery_queries,
    )
    warnings = list(built.warnings)
    index = built.index
    backend = None
    if config.semantic and index.size:
        try:
            index, backend = _with_embeddings(index, config, built.sources)
        except (ImportError, RuntimeError, ValueError, SemanticBackendError) as exc:
            warnings.append(f"Semantic mode unavailable; lexical fallback used: {exc}")

    engine_config = SimilarityConfig(
        exact_threshold=config.exact_threshold,
        near_duplicate_threshold=config.near_exact_threshold,
        lexical_overlap_threshold=config.lexical_threshold,
        min_match_length=config.min_match_words,
        max_results_per_sentence=config.max_sources,
        enable_semantic=backend is not None,
        semantic_threshold=config.semantic_threshold,
        allow_model_download=not config.offline,
        max_candidates=config.max_candidates,
    )
    engine = SimilarityEngine(engine_config)
    source_frequency: dict[str, set[str]] = defaultdict(set)
    for entry in index.entries:
        source_frequency[entry[1]].add(entry[0])

    matches: list[PlagiarismMatch] = []
    match_keys: dict[str, set[tuple[int, int]]] = {}
    paragraph_bases, global_offsets_exact = _paragraph_bases(path, parsed.paragraphs)
    if not global_offsets_exact:
        warnings.append(
            "Document-wide offsets use the parser's canonical paragraph stream; "
            "paragraph-relative offsets remain authoritative."
        )

    for sentence in parsed.sentences:
        if index.size == 0:
            continue
        fingerprint = engine.build_fingerprint(sentence.normalized_text)
        query_embedding = backend.encode([sentence.normalized_text])[0] if backend else None
        candidates = index.retrieve_candidates(
            sentence.normalized_text,
            fingerprint,
            query_embedding=query_embedding,
            top_k=config.max_candidates,
        )
        found = engine.compare_sentence_to_corpus(
            sentence,
            index=index,
            candidate_indices=candidates.indices,
            semantic_scores=candidates.semantic_scores if backend else None,
        )
        source_count = len({match.source_id for match in found})
        for candidate in found:
            verbatim_span = _verbatim_span(sentence, candidate.source_text)
            effective_type = MatchType.EXACT if verbatim_span else candidate.match_type
            spans = candidate.matched_document_spans
            if verbatim_span is not None:
                spans = [verbatim_span]
            elif candidate.match_type in {MatchType.EXACT, MatchType.SEMANTIC_OVERLAP}:
                spans = [
                    TextSpan(sentence.paragraph_index, sentence.start_offset, sentence.end_offset)
                ]
            covered = _covered_words(sentence.text, sentence.start_offset, spans)
            if len(covered) < config.min_match_words:
                continue
            keys = {(sentence.paragraph_index, word) for word in covered}
            document_text = sentence.text
            paragraph_text = parsed.paragraphs[sentence.paragraph_index].text
            is_quoted = _quoted(sentence, spans, paragraph_text)
            cited = bool(sentence.citations) or any(
                citation.paragraph_index == sentence.paragraph_index
                for citation in parsed.citations
            )
            common = _common_phrase(document_text, source_count, config)
            status, severity, explanation, recommendation = _classify(
                effective_type,
                lexical_score=candidate.lexical_similarity,
                quoted=is_quoted,
                cited=cited,
                common=common,
                bibliography=sentence.is_bibliography,
                matched_words=len(covered),
                critical_exact_words=config.critical_exact_words,
            )
            document_start = min(span.start for span in spans)
            document_end = max(span.end for span in spans)
            document_text = paragraph_text[document_start:document_end]
            source_spans = candidate.matched_source_spans
            if effective_type == MatchType.EXACT:
                source_start, source_end = 0, len(candidate.source_text)
            else:
                source_start = min((span.start for span in source_spans), default=0)
                source_end = max(
                    (span.end for span in source_spans), default=len(candidate.source_text)
                )
            source_text = candidate.source_text[source_start:source_end]
            source_base = _source_base_offset(index, candidate.source_id, candidate.source_text)
            digest = hashlib.sha256(
                f"{sentence.paragraph_index}:{sentence.start_offset}:{candidate.source_id}:"
                f"{source_base + source_start}:{source_base + source_end}:"
                f"{effective_type.value}:{candidate.source_text}".encode()
            ).hexdigest()[:16]
            record = PlagiarismMatch(
                match_id=digest,
                document_start=paragraph_bases[sentence.paragraph_index] + document_start,
                document_end=paragraph_bases[sentence.paragraph_index] + document_end,
                document_paragraph=sentence.paragraph_index,
                document_paragraph_start=document_start,
                document_paragraph_end=document_end,
                document_text=document_text,
                source_id=candidate.source_id or "unknown",
                source_start=source_base + source_start,
                source_end=source_base + source_end,
                source_text=source_text,
                match_type=effective_type.value,
                exact_score=round(1.0 if verbatim_span else candidate.exact_overlap, 6),
                lexical_score=round(candidate.lexical_similarity, 6),
                semantic_score=round(candidate.semantic_similarity_raw, 6),
                combined_score=round(
                    max(
                        1.0 if verbatim_span else 0.0,
                        candidate.combined_score,
                        candidate.semantic_similarity_raw,
                    ),
                    6,
                ),
                matched_words=len(covered),
                document_words=len(_word_spans(sentence.text)),
                source_words=len(_word_spans(source_text)),
                quoted=is_quoted,
                citation_present=cited,
                bibliography_region=sentence.is_bibliography,
                attribution_status=status,
                severity=severity,
                confidence=round(
                    max(
                        candidate.exact_overlap,
                        candidate.lexical_similarity,
                        candidate.semantic_similarity_raw,
                        1.0 if verbatim_span else 0.0,
                    ),
                    6,
                ),
                explanation=explanation,
                recommendation=recommendation,
            )
            matches.append(record)
            match_keys[digest] = keys

    matches.sort(
        key=lambda item: (
            _SEVERITY_ORDER[item.severity],
            -item.combined_score,
            item.document_paragraph,
            item.document_start,
            item.source_id,
        )
    )
    matches = matches[: config.max_matches]
    raw_keys = set()
    review_keys = set()
    type_keys = defaultdict(set)
    quoted_keys = set()
    bibliography_keys = set()
    attributed_keys = set()
    unattributed_keys = set()
    for match in matches:
        keys = match_keys[match.match_id]
        raw_keys.update(keys)
        if match.match_type == MatchType.EXACT.value:
            type_keys["exact"].update(keys)
        elif match.match_type == MatchType.SEMANTIC_OVERLAP.value:
            type_keys["semantic"].update(keys)
        else:
            type_keys["lexical"].update(keys)
        if match.quoted:
            quoted_keys.update(keys)
        if match.bibliography_region:
            bibliography_keys.update(keys)
        attributed = match.attribution_status in {
            AttributionStatus.ATTRIBUTED,
            AttributionStatus.QUOTED_AND_ATTRIBUTED,
            AttributionStatus.PARAPHRASE_ATTRIBUTED,
            AttributionStatus.CITED_BUT_VERBATIM,
        }
        (attributed_keys if attributed else unattributed_keys).update(keys)
        reviewable = match.attribution_status != AttributionStatus.POSSIBLE_COMMON_PHRASE
        if match.quoted and not config.include_quotes:
            reviewable = False
        if match.bibliography_region and not config.include_bibliography:
            reviewable = False
        if reviewable:
            review_keys.update(keys)
    all_words = sum(len(_word_spans(sentence.text)) for sentence in parsed.sentences)
    eligible_words = sum(
        len(_word_spans(sentence.text))
        for sentence in parsed.sentences
        if config.include_bibliography or not sentence.is_bibliography
    )
    denominator = max(all_words, 1)

    # Allocate each review-relevant word once to the strongest source.
    allocated: set[tuple[int, int]] = set()
    contribution_words: dict[str, set[tuple[int, int]]] = defaultdict(set)
    contribution_counts: dict[str, int] = defaultdict(int)
    for match in matches:
        keys = match_keys[match.match_id] & review_keys
        unique = keys - allocated
        contribution_words[match.source_id].update(unique)
        contribution_counts[match.source_id] += 1
        allocated.update(unique)
    source_map = {source.source_id: source for source in built.sources}
    contributions = [
        SourceContribution(
            source_id,
            source_map[source_id].title,
            len(keys),
            _coverage_percent(keys, denominator),
            contribution_counts[source_id],
        )
        for source_id, keys in contribution_words.items()
        if keys and source_id in source_map
    ]
    contributions.sort(key=lambda item: (-item.percent, item.source_id))
    exact_review = type_keys["exact"] & review_keys
    lexical_review = (type_keys["lexical"] & review_keys) - exact_review
    semantic_review = (type_keys["semantic"] & review_keys) - exact_review - lexical_review
    summary = PlagiarismSummary(
        _coverage_percent(raw_keys, max(all_words, 1)),
        _coverage_percent(review_keys, denominator),
        _coverage_percent(exact_review, denominator),
        _coverage_percent(lexical_review, denominator),
        _coverage_percent(semantic_review, denominator),
        _coverage_percent(quoted_keys, max(all_words, 1)),
        _coverage_percent(bibliography_keys, max(all_words, 1)),
        _coverage_percent(attributed_keys, denominator),
        _coverage_percent(unattributed_keys, denominator),
        eligible_words,
        all_words,
        len(raw_keys),
        len(review_keys),
        sum(
            1 for match in matches if match.severity in {MatchSeverity.CRITICAL, MatchSeverity.HIGH}
        ),
    )
    return PlagiarismResult(
        path, built.sources, matches, contributions, summary, config, warnings, index
    )
