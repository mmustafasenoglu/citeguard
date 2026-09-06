"""Deterministic Turkish-aware sentence splitter.

Uses regex-based boundary detection with protection for abbreviations,
decimal numbers, DOI strings, URLs, and numbered headings.
Does NOT use heavy NLP models or external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Protected abbreviations and patterns that must NOT trigger sentence split
# ---------------------------------------------------------------------------

# Lowercase abbreviation tokens (period included)
ABBREVIATIONS: set[str] = {
    "dr.", "prof.", "doç.", "müh.", "yrd.", "tek.", "mim.", "ibid.",
    "bkz.", "vb.", "vd.", "vs.", "örn.", "t.c.", "rh.", "mp.",
}

# Protected patterns' start-of-period positions.
# These are regex groups that, when they include a period at the given
# position, the period is NOT a sentence boundary.
PROTECTED_PATTERNS: list[re.Pattern[str]] = [
    # Prof. Dr. combinations anywhere in text
    re.compile(r"Prof\. Dr\.", re.IGNORECASE),
    # Decimal numbers like 3.5 or 1,5  (period or comma followed by digit)
    re.compile(r"(?<!\d)[.,](?=\d)"),
    # DOI-like: 10.xxxx  (must start at word boundary)
    re.compile(r"\b10\.", re.IGNORECASE),
    # URL scheme: http:// https:// ftp://  (period after scheme)
    re.compile(r"(?:https?|ftp)://", re.IGNORECASE),
    # Numbered headings like "1. Giriş"  (digit period at start of token)
    # Use a non-lookbehind approach: match digit+period at start or after space
    re.compile(r"(^|\s)\d+\.", re.IGNORECASE),
    # Ellipsis: three dots "..."  (treated as one protective token)
    re.compile(r"\.{3}(?=\s|$)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Sentence-split boundary: a period followed by whitespace then uppercase
#—but only when the period is NOT protected.
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])",
    re.UNICODE,
)


# ---------------------------------------------------------------------------
# TextSpan — paragraph-aware character span
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TextSpan:
    """A span of text within a specific paragraph.

    Unlike global offsets, these are paragraph-relative so that overlapping
    spans from different paragraphs do not collide.

    Attributes
    ----------
    paragraph_index:
        Which paragraph (0-based) this span belongs to.
    start:
        Inclusive character offset inside that paragraph.
    end:
        Exclusive character offset inside that paragraph.
    """

    paragraph_index: int
    start: int
    end: int

    def length(self) -> int:
        return self.end - self.start

    def includes(self, other: TextSpan) -> bool:
        """Return True if this span fully contains *other*."""
        return (
            self.paragraph_index == other.paragraph_index
            and self.start <= other.start
            and other.end <= self.end
        )

    def __gt__(self, other: TextSpan) -> bool:
        if self.paragraph_index != other.paragraph_index:
            return self.paragraph_index > other.paragraph_index
        return self.start > other.start

    def __lt__(self, other: TextSpan) -> bool:
        if self.paragraph_index != other.paragraph_index:
            return self.paragraph_index < other.paragraph_index
        return self.start < other.start


# ---------------------------------------------------------------------------
# Sentence model — paragraph-granular
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Sentence:
    """A sentence inside a paragraph, with accurate character offsets.

    Attributes
    ----------
    text:
        Original (non-normalized) sentence text.
    normalized_text:
        Lowercased, whitespace-normalized version for retrieval only.
    paragraph_index:
        Which paragraph (0-based) this sentence lives in.
    sentence_index:
        Which sentence within its paragraph (0-based).
    start_offset:
        Inclusive character offset inside the paragraph text.
    end_offset:
        Exclusive character offset inside the paragraph text.
    citations:
        ExistingCitation objects that fall inside this sentence's
        character range (filled in by EnrichedDocument builder).
    is_bibliography:
        True if this sentence lives inside the bibliography section.
    """

    text: str
    normalized_text: str
    paragraph_index: int
    sentence_index: int
    start_offset: int
    end_offset: int
    citations: list = field(default_factory=list)
    is_bibliography: bool = False

    @property
    def length(self) -> int:
        return self.end_offset - self.start_offset


# ---------------------------------------------------------------------------
# Abbreviation / protected-pattern check
# ---------------------------------------------------------------------------

def _is_protected_period(text: str, period_idx: int) -> bool:
    """Return True if the period at *period_idx* is *not* a sentence boundary.

    We check all protected patterns to see if any of them 'own' this period.
    Each pattern check is confined to a small window around the period to avoid
    false positives from earlier text.
    """
    # 1) Abbreviation list — if the period is immediately preceded by a known
    #    abbreviation word (including its trailing dot), it's protected.
    #    We look backwards for a word boundary + abbreviation.
    before = text[:period_idx].strip()
    words = before.split()
    if words:
        last_word = words[-1].lower()
        # ABBREVIATIONS entries include the trailing dot (e.g. "prof.")
        if last_word + "." in ABBREVIATIONS or last_word in ABBREVIATIONS:
            return True

    # 2) Regex-protected patterns — check only in a window around the period
    #    to avoid matching earlier occurrences (e.g. "Prof. Dr." when checking
    #    a later sentence-ending period).
    window_start = max(0, period_idx - 20)
    window_text = text[window_start:period_idx + 1]

    return any(pat.search(window_text) for pat in PROTECTED_PATTERNS)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def split_sentences(
    text: str,
    paragraph_index: int = 0,
    *,
    is_bibliography: bool = False,
) -> list[Sentence]:
    """Deterministic Turkish-aware sentence splitter.

    Returns a list of ``Sentence`` objects with accurate character offsets
    relative to *text* (which is expected to be one paragraph).

    Parameters
    ----------
    text:
        A single paragraph of text.
    paragraph_index:
        Which paragraph this text belongs to (used for the returned
        ``Sentence.paragraph_index``).
    is_bibliography:
        If True, every returned Sentence gets ``is_bibliography=True``.
        Used when parsing the bibliography section of a document.

    Returns
    -------
    list[Sentence]
    """
    if not text or not text.strip():
        return []

    working = text.strip()
    sentences: list[Sentence] = []
    pos = 0  # current search start (inclusive) inside *working*
    sent_idx = 0
    sent_start = 0  # start of the current sentence's text segment

    while pos < len(working):
        # Find the next sentence-ending boundary.
        m = _SENTENCE_END_RE.search(working, pos)
        if m is None:
            # No more boundaries — the remaining text is the final sentence.
            remainder = working[pos:].strip()
            if remainder:
                sentences.append(
                    Sentence(
                        text=remainder,
                        normalized_text=_normalize_turkish(remainder),
                        paragraph_index=paragraph_index,
                        sentence_index=sent_idx,
                        start_offset=sent_start,
                        end_offset=len(working),
                        is_bibliography=is_bibliography,
                    )
                )
            break

        boundary_start = m.start()  # position of the first char after the whitespace
        boundary_end = m.end()  # one-past the whitespace

        # Is the period before this whitespace protected?
        period_idx = m.start() - 1  # position of the period itself
        if _is_protected_period(working, period_idx):
            # Protected: skip this boundary but preserve sent_start
            pos = boundary_end
            continue

        # --- Valid sentence boundary ---
        sentence_text = working[sent_start:boundary_start].strip()
        if sentence_text:
            sentences.append(
                Sentence(
                    text=sentence_text,
                    normalized_text=_normalize_turkish(sentence_text),
                    paragraph_index=paragraph_index,
                    sentence_index=sent_idx,
                    start_offset=sent_start,
                    end_offset=boundary_start,
                    is_bibliography=is_bibliography,
                )
            )
            sent_idx += 1
            sent_start = boundary_end  # next sentence starts after this boundary

        else:
            # Didn't find a boundary - advance pos but keep sent_start
            pos = boundary_end

    return sentences


# ---------------------------------------------------------------------------
# Normalization (Turkish-aware, retrieval-only)
# ---------------------------------------------------------------------------

def _normalize_turkish(text: str) -> str:
    """Lowercase + whitespace normalize *only* these transformations:

    - ``İ`` → ``i``
    - ``I`` → ``ı``

    All other characters (ç, ğ, ı, İ, ö, ş, ü) are preserved; they are
    simply lowercased by Python's ``str.lower`` after the two critical
    substitutions.  Whitespace is collapsed.

    This function is *intentionaly* used only for retrieval / matching.
    Original text is never replaced in the model.
    """
    # 1) Handle the two cross-dotted/I problems first.
    text = text.replace("İ", "i").replace("I", "ı")

    # 2) Standard lower + collapse whitespace.
    #    Python's str.lower() is safe for all remaining Unicode chars
    #    (ç, ğ, ö, ş, ü remain correct).
    lowered = text.lower()
    collapsed = re.sub(r"\s+", " ", lowered).strip()
    return collapsed


# ---------------------------------------------------------------------------
# Exported symbol
# ---------------------------------------------------------------------------

__all__ = [
    "split_sentences",
    "Sentence",
    "TextSpan",
    "_normalize_turkish",
]