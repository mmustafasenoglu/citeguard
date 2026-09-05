from citeguard.bibliography import parse_bibliography
from citeguard.extractor import extract_citations
from citeguard.linking import link_citations_to_contexts


def test_links_citation_to_sentence_and_bibliography_entry() -> None:
    paragraphs = [
        "Background is provided. Smith (2020) reports the result. Another sentence.",
        "References",
        "Smith, J. (2020). Useful paper.",
    ]
    citations = extract_citations(paragraphs[:1])
    _, entries = parse_bibliography(paragraphs)

    contexts = link_citations_to_contexts(paragraphs, citations, entries)

    assert len(contexts) == 1
    assert contexts[0].sentence == "Smith (2020) reports the result."
    assert contexts[0].bibliography_entry_indexes == [0]


def test_keeps_two_citations_in_the_same_sentence() -> None:
    paragraph = "Two studies agree (Smith, 2020; Jones, 2021)."
    citations = extract_citations([paragraph])

    contexts = link_citations_to_contexts([paragraph], citations, [])

    assert len(contexts) == 2
    assert all(context.sentence == paragraph for context in contexts)


def test_empty_citations_returns_empty_contexts() -> None:
    paragraphs = ["Just some text with no citations."]
    contexts = link_citations_to_contexts(paragraphs, [], [])
    assert contexts == []


def test_citation_in_first_sentence() -> None:
    paragraphs = ["Smith (2020) found results. Another sentence explains more."]
    citations = extract_citations(paragraphs)

    contexts = link_citations_to_contexts(paragraphs, citations, [])

    assert len(contexts) == 1
    assert contexts[0].sentence == "Smith (2020) found results."


def test_citation_in_last_sentence() -> None:
    paragraphs = ["First sentence. Then Smith (2020) confirmed it."]
    citations = extract_citations(paragraphs)

    contexts = link_citations_to_contexts(paragraphs, citations, [])

    assert len(contexts) == 1
    assert contexts[0].sentence == "Then Smith (2020) confirmed it."


def test_multiple_paragraphs_different_citations() -> None:
    paragraphs = [
        "First claim (Smith, 2020).",
        "Second claim (Jones, 2021).",
    ]
    citations = extract_citations(paragraphs)

    contexts = link_citations_to_contexts(paragraphs, citations, [])

    assert len(contexts) == 2
    assert contexts[0].citation.paragraph_index == 0
    assert contexts[1].citation.paragraph_index == 1


def test_citation_not_matched_to_bibliography() -> None:
    paragraphs = [
        "A claim (Unknown, 2020).",
        "References",
        "Smith, J. (2020). Paper.",
    ]
    citations = extract_citations(paragraphs[:1])
    _, entries = parse_bibliography(paragraphs)

    contexts = link_citations_to_contexts(paragraphs, citations, entries)

    assert len(contexts) == 1
    assert contexts[0].bibliography_entry_indexes == []


def test_doi_citation_matches_bibliography_entry() -> None:
    paragraphs = [
        "See 10.1000/example for details.",
        "References",
        "Smith, J. (2020). Paper. 10.1000/example",
    ]
    citations = extract_citations(paragraphs[:1])
    _, entries = parse_bibliography(paragraphs)

    contexts = link_citations_to_contexts(paragraphs, citations, entries)

    assert len(contexts) == 1
    assert contexts[0].bibliography_entry_indexes == [0]


def test_citation_offsets_are_preserved() -> None:
    paragraphs = ["Several studies confirm this. Smith (2020) found results."]
    citations = extract_citations(paragraphs)

    contexts = link_citations_to_contexts(paragraphs, citations, [])

    assert len(contexts) == 1
    assert contexts[0].citation.char_offset > 0
    assert contexts[0].citation.authors == "Smith"
    assert contexts[0].citation.year == 2020
