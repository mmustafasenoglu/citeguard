from citeguard.claims import extract_claims
from citeguard.extractor import extract_citations
from citeguard.models import ClaimType, Severity


def test_extracts_statistic_claim() -> None:
    paragraphs = [
        "Over 70% of neural networks use dropout regularization.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.STATISTIC
    assert claims[0].severity == Severity.HIGH
    assert "70%" in claims[0].text


def test_extracts_causal_claim() -> None:
    paragraphs = [
        "Dropout regularization causes improved generalization in deep models.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.CAUSAL
    assert claims[0].severity == Severity.HIGH


def test_extracts_comparative_claim() -> None:
    paragraphs = [
        "Transformers perform more than 20% better than RNNs on translation tasks.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.COMPARATIVE
    assert claims[0].severity == Severity.MEDIUM


def test_extracts_historical_claim() -> None:
    paragraphs = [
        "The transformer architecture was introduced in 2017 by Vaswani et al.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.HISTORICAL
    assert claims[0].severity == Severity.MEDIUM


def test_extracts_definition_claim() -> None:
    paragraphs = [
        "Transfer learning is defined as leveraging knowledge from one task "
        "to improve performance on a different but related task.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.DEFINITION
    assert claims[0].severity == Severity.MEDIUM


def test_extracts_prior_work_claim() -> None:
    paragraphs = [
        "Previous work has established that attention mechanisms improve sequence modeling.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.PRIOR_WORK


def test_extracts_quotation_claim() -> None:
    paragraphs = [
        'The author stated that "deep learning has transformed NLP".',
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.QUOTATION
    assert claims[0].severity == Severity.HIGH


def test_skips_headings() -> None:
    paragraphs = [
        "# Introduction",
        "Some real claim here with enough words to pass.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].paragraph_index == 1


def test_skips_short_paragraphs() -> None:
    paragraphs = [
        "See below.",
        "A real claim with enough detail should be extracted properly.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].paragraph_index == 1


def test_skips_instruction_paragraphs() -> None:
    paragraphs = [
        "Please ensure all citations are formatted correctly.",
        "The model achieves 95% accuracy on the benchmark dataset.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].paragraph_index == 1


def test_skips_turkish_form_instructions_but_keeps_research_prose() -> None:
    paragraphs = [
        "Başvuru formunun verilen açıklamalara göre hazırlanması beklenir.",
        "Bu bölümün başvuru tamamlanmadan önce yazılması önerilir.",
        "Uydu görüntüleri eğitim sürecinde karşılaştırmalı olarak kullanılacaktır.",
    ]

    claims = extract_claims(paragraphs, [], bibliography_start=None)

    assert [claim.paragraph_index for claim in claims] == [2]


def test_skips_uppercase_form_headings_and_identity_fields() -> None:
    paragraphs = [
        "ARAŞTIRMA ÖNERİSİNİN BİLİMSEL NİTELİĞİ",
        "Başvuru Sahibinin Adı Soyadı: Example Student",
        "Model performs consistently across independent evaluation regions.",
    ]

    claims = extract_claims(paragraphs, [], bibliography_start=None)

    assert [claim.paragraph_index for claim in claims] == [2]


def test_paragraph_offset_preserves_global_index_and_citation_link() -> None:
    paragraph = "Transformers were introduced in 2017 (Vaswani et al., 2017)."
    citations = extract_citations(["placeholder"] * 7 + [paragraph])

    claims = extract_claims(
        [paragraph],
        citations,
        bibliography_start=None,
        paragraph_offset=7,
    )

    assert len(claims) == 1
    assert claims[0].paragraph_index == 7
    assert claims[0].has_existing_citation is True


def test_skips_bibliography_section() -> None:
    paragraphs = [
        "Transformers achieve state-of-the-art results across many NLP benchmarks.",
        "References",
        "Vaswani, A. (2017). Attention Is All You Need.",
    ]
    claims = extract_claims(
        paragraphs, [], bibliography_start=1
    )
    assert len(claims) == 1
    assert claims[0].paragraph_index == 0


def test_respects_max_claims() -> None:
    paragraphs = [
        "A study shows that 80% of models use batch normalization techniques.",
        "Research indicates that dropout improves generalization in neural networks.",
        "Experiments demonstrate that 90% accuracy was achieved on the test set.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None, max_claims=2)
    assert len(claims) == 2


def test_marks_cited_claims() -> None:
    paragraphs = [
        "Transformers were introduced in 2017 (Vaswani et al., 2017).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].has_existing_citation is True


def test_marks_uncited_claims() -> None:
    paragraphs = [
        "Large language models can generate realistic but false citations.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    assert claims[0].has_existing_citation is False


def test_severity_classification() -> None:
    paragraphs = [
        "Over 50% of papers use transformer architectures (statistic).",
        "Dropout causes regularization effects (causal).",
        "Model A is more than Model B (comparative).",
        "The method was proposed in 2017 (historical).",
        "Deep learning refers to multi-layer neural networks (definition).",
        "Some general observation about the world (general fact).",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    severity_map = {c.claim_type: c.severity for c in claims}
    assert severity_map[ClaimType.STATISTIC] == Severity.HIGH
    assert severity_map[ClaimType.CAUSAL] == Severity.HIGH
    assert severity_map[ClaimType.COMPARATIVE] == Severity.MEDIUM
    assert severity_map[ClaimType.HISTORICAL] == Severity.MEDIUM
    assert severity_map[ClaimType.DEFINITION] == Severity.MEDIUM
    assert severity_map[ClaimType.GENERAL_FACT] == Severity.LOW


def test_search_query_generation() -> None:
    paragraphs = [
        "Transformers have achieved state-of-the-art results in natural language processing.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    query = claims[0].search_query
    assert "transformers" in query.lower()
    assert len(query.split()) > 0


def test_claim_text_capped_at_25_words() -> None:
    long_text = " ".join(["word"] * 30)
    paragraphs = [long_text + "."]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert len(claims) == 1
    word_count = len(claims[0].text.split())
    assert word_count <= 26  # 25 words + possible "..."


def test_links_claim_to_citation_in_same_sentence() -> None:
    paragraphs = [
        "Transformers were introduced in 2017 (Vaswani et al., 2017).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.has_existing_citation is True
    assert len(claim.linked_citations) == 1
    assert claim.linked_citation is not None
    assert claim.linked_citation.authors == "Vaswani et al."
    assert claim.linked_citation.year == 2017


def test_links_multiple_citations_in_sentence() -> None:
    paragraphs = [
        "Prior work supports this (Smith, 2020; Jones, 2021).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.has_existing_citation is True
    assert len(claim.linked_citations) == 2
    assert claim.linked_citations[0].authors == "Smith"
    assert claim.linked_citations[1].authors == "Jones"


def test_no_citation_returns_empty_linked() -> None:
    paragraphs = [
        "Large language models can generate realistic but false citations.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.has_existing_citation is False
    assert claim.linked_citations == []
    assert claim.linked_citation is None
    assert claim.link_confidence is None


def test_link_confidence_sentence_final() -> None:
    paragraphs = [
        "Transformers were introduced in 2017 (Vaswani et al., 2017).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    assert claims[0].link_confidence == 100


def test_link_confidence_mid_sentence() -> None:
    paragraphs = [
        "As shown by Vaswani et al. (2017), transformers are effective.",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    assert len(claims) == 1
    assert claims[0].link_confidence == 80


def test_narrative_citation_links_correctly() -> None:
    paragraphs = [
        "Filipponi (2018) proposed a new index for evaluation.",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.has_existing_citation is True
    assert claim.linked_citation.authors == "Filipponi"
    assert claim.linked_citation.year == 2018


def test_paragraph_final_citation_upgrades_confidence() -> None:
    paragraphs = [
        "Transformers were introduced in 2017. "
        "They changed NLP forever (Vaswani et al., 2017).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    assert len(claims) == 2
    last = claims[-1]
    assert last.has_existing_citation is True
    assert last.link_confidence == 100


def test_ambiguous_citation_link_warning() -> None:
    paragraphs = [
        "Transformers changed NLP (Smith, 2020). "
        "Attention is all you need (Smith, 2020).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    assert len(claims) >= 2
    warnings = [w for c in claims for w in c.warnings if "ambiguous_citation_link" in w]
    assert len(warnings) >= 1


def test_one_citation_links_to_multiple_claims() -> None:
    paragraphs = [
        "Dropout improves generalization. "
        "It also reduces overfitting (Smith, 2020).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    cited = [c for c in claims if c.has_existing_citation]
    assert len(cited) >= 1


def test_warnings_field_defaults_empty() -> None:
    paragraphs = [
        "Large language models can generate realistic but false citations.",
    ]
    claims = extract_claims(paragraphs, [], bibliography_start=None)
    assert claims[0].warnings == []


def test_paragraph_final_multi_paragraph() -> None:
    paragraphs = [
        "Transformers were introduced in 2017 (Vaswani et al., 2017).",
        "Social media use has increased among adolescents. "
        "Higher usage was associated with depressive symptoms (Smith et al., 2023).",
    ]
    citations = extract_citations(paragraphs)
    claims = extract_claims(paragraphs, citations, bibliography_start=None)

    para0_claims = [c for c in claims if c.paragraph_index == 0]
    para1_claims = [c for c in claims if c.paragraph_index == 1]

    assert len(para0_claims) >= 1
    assert len(para1_claims) >= 1

    assert para0_claims[0].link_confidence == 100

    last_para1 = para1_claims[-1]
    assert last_para1.has_existing_citation is True
    assert last_para1.link_confidence == 100
