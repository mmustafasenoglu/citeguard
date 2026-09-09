from citeguard.benchmarks import (
    GroundTruthPassage,
    PredictedPassage,
    bootstrap_document_ci,
    document_split,
    parse_pan_xml,
    score_passages,
)


def test_pan_xml_and_span_scoring(tmp_path) -> None:
    xml = tmp_path / "suspicious.xml"
    xml.write_text(
        '<document reference="suspicious.txt"><feature name="artificial" '
        'this_offset="10" this_length="20" source_reference="source.txt" '
        'source_offset="3" source_length="20" obfuscation="none"/></document>',
        encoding="utf-8",
    )
    truth = parse_pan_xml(str(xml))
    assert truth[0].suspicious_document == "suspicious.txt"
    prediction = PredictedPassage("suspicious.txt", 12, 18, "source.txt", 4, 18, 0.9)
    result = score_passages(truth, [prediction])
    assert result["precision"] == result["recall"] == result["f1"] == 1.0


def test_span_scoring_is_one_to_one_and_handles_zero_denominators() -> None:
    truth = [GroundTruthPassage("a", 0, 10, "b", 0, 10)]
    predictions = [PredictedPassage("a", 0, 10, "b", 0, 10, 1.0)] * 2
    result = score_passages(truth, predictions)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert score_passages([], [])["precision"] == 1.0


def test_split_and_bootstrap_are_deterministic() -> None:
    assert document_split(["b", "a", "c"], holdout_fraction=1 / 3) == document_split(
        ["c", "a", "b"], holdout_fraction=1 / 3
    )
    rows = {"a": {"precision": 1.0, "recall": 0.5, "f1": 2 / 3}}
    assert bootstrap_document_ci(rows, samples=10)["precision"] == (1.0, 1.0)
