"""extraction_diag categorization + report. No LLM needed (synthetic diag dicts)."""

from collections import Counter

from turnstyle.extraction_diag import ExtractionReport, _categorize


def test_exception():
    cat, ev = _categorize("p", None, "(A)", {"error": "RuntimeError: boom"})
    assert cat == "exception"
    assert "boom" in ev


def test_empty_entities():
    cat, _ = _categorize("p", None, "(A)", {"entities": []})
    assert cat == "empty_entities"


def test_unparseable_json():
    diag = {"entities": ["x"],
            "sentence_extractions": [{"response": "not json at all", "parsed": False}]}
    cat, _ = _categorize("p", None, "(A)", diag)
    assert cat == "unparseable_json"


def test_hallucinated_entity():
    diag = {"entities": ["motorcycle"],
            "sentence_extractions": [
                {"response": '{"subj": "tiger", "pred": "gt", "obj": "motorcycle"}', "parsed": True}]}
    cat, ev = _categorize("the motorcycle is fast", None, "(A)", diag)
    assert cat == "hallucinated_entity"
    assert "tiger" in ev


def test_normalization():
    # 'bus' is in the prompt (not a hallucination) but matches no extracted entity
    diag = {"entities": ["car"],
            "sentence_extractions": [
                {"response": '{"subj": "car", "pred": "gt", "obj": "bus"}', "parsed": True}]}
    cat, _ = _categorize("the car is faster than the bus", None, "(A)", diag)
    assert cat == "normalization"


def test_aggregation_no_answer():
    diag = {"entities": ["car", "bus"],
            "sentence_extractions": [
                {"response": '{"subj": "car", "pred": "gt", "obj": "bus"}', "parsed": True}]}
    cat, _ = _categorize("the car is faster than the bus", None, "(A)", diag)
    assert cat == "aggregation_no_answer"


def test_wrong_answer():
    diag = {"entities": ["car", "bus"],
            "sentence_extractions": [
                {"response": '{"subj": "car", "pred": "gt", "obj": "bus"}', "parsed": True}]}
    cat, _ = _categorize("the car is faster than the bus", "(B)", "(A)", diag)
    assert cat == "wrong_answer"


def test_report_summary():
    rep = ExtractionReport(
        task="t", n=10, correct=4,
        categories=Counter({"empty_entities": 4, "hallucinated_entity": 2}),
        failures=[],
    )
    assert rep.accuracy == 0.4
    assert rep.top_category() == "empty_entities"
    s = rep.summary()
    assert "4/10" in s
    assert "fix first: empty_entities" in s
