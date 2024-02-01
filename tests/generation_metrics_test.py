import pytest

from continuous_eval.metrics import (
    DeterministicAnswerCorrectness,
    DeterministicFaithfulness,
    FleschKincaidReadability,
    LLMBasedAnswerCorrectness,
    LLMBasedAnswerRelevance,
    LLMBasedFaithfulness,
    LLMBasedStyleConsistency,
)
from tests.helpers import example_datum
from tests.helpers.utils import all_close


def test_deterministic_answer_relevance():
    data = [example_datum.ROMEO_AND_JULIET, example_datum.IMPLICATIONS_GLOBAL_WARMING]
    expected_results = [
        {
            "rouge_l_recall": 1.0,
            "rouge_l_precision": 1.0,
            "rouge_l_f1": 0.999999995,
            "token_overlap_recall": 1.0,
            "token_overlap_precision": 1.0,
            "token_overlap_f1": 1.0,
            "bleu_score": 1.0,
        },
        {
            "rouge_l_recall": 0.75,
            "rouge_l_precision": 0.375,
            "rouge_l_f1": 0.49999999555555563,
            "token_overlap_recall": 1.0,
            "token_overlap_precision": 0.5714285714285714,
            "token_overlap_f1": 0.7272727272727273,
            "bleu_score": 0.4734525552325106,
        },
    ]
    metric = DeterministicAnswerCorrectness()
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))


def test_rouge_sentence_faithfulness():
    data = [example_datum.CAPITAL_OF_FRANCE]
    expected_results = [
        {
            "rouge_faithfulness": 1.0,
            "token_overlap_faithfulness": 1.0,
            "bleu_faithfulness": 3.3720152341391845e-06,
            "rouge_p_by_sentence": [1.0],
            "token_overlap_p_by_sentence": [1.0],
            "bleu_score_by_sentence": [3.3720152341391845e-06],
        },
    ]

    metric = DeterministicFaithfulness()
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))


def test_llm_based_faithfulness():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]

    metric = LLMBasedFaithfulness()
    results = [metric.calculate(**datum) for datum in data]
    for result in results:
        assert isinstance(result["LLM_based_faithfulness"], bool)


def test_llm_based_answer_correctness():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]

    metric = LLMBasedAnswerCorrectness()
    results = [metric.calculate(**datum) for datum in data]
    for result in results:
        assert 0.0 <= result["LLM_based_answer_correctness"] <= 1.0


def test_llm_based_answer_relevance():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]

    metric = LLMBasedAnswerRelevance()
    results = [metric.calculate(**datum) for datum in data]
    for result in results:
        assert 0.0 <= result["LLM_based_answer_relevance"] <= 1.0


def test_llm_based_style_consistency():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]

    metric = LLMBasedStyleConsistency()
    results = [metric.calculate(**datum) for datum in data]
    for result in results:
        assert 0.0 <= result["LLM_based_style_consistency"] <= 1.0


def test_flesch_kincaid():
    expected = {
        "flesch_reading_ease": 116.14500000000001,
        "flesch_kincaid_grade_level": -1.4499999999999993,
    }
    result = FleschKincaidReadability().calculate(answer="The cat sat on the mat.")
    assert all_close(result, expected)
