import pytest

from continuous_eval.metrics import (
    ExactSentenceMatch,
    LLMBasedContextCoverage,
    LLMBasedContextPrecision,
    PrecisionRecallF1,
    RankedRetrievalMetrics,
    RougeChunkMatch,
    RougeSentenceMatch,
)
from tests.helpers import example_datum
from tests.helpers.utils import all_close, in_zero_one, is_close


def test_precision_recall_exact_chunk_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        {"precision": 1.0, "recall": 1.0, "f1": 1.0},
    ]

    metric = PrecisionRecallF1(RougeChunkMatch(threshold=0.7))
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))


def test_precision_recall_exact_sentence_match():
    data = [example_datum.ROMEO_AND_JULIET]
    expected_results = [{"precision": 1.0, "recall": 1.0, "f1": 1.0}]

    metric = PrecisionRecallF1(RougeSentenceMatch(threshold=0.8))
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))


def test_precision_recall_rouge_sentence_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]
    expected_results = [
        {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        {"precision": 0.09090909090909091, "recall": 0.5, "f1": 0.15384615384615385},
    ]

    metric = PrecisionRecallF1(RougeSentenceMatch())
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))


def test_ranked_retrieval_exact_chunk_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"Average Precision": 0, "Mean Reciprocal Rank": 0, "NDCG": 0.0},
        {"Average Precision": 1.0, "Mean Reciprocal Rank": 1.0, "NDCG": 1.0},
    ]

    metric = RankedRetrievalMetrics(RougeChunkMatch())
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))


def test_ranked_retrieval_exact_sentence_match():
    with pytest.raises(AssertionError):
        RankedRetrievalMetrics(ExactSentenceMatch())


def test_llm_based_context_precision():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    metric = LLMBasedContextPrecision()
    assert all(in_zero_one(metric.calculate(**datum)) for datum in data)


def test_llm_based_context_coverage_openai():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]

    metric = LLMBasedContextCoverage(model="gpt-3.5-turbo-1106")
    assert all(in_zero_one(metric.calculate(**datum)["LLM_based_context_coverage"]) for datum in data)


def test_llm_based_context_coverage_claude():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    metric = LLMBasedContextCoverage(model="claude-2.1")
    assert all(in_zero_one(metric.calculate(**datum)["LLM_based_context_coverage"]) for datum in data)


def test_llm_based_context_coverage_gemini():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]

    metric = LLMBasedContextCoverage(model="gemini-pro")
    assert all(in_zero_one(metric.calculate(**datum)["LLM_based_context_coverage"]) for datum in data)
