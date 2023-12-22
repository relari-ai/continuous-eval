import pytest

from continuous_eval.metrics import (
    LLMBasedContextCoverage,
    LLMBasedContextPrecision,
    MatchingStrategy,
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)
from tests import example_datum
from tests.utils import all_close, in_zero_one, is_close


def test_precision_recall_exact_chunk_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        {"precision": 0.5, "recall": 0.5, "f1": 0.5},
    ]

    metric = PrecisionRecallF1(MatchingStrategy.EXACT_CHUNK_MATCH)
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_precision_recall_exact_sentence_match():
    data = [example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }
    ]

    metric = PrecisionRecallF1(MatchingStrategy.EXACT_SENTENCE_MATCH)
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_precision_recall_rouge_sentence_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]
    expected_results = [
        {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        {"precision": 0.09090909090909091, "recall": 0.5, "f1": 0.15384615384615385},
    ]

    metric = PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH)
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_ranked_retrieval_exact_chunk_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"Average Precision": 0, "Mean Reciprocal Rank": 0, "NDCG": 0.0},
        {
            "Average Precision": 1.0,
            "Mean Reciprocal Rank": 1.0,
            "NDCG": 0.6131471927654585,
        },
    ]

    metric = RankedRetrievalMetrics(MatchingStrategy.EXACT_CHUNK_MATCH)
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_ranked_retrieval_exact_sentence_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"Average Precision": 0, "Mean Reciprocal Rank": 0, "NDCG": 0.0},
        {
            "Average Precision": 1.0,
            "Mean Reciprocal Rank": 1.0,
            "NDCG": 0.6131471927654585,
        },
    ]

    metric = RankedRetrievalMetrics(MatchingStrategy.EXACT_CHUNK_MATCH)
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_llm_based_context_precision():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    metric = LLMBasedContextPrecision()
    assert all(in_zero_one(metric.calculate(**datum)) for datum in data)


def test_llm_based_context_coverage():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [0.0, 1.0]

    metric = LLMBasedContextCoverage()
    assert all(
        is_close(metric.calculate(**datum)["LLM_based_context_coverage"], expected)
        for datum, expected in zip(data, expected_results)
    )
