import pytest

from continuous_eval.metrics.retrieval import (
    ContextCoverage,
    ContextPrecision,
    ExactSentenceMatch,
    PrecisionRecallF1,
    RankedRetrievalMetrics,
    RougeChunkMatch,
    RougeSentenceMatch,
    TokenCount,
)
from tests.helpers import example_datum
from tests.helpers.utils import all_close, validate_metric_metadata


def test_precision_recall_exact_chunk_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"context_precision": 0.0, "context_recall": 0.0, "context_f1": 0.0},
        {"context_precision": 1.0, "context_recall": 1.0, "context_f1": 1.0},
    ]
    metric = PrecisionRecallF1(RougeChunkMatch(threshold=0.7))
    results = [metric(**datum) for datum in data]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(metric(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )  # type: ignore


def test_precision_recall_exact_sentence_match():
    data = [example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"context_precision": 1.0, "context_recall": 1.0, "context_f1": 1.0}
    ]
    metric = PrecisionRecallF1(RougeSentenceMatch(threshold=0.8))
    results = [metric(**datum) for datum in data]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(metric(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )  # type: ignore


def test_precision_recall_rouge_sentence_match():
    data = [
        example_datum.CAPITAL_OF_FRANCE,
        example_datum.IMPLICATIONS_GLOBAL_WARMING,
    ]
    expected_results = [
        {"context_precision": 0.0, "context_recall": 0.0, "context_f1": 0.0},
        {
            "context_precision": 0.09090909090909091,
            "context_recall": 0.5,
            "context_f1": 0.15384615384615385,
        },
    ]
    metric = PrecisionRecallF1(RougeSentenceMatch())
    results = [metric(**datum) for datum in data]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(metric(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )  # type: ignore


def test_ranked_retrieval_exact_chunk_match():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"average_precision": 0, "reciprocal_rank": 0, "ndcg": 0.0},
        {"average_precision": 1.0, "reciprocal_rank": 1.0, "ndcg": 1.0},
    ]
    metric = RankedRetrievalMetrics(RougeChunkMatch())
    results = [metric(**datum) for datum in data]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(metric(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )  # type: ignore


def test_ranked_retrieval_exact_sentence_match():
    with pytest.raises(AssertionError):
        RankedRetrievalMetrics(ExactSentenceMatch())


def test_context_precision():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    metric = ContextPrecision()
    results = [metric(**datum) for datum in data]
    validate_metric_metadata(metric, results)


def test_context_coverage():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    metric = ContextCoverage(model="openai:gpt-4o")
    results = [metric(**datum) for datum in data]
    validate_metric_metadata(metric, results)


def test_token_count():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    metric = TokenCount("o200k_base")
    expected = [17, 16]
    assert (
        result := [metric(**datum)["num_tokens"] for datum in data]
    ) == expected, result
    validate_metric_metadata(metric)
    expected = [17, 18]
    metric = TokenCount("approx")
    assert (
        result := [metric(**datum)["num_tokens"] for datum in data]
    ) == expected, result
