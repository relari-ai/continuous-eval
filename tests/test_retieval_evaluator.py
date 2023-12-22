import pytest

from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import (
    MatchingStrategy,
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)
from tests import example_datum
from tests.utils import all_close


def test_retieval_evaluator():
    data = [
        example_datum.CAPITAL_OF_FRANCE,
        example_datum.IMPLICATIONS_GLOBAL_WARMING,
    ]
    expected_results = {
        "precision": 0.045454545454545456,
        "recall": 0.25,
        "f1": 0.07692307692307693,
        "Average Precision": 0.0,
        "Mean Reciprocal Rank": 0.0,
        "NDCG": 0.0,
    }

    evaluator = RetrievalEvaluator(
        [
            PrecisionRecallF1(MatchingStrategy.ROUGE_SENTENCE_MATCH),
            RankedRetrievalMetrics(MatchingStrategy.ROUGE_CHUNK_MATCH),
        ]
    )
    results = evaluator.run(data, aggregate=True)
    assert all_close(results, expected_results)
