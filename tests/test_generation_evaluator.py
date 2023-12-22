import pytest

from continuous_eval.evaluators import GenerationEvaluator
from continuous_eval.metrics import (
    DeterministicAnswerRelevance,
    DeterministicFaithfulness,
)
from tests import example_datum
from tests.utils import all_close


def test_generation_evaluator():
    data = [
        example_datum.CAPITAL_OF_FRANCE,
        example_datum.IMPLICATIONS_GLOBAL_WARMING,
    ]
    expected_results = {
        "rouge_l_recall": 0.875,
        "token_recall": 1.0,
        "token_precision": 0.7222222222222222,
        "rouge_l_precision": 0.6875,
        "token_f1": 0.8076923076923077,
        "rouge_l_f1": 0.7499999952777778,
        "bleu_score": 0.7367262776162553,
        "rouge_faithfulness": 0.5,
        "token_overlap_faithfulness": 1.0,
        "avg_sentence_bleu": 0.0,
        "min_sentence_bleu": 0.0,
    }

    evaluator = GenerationEvaluator(
        [
            DeterministicAnswerRelevance(),
            DeterministicFaithfulness(),
        ]
    )
    results = evaluator.run(data, aggregate=True)
    assert all_close(results, expected_results)
