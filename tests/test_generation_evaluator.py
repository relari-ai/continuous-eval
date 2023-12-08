import pytest
from continuous_eval.metrics import (
    DeterministicAnswerRelevance,
    RougeSentenceFaithfulness,
)
from continuous_eval.evaluators import GenerationEvaluator
from tests.utils import all_close
from tests import example_datum


def test_generation_evaluator():
    data = [
        example_datum.CAPITAL_OF_FRANCE,
        example_datum.IMPLICATIONS_GLOBAL_WARMING,
    ]
    expected_results = {
        "token_f1": 0.8076923076923077,
        "token_precision": 0.7222222222222222,
        "rouge_l_precision": 0.6875,
        "bleu_score": 0.7367262776162553,
        "rouge_l_recall": 0.875,
        "rouge_l_f1": 0.7499999952777778,
        "token_recall": 1.0,
        "rouge_sentence_faithfulness": 0.25,
    }

    evaluator = GenerationEvaluator(
        [
            DeterministicAnswerRelevance(),
            RougeSentenceFaithfulness(),
        ]
    )
    results = evaluator.run(data, aggregate=True)
    assert all_close(results, expected_results)
