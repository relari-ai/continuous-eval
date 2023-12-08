import pytest

from tests.utils import all_close
from tests import example_datum
from continuous_eval.metrics import (
    DeterministicAnswerRelevance,
    RougeSentenceFaithfulness,
    BertAnswerRelevance,
    BertAnswerSimilarity,
    LLMBasedFaithfulness,
    LLMBasedAnswerCorrectness,
)


def test_deterministic_answer_relevance():
    data = [example_datum.ROMEO_AND_JULIET, example_datum.IMPLICATIONS_GLOBAL_WARMING]
    expected_results = [
        {
            "rouge_l_f1": 0.999999995,
            "token_f1": 1.0,
            "rouge_l_recall": 1.0,
            "token_precision": 1.0,
            "token_recall": 1.0,
            "rouge_l_precision": 1.0,
            "bleu_score": 1.0,
        },
        {
            "rouge_l_f1": 0.49999999555555563,
            "token_f1": 0.6153846153846153,
            "token_recall": 1.0,
            "rouge_l_recall": 0.75,
            "bleu_score": 0.4734525552325106,
            "token_precision": 0.4444444444444444,
            "rouge_l_precision": 0.375,
        },
    ]

    metric = DeterministicAnswerRelevance()
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_rouge_sentence_faithfulness():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"rouge_sentence_faithfulness": 0.5},
        {"rouge_sentence_faithfulness": 0.0},
    ]

    metric = RougeSentenceFaithfulness()
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_bert_answer_relevance():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"bert_answer_relevance": 0.3805941939353943},
        {"bert_answer_relevance": 0.48182404041290283},
    ]

    metric = BertAnswerRelevance()
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_bert_answer_similarity():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.ROMEO_AND_JULIET]
    expected_results = [
        {"bert_answer_similarity": 1.0},
        {"bert_answer_similarity": 1.0},
    ]

    metric = BertAnswerSimilarity()
    assert all(
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(data, expected_results)
    )


def test_llm_based_faithfulness():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]

    metric = LLMBasedFaithfulness()
    results = [metric.calculate(**datum) for datum in data]
    for result in results:
        assert isinstance(result["LLM_based_faithfulness_score"], bool)


def test_llm_based_answer_correctness():
    data = [example_datum.CAPITAL_OF_FRANCE, example_datum.IMPLICATIONS_GLOBAL_WARMING]

    metric = LLMBasedAnswerCorrectness()
    results = [metric.calculate(**datum) for datum in data]
    for result in results:
        assert 1.0 <= result["LLM_based_answer_correctness"] <= 5.0
