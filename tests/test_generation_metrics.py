import pytest

from continuous_eval.metrics import (
    BertAnswerRelevance,
    BertAnswerSimilarity,
    DebertaAnswerScores,
    DeterministicAnswerRelevance,
    DeterministicFaithfulness,
    LLMBasedAnswerCorrectness,
    LLMBasedFaithfulness,
)
from tests import example_datum
from tests.utils import all_close


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
    data = [example_datum.CAPITAL_OF_FRANCE]
    expected_results = [
        {
            "rouge_faithfulness": 1.0,
            "token_overlap_faithfulness": 1.0,
            "avg_sentence_bleu": 0.0,
            "min_sentence_bleu": 0.0,
            "rouge_scores_p_by_sentence": [1.0],
            "token_overlap_p_by_sentence": [1.0],
        },
    ]

    metric = DeterministicFaithfulness()
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


def test_semantic():
    correct = {
        "question": "What are the implications of global warming?",
        "answer": "Reducing greenhouse gas emissions, transitioning to renewable energy",
        "ground_truths": ["Reducing greenhouse gas emissions"],
    }
    wrong = {
        "question": "What are the implications of global warming?",
        "answer": "The diverse culinary traditions of Italy offer a fascinating insight into the country's history",
        "ground_truths": ["Reducing greenhouse gas emissions"],
    }
    d1 = {
        "AnswerScore": DebertaAnswerScores().calculate(**correct),
        "AnswerSimilarity": BertAnswerSimilarity().calculate(**correct),
        "AnswerRelevance": BertAnswerRelevance().calculate(**correct),
    }
    d2 = {
        "AnswerScore": DebertaAnswerScores().calculate(**wrong),
        "AnswerSimilarity": BertAnswerSimilarity().calculate(**wrong),
    }
    assert (
        d1["AnswerScore"]["deberta_answer_entailment"]
        > d2["AnswerScore"]["deberta_answer_entailment"]
    )
    assert (
        d1["AnswerSimilarity"]["bert_answer_similarity"]
        > d2["AnswerSimilarity"]["bert_answer_similarity"]
    )
    assert (
        d1["AnswerRelevance"]["bert_answer_relevance"] > 0
        and d1["AnswerRelevance"]["bert_answer_relevance"] < 1.0
    )
