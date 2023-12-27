import pytest

from continuous_eval.metrics.generation_semantic_metrics import (
    BertAnswerRelevance,
    BertAnswerSimilarity,
    BertSimilarity,
    DebertaAnswerScores,
)


def test_bert_similarity_mean():
    data = [
        {"prediction": "This is a test", "reference": "This is a test"},
        {"prediction": "This is cat", "reference": "A cat is on the table"},
    ]

    metric = BertSimilarity()
    x = metric.batch_calculate(data, pooler_output=False)
    assert x["bert_similarity"][0] > x["bert_similarity"][1]

    x = metric.calculate("The pen is on the table", "This book is red", pooler_output=False)
    assert x["bert_similarity"] > 0 and x["bert_similarity"] < 1


def test_bert_similarity_mean_pooler_output():
    data = [
        {"prediction": "This is a test", "reference": "This is a test"},
        {"prediction": "This is cat", "reference": "A cat is on the table"},
    ]

    metric = BertSimilarity()
    x = metric.batch_calculate(data, pooler_output=True)
    assert x["bert_similarity"][0] > x["bert_similarity"][1]

    x = metric.calculate("The pen is on the table", "This book is red", pooler_output=True)
    assert x["bert_similarity"] > 0 and x["bert_similarity"] < 1


def test_answer_relevance():
    dataset = [
        {
            "question": "Who wrote the 'The Hitchhiker's Guide'?",
            "answer": "Douglas Adams",
        },
        {
            "question": "Answer to the Ultimate Question of Life, the Universe, and Everything",
            "answer": "The number 42",
        },
    ]
    metric = BertAnswerRelevance()
    metric.batch_calculate(dataset)
    metric.calculate(**dataset[0])


def test_answer_similarity():
    dataset = [
        {
            "answer": "Samuel Adams",
            "ground_truths": ["Douglas Adams"],
        },
        {
            "answer": "The number 42",
            "ground_truths": ["The number 42", "42"],
        },
    ]
    metric = BertAnswerSimilarity()
    metric.batch_calculate(dataset)
    metric.calculate(**dataset[0])


def test_deberta_answer_scores():
    dataset = [
        {
            "answer": "Samuel Adams",
            "ground_truths": ["Douglas Adams"],
        },
        {
            "answer": "The number 42",
            "ground_truths": ["The number 42", "42"],
        },
    ]
    metric = DebertaAnswerScores()
    metric.batch_calculate(dataset)
    metric.calculate(**dataset[0])


def test_semantic_outputs():
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
    assert d1["AnswerScore"]["deberta_answer_entailment"] > d2["AnswerScore"]["deberta_answer_entailment"]
    assert d1["AnswerSimilarity"]["bert_answer_similarity"] > d2["AnswerSimilarity"]["bert_answer_similarity"]
    assert d1["AnswerRelevance"]["bert_answer_relevance"] > 0 and d1["AnswerRelevance"]["bert_answer_relevance"] < 1.0
