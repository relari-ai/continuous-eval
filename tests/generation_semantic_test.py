import pytest

# Try to import pytorch; skip tests if the import fails
required_module = pytest.importorskip("torch")

from continuous_eval.metrics.generation.text.semantic import (
    BertAnswerRelevance,
    BertAnswerSimilarity,
    BertSimilarity,
    DebertaAnswerScores,
)
from tests.helpers.utils import list_of_dicts_to_dict_of_lists


def test_bert_similarity_mean():
    data = [
        {"prediction": "This is a test", "reference": "This is a test"},
        {"prediction": "This is cat", "reference": "A cat is on the table"},
    ]

    metric = BertSimilarity()
    x = metric.batch(**list_of_dicts_to_dict_of_lists(data))
    assert x["bert_similarity"][0] > x["bert_similarity"][1]

    y = metric("The pen is on the table", "This book is red")
    assert y["bert_similarity"] > 0 and y["bert_similarity"] < 1


def test_bert_similarity_mean_pooler_output():
    data = [
        {"prediction": "This is a test", "reference": "This is a test"},
        {"prediction": "This is cat", "reference": "A cat is on the table"},
    ]

    metric = BertSimilarity(pooler_output=True)
    x = metric.batch(**list_of_dicts_to_dict_of_lists(data))
    assert x["bert_similarity"][0] > x["bert_similarity"][1]

    y = metric("The pen is on the table", "This book is red")
    assert y["bert_similarity"] > 0 and y["bert_similarity"] < 1


def test_answer_relevance():
    data = [
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
    x = metric.batch(**list_of_dicts_to_dict_of_lists(data))
    assert all(z["bert_answer_relevance"] > 0 and z["bert_answer_relevance"] < 1 for z in x)


def test_answer_similarity():
    dataset = [
        {
            "answer": "Samuel Adams",
            "ground_truth_answers": ["Douglas Adams"],
        },
        {
            "answer": "The number 42",
            "ground_truth_answers": ["The number 42", "42"],
        },
    ]
    metric = BertAnswerSimilarity()
    x = metric.batch(**list_of_dicts_to_dict_of_lists(dataset))
    y = metric(**dataset[1])
    assert abs(x[1]["bert_answer_similarity"] - y["bert_answer_similarity"]) < 1e-1


def test_deberta_answer_scores():
    data = [
        {
            "answer": "Samuel Adams",
            "ground_truth_answers": ["Douglas Adams"],
        },
        {
            "answer": "The number 42",
            "ground_truth_answers": ["The number 42", "42"],
        },
    ]
    metric = DebertaAnswerScores()
    x = metric.batch(**list_of_dicts_to_dict_of_lists(data))
    y = metric(**data[0])
    assert abs(x[0]["deberta_answer_entailment"] - y["deberta_answer_entailment"]) < 1e-5
    assert abs(x[0]["deberta_answer_contradiction"] - y["deberta_answer_contradiction"]) < 1e-5
