import tempfile

import pandas as pd
import pytest

from continuous_eval.dataset import Dataset
from continuous_eval.evaluators import GenerationEvaluator, RetrievalEvaluator
from continuous_eval.metrics import DeterministicAnswerCorrectness
from tests.helpers.dummy_metric import DummyMetric

retrieval_dataset = Dataset.from_jsonl("tests/data/retrieval_sm.jsonl")
generation_dataset = Dataset.from_jsonl("tests/data/correctness_sm.jsonl")


def test_retieval_evaluator():
    expected_keys = {"precision", "NDCG", "recall"}

    evaluator = RetrievalEvaluator(
        dataset=retrieval_dataset,
        metrics=[
            DummyMetric({"precision", "recall"}),
            DummyMetric({"NDCG"}),
        ],
    )
    evaluator.run(k=2, quiet=True)
    assert len(evaluator.results) == len(retrieval_dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys

    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f"{tmpdirname}/results.jsonl"
        evaluator.save(fname)
        loaded = pd.read_json(fname, lines=True)
        assert set(loaded.columns) == expected_keys
        assert len(loaded) == len(evaluator.results)


def test_retieval_evaluator_int_batch_size():
    expected_keys = {"precision", "NDCG", "recall"}

    evaluator = RetrievalEvaluator(
        dataset=retrieval_dataset,
        metrics=[
            DummyMetric({"precision", "recall"}),
            DummyMetric({"NDCG"}),
        ],
    )
    evaluator.run(k=2, batch_size=3, quiet=True)
    assert len(evaluator.results) == len(retrieval_dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys


def test_retieval_evaluator_float_batch_size():
    expected_keys = {"precision", "NDCG", "recall"}

    evaluator = RetrievalEvaluator(
        dataset=retrieval_dataset,
        metrics=[
            DummyMetric({"precision", "recall"}),
            DummyMetric({"NDCG"}),
        ],
    )
    evaluator.run(k=2, batch_size=0.25, quiet=True)
    assert len(evaluator.results) == len(retrieval_dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys


def test_generation_evaluator():
    expected_keys = {
        "rouge_l_f1",
        "token_f1",
        "bleu_score",
        "rouge_l_precision",
        "rouge_l_recall",
        "token_recall",
        "token_precision",
        "dummy_correctness",
    }

    evaluator = GenerationEvaluator(
        dataset=generation_dataset,
        metrics=[
            DeterministicAnswerCorrectness(),
            DummyMetric({"dummy_correctness"}),
        ],
    )
    evaluator.run(quiet=True)
    assert len(evaluator.results) == len(generation_dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys

    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f"{tmpdirname}/results.jsonl"
        evaluator.save(fname)
        loaded = pd.read_json(fname, lines=True)
        assert set(loaded.columns) == expected_keys
        assert len(loaded) == len(evaluator.results)


def test_generation_evaluator_int_batch_size():
    expected_keys = {
        "rouge_l_f1",
        "token_f1",
        "bleu_score",
        "rouge_l_precision",
        "rouge_l_recall",
        "token_recall",
        "token_precision",
        "dummy_correctness",
    }

    evaluator = GenerationEvaluator(
        dataset=generation_dataset,
        metrics=[
            DeterministicAnswerCorrectness(),
            DummyMetric({"dummy_correctness"}),
        ],
    )
    evaluator.run(batch_size=3, quiet=True)
    assert len(evaluator.results) == len(generation_dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys


def test_generation_evaluator_float_batch_size():
    expected_keys = {
        "rouge_l_f1",
        "token_f1",
        "bleu_score",
        "rouge_l_precision",
        "rouge_l_recall",
        "token_recall",
        "token_precision",
        "dummy_correctness",
    }

    evaluator = GenerationEvaluator(
        dataset=generation_dataset,
        metrics=[
            DeterministicAnswerCorrectness(),
            DummyMetric({"dummy_correctness"}),
        ],
    )
    evaluator.run(batch_size=0.25, quiet=True)
    assert len(evaluator.results) == len(generation_dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys
