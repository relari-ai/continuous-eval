import tempfile

import pandas as pd
import pytest

from continuous_eval.dataset import Dataset
from continuous_eval.evaluators import GenerationEvaluator, RetrievalEvaluator
from continuous_eval.metrics import DeterministicAnswerRelevance
from tests.helpers.dummy_metric import DummyMetric


def test_retieval_evaluator():
    expected_keys = {"precision", "NDCG", "recall"}

    dataset = Dataset.from_jsonl("tests/data/retrieval_sm.jsonl")
    evaluator = RetrievalEvaluator(
        dataset=dataset,
        metrics=[
            DummyMetric({"precision", "recall"}),
            DummyMetric({"NDCG"}),
        ],
    )
    evaluator.run(k=2)
    assert len(evaluator.results) == len(dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys

    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f"{tmpdirname}/results.jsonl"
        evaluator.save(fname)
        loaded = pd.read_json(fname, lines=True)
        assert set(loaded.columns) == expected_keys
        assert len(loaded) == len(evaluator.results)


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

    dataset = Dataset.from_jsonl("tests/data/correctness_sm.jsonl")
    evaluator = GenerationEvaluator(
        dataset=dataset,
        metrics=[
            DeterministicAnswerRelevance(),
            DummyMetric({"dummy_correctness"}),
        ],
    )
    evaluator.run()
    assert len(evaluator.results) == len(dataset)
    assert set(evaluator.aggregated_results.keys()) == expected_keys

    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f"{tmpdirname}/results.jsonl"
        evaluator.save(fname)
        loaded = pd.read_json(fname, lines=True)
        assert set(loaded.columns) == expected_keys
        assert len(loaded) == len(evaluator.results)
