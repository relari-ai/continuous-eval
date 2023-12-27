import tempfile

import pytest

from continuous_eval.dataset import Dataset


def test_dataset_from_jsonl():
    expected_cols = {'question', 'answer', 'ground_truths', 'dataset', 'model', 'annotation'}
    dataset = Dataset.from_jsonl('tests/data/correctness_sm.jsonl')
    assert set(dataset.columns) == expected_cols
    assert len(dataset) == 10


def test_dataset_invalid():
    with pytest.raises(ValueError):
        # Must have requirements on columns
        Dataset.from_jsonl('tests/data/invalid.jsonl')


def test_save_load():
    dataset = Dataset.from_jsonl('tests/data/correctness_sm.jsonl')

    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = f'{tmpdirname}/dataset.jsonl'
        dataset.to_jsonl(fname)
        loaded = Dataset.from_jsonl(fname)
        assert set(loaded.columns) == set(dataset.columns)
        assert len(loaded) == len(dataset)
