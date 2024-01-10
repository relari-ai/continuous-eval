import numpy as np
import pytest

from continuous_eval.classifiers import EnsembleMetric
from continuous_eval.classifiers.utils import eval_prediction
from continuous_eval.datatypes import DataSplit, SplitRatios
from tests.helpers.dummy_dataset import dummy_results

X, y = dummy_results(1000, ["a", "b", "c"], prob_detection=1, prob_false_alarm=0)


def test_data_split():
    datasplit = DataSplit(
        X=X,
        y=y,
        split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
        features=["a", "b"],
        oversample=False,
    )
    assert len(datasplit.train.X) == 600
    assert len(datasplit.train.y) == 600
    assert len(datasplit.test.X) == 200
    assert len(datasplit.test.y) == 200
    assert len(datasplit.calibration.X) == 200
    assert len(datasplit.calibration.y) == 200
    assert datasplit.train.X.columns.tolist() == ["a", "b"]
    assert datasplit.test.X.columns.tolist() == ["a", "b"]
    assert datasplit.calibration.X.columns.tolist() == ["a", "b"]


def test_data_split_oversample():
    datasplit = DataSplit(
        X=X,
        y=y,
        split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
        features=["a", "b"],
        oversample=True,
        random_state=42,
    )
    _, counts = np.unique(datasplit.train.y, return_counts=True)
    assert len(counts) == 2
    assert counts[0] == counts[1]


def test_ensemble_classifier():
    datasplit = DataSplit(
        X=X,
        y=y,
        split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
        features=["a", "b"],
        oversample=False,
        random_state=42,
    )
    clf = EnsembleMetric(training=datasplit.train, calibration=datasplit.calibration)
    y_hat, y_set = clf.predict(datasplit.test.X)
    num_undecided = np.sum(np.all(y_set, axis=1))
    performance = eval_prediction(datasplit.test.y, y_hat)
    assert performance["accuracy"] == 1.0
    assert performance["precision"] == 1.0
    assert performance["recall"] == 1.0
    assert performance["f1"] == 1.0
    assert num_undecided == 0


def test_ensemble_classifier_with_judicator():
    X, y = dummy_results(1000, ["a", "b", "c"], prob_detection=0.6, prob_false_alarm=0.3)
    datasplit = DataSplit(
        X=X,
        y=y,
        split_ratios=SplitRatios(train=0.6, test=0.2, calibration=0.2),
        features=["a", "b"],
        oversample=True,
        random_state=42,
    )
    clf = EnsembleMetric(
        training=datasplit.train,
        calibration=datasplit.calibration,
    )
    y_hat_no_judge, _ = clf.predict(datasplit.test.X)
    eval_no_judge = eval_prediction(datasplit.test.y, y_hat_no_judge)

    clf = EnsembleMetric(
        training=datasplit.train,
        calibration=datasplit.calibration,
    )
    y_hat_with_judge, _ = clf.predict(datasplit.test.X, judicator=lambda idx: datasplit.test.y[idx])
    eval_with_judge = eval_prediction(datasplit.test.y, y_hat_with_judge)

    assert eval_no_judge["f1"] <= eval_with_judge["accuracy"]
    assert eval_no_judge["accuracy"] <= eval_with_judge["accuracy"]
