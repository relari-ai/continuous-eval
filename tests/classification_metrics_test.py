from continuous_eval.metrics.classification import SingleLabelClassification
from tests.helpers.utils import all_close, validate_metric_metadata


def test_numeric():
    y_pred = [0, 2, 1, 3]
    y_true = [0, 1, 2, 3]
    classes = 4

    expected = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    metric = SingleLabelClassification(classes=classes)
    results = [metric(y, y_gt) for y, y_gt in zip(y_pred, y_true)]
    validate_metric_metadata(metric, results)
    agg = metric.aggregate(results)
    assert all_close(agg, expected)


def test_string_class():
    y_pred = ["0", "2", "1", "3"]
    y_true = ["0", "1", "2", "3"]
    classes = {"0", "1", "2", "3"}

    expected = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    metric = SingleLabelClassification(classes=classes)
    results = [metric(y, y_gt) for y, y_gt in zip(y_pred, y_true)]
    validate_metric_metadata(metric, results)
    agg = metric.aggregate(results)
    assert all_close(agg, expected)


def test_probability_scores():
    y_pred = [
        [0.6, 0.1, 0.2, 0.1],  # 0
        [0.1, 0.3, 0.5, 0.1],  # 2
        [0.2, 0.5, 0.2, 0.1],  # 1
        [0.0, 0.05, 0.05, 0.9],  # 3
    ]
    y_true = [0, 1, 2, 3]
    classes = 4

    expected = {
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    metric = SingleLabelClassification(classes=classes)
    results = [metric(y, y_gt) for y, y_gt in zip(y_pred, y_true)]
    validate_metric_metadata(metric, results)
    agg = metric.aggregate(results)
    assert all_close(agg, expected)
