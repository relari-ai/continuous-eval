from typing import Any, Dict, List, Literal, Set, Union

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from continuous_eval.metrics.base import Metric


class SingleLabelClassification(Metric):
    def __init__(
        self,
        classes: Union[int, Set[str]],
        average: Literal["micro", "macro", "weighted"] = "macro",
    ):
        super().__init__()
        assert average in ["macro", "micro", "weighted"]
        self._average = average
        if isinstance(classes, int):
            # Assume classes are 0, 1, 2, ..., classes-1
            self._classes = list(range(classes))
        else:
            self._classes = classes if isinstance(classes, list) else set(classes)
            if len(self._classes) != len(classes):
                raise ValueError("Classes must be unique")
        # Create a mapping from class labels to integers
        self._class_to_index = {label: index for index, label in enumerate(self._classes)}

    def __call__(
        self,
        predicted_class: Union[str, int, List[float]],
        ground_truth_class: Union[str, int],
    ):
        if isinstance(predicted_class, list):
            predicted_class = np.argmax(predicted_class).item()  # Convert to int
        return {
            "classification_prediction": predicted_class,
            "classification_ground_truth": ground_truth_class,
            "classification_correct": predicted_class == ground_truth_class,
        }

    def aggregate(self, results: List[Dict[str, Union[str, int]]]) -> Any:
        pred = [self._class_to_index[r["classification_prediction"]] for r in results]
        gt = [self._class_to_index[r["classification_ground_truth"]] for r in results]
        return {
            "accuracy": accuracy_score(gt, pred),
            "balanced_accuracy": balanced_accuracy_score(gt, pred),
            "precision": precision_score(gt, pred, average=self._average, zero_division=1.0),  # type: ignore
            "recall": recall_score(gt, pred, average=self._average, zero_division=1.0),  # type: ignore
            "f1": f1_score(gt, pred, average=self._average, zero_division=1.0),  # type: ignore
        }
