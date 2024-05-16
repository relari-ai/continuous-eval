import pickle
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from continuous_eval.datatypes import XYData
from continuous_eval.utils.telemetry import telemetry


def _default_regressor(X: pd.DataFrame, y: pd.Series):
    classifier = LogisticRegression()
    parameters = {
        "penalty": ["l1", "l2"],
        "C": [0.1, 1, 10],
        "solver": ["liblinear"],
    }
    clf = GridSearchCV(classifier, parameters)
    clf.fit(X, y)
    return clf


class EnsembleMetric:
    def __init__(
        self,
        training: XYData,
        calibration: XYData,
        alpha: float = 0.1,
        random_state: Optional[int] = None,
        predictor_factory: Callable = _default_regressor,
    ) -> None:
        telemetry.log_metric_call(self.__class__.__name__)
        # fmt: off
        assert alpha > 0.0 and alpha < 1.0, "Alpha must be between 0 and 1"
        assert isinstance(training, XYData), "Training data must be an XYData object"
        assert isinstance(calibration, XYData), "Calibration data must be an XYData object"
        assert (len(training.X.columns) > 0) and (len(training) > 0), "Training data must not be empty"
        assert (len(calibration.X.columns) > 0) and (len(calibration) > 0), "Calibration data must not be empty"
        assert (set(training.X.columns) == set(calibration.X.columns)), "Training and calibration data must have the same features"
        # fmt: on
        self.features = training.X.columns
        self._regressor = predictor_factory(training.X, training.y)
        self._alpha = alpha
        self._classifier = MapieClassifier(
            estimator=self._regressor,  # type: ignore
            cv="prefit",
            method="lac",
            random_state=random_state,
        )
        self._classifier.fit(calibration.X, calibration.y)

    def predict(
        self, X: pd.DataFrame, judicator: Optional[Callable] = None, quiet=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        y_pred, y_set = self._classifier.predict(X, alpha=self._alpha)
        if judicator is None:
            return y_pred, y_set
        y_set = y_set.squeeze()
        y_hat = np.empty(len(y_set), dtype=int)
        iterator = range(len(y_set))
        if not quiet:
            from tqdm import tqdm

            iterator = tqdm(iterator)
        for i in iterator:
            if np.sum(y_set[i]) == 1:
                y_hat[i] = np.argmax(y_set[i])
            else:
                y_hat[i] = judicator(X.index[i])
                y_set[i] = np.zeros(len(y_set[i]), dtype=int)
                y_set[i][y_hat[i]] = 1
        return y_hat, y_set

    def save(self, savepath: str) -> None:
        pickle.dump(self, open(savepath, "wb"))

    @staticmethod
    def load(loadpath: str) -> "EnsembleMetric":
        return pickle.load(open(loadpath, "rb"))
