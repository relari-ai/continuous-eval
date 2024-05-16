from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


class XYData(tuple):
    def __new__(cls, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(y.shape) != 1:
            raise ValueError("y must be a 1-dimensional array")
        return super().__new__(cls, (X, y))

    def __len__(self) -> int:
        return len(self.X)

    @property
    def X(self):
        return self[0]

    @property
    def y(self):
        return self[1]


@dataclass
class SplitRatios:
    train: float = 0.6
    test: float = 0.2
    calibration: float = 0.2

    def __post_init__(self):
        assert np.abs(self.train + self.test + self.calibration - 1) < 1e-6, "Data split must sum to 1"


class DataSplit:
    def __init__(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series, Iterable],
        split_ratios: SplitRatios,
        features: Optional[List[str]] = None,
        oversample: bool = False,
        random_state: Optional[int] = None,
    ):
        if isinstance(y, Iterable):
            y = np.array(list(y))
        elif isinstance(y, pd.Series):
            y = y.to_numpy()

        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame"
        assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
        assert len(X) == len(y), "X and y must have the same number of rows"
        assert len(X.columns) > 0, "X must not be empty"
        assert len(y.shape) == 1, "y must be a 1-dimensional array"

        self.features = features
        if self.features is None:
            # If no features are provided, assume all numeric columns are features
            self.features = X.select_dtypes(include=np.number).columns.tolist()

        # Split the data into training, testing and calibration sets
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=split_ratios.test, random_state=random_state)
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_temp,
            y_temp,
            test_size=split_ratios.calibration / (1 - split_ratios.test),
            random_state=random_state,
        )

        self.X_train = X_train[self.features].astype(float)
        self.y_train = y_train.astype(int)
        # Oversample the training set (if needed)
        if oversample:
            self.X_train_unbalanced = self.X_train.copy()
            self.y_train_unbalanced = self.y_train.copy()
            self.X_train, self.y_train = SMOTE().fit_resample(self.X_train, self.y_train)  # type: ignore

        self.X_cal = X_cal[self.features]
        self.y_cal = y_cal.astype(int)

        self.X_test = X_test[self.features]
        self.y_test = y_test.astype(int)

    @property
    def train(self) -> XYData:
        return XYData(self.X_train, self.y_train)

    @property
    def test(self) -> XYData:
        return XYData(self.X_test, self.y_test)

    @property
    def calibration(self) -> XYData:
        return XYData(self.X_cal, self.y_cal)


class DatumField(Enum):
    """
    Enum class for data fields.
    """

    QUESTION = "question"  # user question
    retrieved_context = "retrieved_context"
    ground_truth_context = "ground_truth_context"
    ANSWER = "answer"  # generated answer
    GROUND_TRUTH_ANSWER = "ground_truths"  # ground truth answers
