import json
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Union

import pandas as pd

from continuous_eval.dataset import Dataset
from continuous_eval.metrics.base import Metric


def _required_args(fn):
    return set(fn.__code__.co_varnames[1 : fn.__code__.co_argcount])


class BaseEvaluator(ABC):
    def __init__(self, dataset: Union[Dataset, pd.DataFrame], metrics: List[Metric]):
        if not isinstance(dataset, (Dataset, pd.DataFrame)):
            raise ValueError("dataset must be a Dataset or DataFrame object")
        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list of Metric objects")
        if not all([isinstance(metric, Metric) for metric in metrics]):
            raise ValueError("metrics must be a list of Metric objects")
        if not metrics:
            raise ValueError("At least one metric must be provided")
        self.metrics = metrics
        self.dataset = dataset
        self._results = list()
        self._validate_metrics()

    @property
    def results(self):
        if not self._results:
            raise ValueError("No results found. Did you run the evaluator?")
        return self._results

    @cached_property
    def aggregated_results(self):
        if not self._results:
            raise ValueError("No results found. Did you run the evaluator?")
        agg = pd.DataFrame(BaseEvaluator._sanitize_pre_aggregate(self._results))
        return agg.mean().to_dict()

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def save(self, savepath: str):
        def _sanitize(v):
            if isinstance(v, list):
                return [_sanitize(item) for item in v]
            return v

        if not self._results:
            raise ValueError("No results found. Did you run the evaluator?")
        with open(savepath, "w") as f:
            for item in self._results:
                f.write(json.dumps({k: _sanitize(v) for k, v in item.items()}) + "\n")

    @staticmethod
    def _sanitize_pre_aggregate(results):
        return [{k: v for k, v in r.items() if not isinstance(v, (list, str))} for r in results]

    def _validate_metrics(self):
        cols = set(self.dataset.columns)
        for metric in self.metrics:
            required = _required_args(metric.calculate)
            if not required.issubset(cols):
                raise ValueError(
                    f"Metric {metric.__class__.__name__} requires {required} " f"but only {cols} are present."
                )
