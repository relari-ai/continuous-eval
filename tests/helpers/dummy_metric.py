from random import random
from typing import Set

from continuous_eval.metrics.base import Metric


class DummyMetric(Metric):
    def __init__(self, result_keys: Set[str]):
        self._result_keys = result_keys

    def calculate(self, **kwargs):
        return {k: random() for k in self._result_keys}
