from collections import ChainMap
from typing import List, Union

import pandas as pd

from continuous_eval.dataset import Dataset
from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.metrics import DeterministicFaithfulness, Metric


class GenerationEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        metrics: List[Metric] = [DeterministicFaithfulness()],
    ):
        super().__init__(dataset=dataset, metrics=metrics)

    def run(self):
        data = self.dataset.to_dict(orient="records")
        metrics = {id(metric): metric.batch_calculate(data) for metric in self.metrics}
        self._results = [dict(ChainMap(*x)) for x in zip(*metrics.values())]
        return self._results
