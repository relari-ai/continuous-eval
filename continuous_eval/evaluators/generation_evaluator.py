from collections import ChainMap
from typing import List, Union

import pandas as pd
from tqdm import tqdm

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

    def run(self, batch_size: int = 32, quiet: bool = False):
        batches = self._get_batches(batch_size=batch_size)
        results = {id(metric): list() for metric in self.metrics}

        pbar = tqdm(total=len(self.dataset), desc="Processing", disable=quiet)
        for batch in batches:
            for metric in self.metrics:
                results[id(metric)].extend(metric.batch_calculate(batch))
            pbar.update(len(batch))
        pbar.close()

        # metrics = {id(metric): metric.batch_calculate(data) for metric in self.metrics}
        self._results = [dict(ChainMap(*x)) for x in zip(*results.values())]
        return self._results
