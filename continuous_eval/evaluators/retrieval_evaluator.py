from collections import ChainMap
from typing import List, Optional, Union

from tqdm import tqdm

from continuous_eval.dataset import Dataset
from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.metrics import Metric


class RetrievalEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset: Dataset,
        metrics: List[Metric],
    ):
        super().__init__(dataset=dataset, metrics=metrics)

    def run(
        self,
        k: int = None,
        batch_size: Optional[Union[int, float]] = 32,
        quiet: bool = False,
    ):
        assert k is None or isinstance(k, int) and k > 0, "K must be a positive integer or None."

        batches = self._get_batches(batch_size=batch_size)
        results = {id(metric): list() for metric in self.metrics}

        pbar = tqdm(total=len(self.dataset), desc="Processing", disable=quiet)
        for batch in batches:
            batch = self._preprocess_batch(batch, k)
            for metric in self.metrics:
                results[id(metric)].extend(metric.batch_calculate(batch))
            pbar.update(len(batch))
        pbar.close()

        self._results = [dict(ChainMap(*x)) for x in zip(*results.values())]
        return self._results

    def _preprocess_batch(self, batch, k):
        if k is None:
            return batch
        if k is not None:
            # Filer out the retrieved contexts
            batch = batch.copy()
            for datum in batch:
                if len(datum["retrieved_contexts"]) >= k:
                    datum["retrieved_contexts"] = datum["retrieved_contexts"][:k]
        return batch
