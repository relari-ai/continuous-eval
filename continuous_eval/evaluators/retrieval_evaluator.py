from collections import ChainMap
from typing import List

from continuous_eval.dataset import Dataset
from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.metrics import MatchingStrategy, Metric, PrecisionRecallF1


class RetrievalEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset: Dataset,
        metrics: List[Metric] = [PrecisionRecallF1(MatchingStrategy.EXACT_CHUNK_MATCH)],
    ):
        super().__init__(dataset=dataset, metrics=metrics)

    def run(self, k: int = None):
        if k is not None and k < 0:
            print("K must be a positive integer. " "Leave it as None to consider all retrieved chunks.")

        data = self._preprocess_dataset(k)
        metrics = {id(metric): metric.batch_calculate(data) for metric in self.metrics}
        self._results = [dict(ChainMap(*x)) for x in zip(*metrics.values())]
        return self._results

    def _preprocess_dataset(self, k):
        dataset_ = self.dataset.to_dict(orient="records")
        if k is not None:
            # Filer out the retrieved contexts
            dataset_ = dataset_.copy()
            for datum in dataset_:
                if len(datum["retrieved_contexts"]) >= k:
                    datum["retrieved_contexts"] = datum["retrieved_contexts"][:k]
        return dataset_
