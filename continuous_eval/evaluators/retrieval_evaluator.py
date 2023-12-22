from typing import List

from tqdm import tqdm

from continuous_eval.evaluators.base_evaluator import BaseEvaluator
from continuous_eval.metrics import MatchingStrategy, Metric, PrecisionRecallF1
from continuous_eval.dataset import Dataset


class RetrievalEvaluator(BaseEvaluator):
    def __init__(
        self,
        dataset: Dataset,
        metrics: List[Metric] = [PrecisionRecallF1(MatchingStrategy.EXACT_CHUNK_MATCH)],
    ):
        super().__init__(dataset=dataset, metrics=metrics)

    def run(
        self,
        k: int = None,
    ):
        if k is not None and k < 0:
            print(
                "K must be a positive integer. "
                "Leave it as None to consider all retrieved chunks."
            )

        self._results = self._calculate_metrics(k)

    def _calculate_metrics(self, k):
        results = []
        for _, row in tqdm(
            self.dataset.iterrows(),
            total=len(self.dataset),
            desc="Examples evaluated",
        ):
            datum = row.to_dict()
            if k is not None:
                datum = datum.copy()
                datum["retrieved_contexts"] = datum["retrieved_contexts"][:k]

            result = {}
            for metric in self.metrics:
                try:
                    result.update(metric.calculate(**datum))
                except Exception as e:
                    print(f"Error calculating {metric}. Skipping...")
                    print(e)
                    continue
            results.append(result)
        return results
