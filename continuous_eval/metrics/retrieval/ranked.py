from math import log

from continuous_eval.metrics.base import Metric
from continuous_eval.metrics.retrieval.matching_strategy import MatchingStrategy, MatchingStrategyType, RougeChunkMatch


class RankedRetrievalMetrics(Metric):
    def __init__(self, matching_strategy: MatchingStrategy = RougeChunkMatch()) -> None:
        super().__init__()
        self.matching_strategy = matching_strategy
        assert isinstance(
            matching_strategy, MatchingStrategy
        ), "Matching strategy must be an instance of MatchingStrategy."
        assert (
            self.matching_strategy.type == MatchingStrategyType.CHUNK_MATCH
        ), "Ranked metrics are calculated at chunk level."

    def __call__(self, retrieved_context, ground_truth_context, **kwargs):
        # Calculate ranked metrics (MAP, MRR, NDCG) based on different matching strategies.
        map = self.calculate_average_precision(retrieved_context, ground_truth_context)
        mrr = self.calculate_reciprocal_rank(retrieved_context, ground_truth_context)
        ndcg = self.calculate_normalized_discounted_cumulative_gain(retrieved_context, ground_truth_context)
        return {"average_precision": map, "reciprocal_rank": mrr, "ndcg": ndcg}

    def calculate_average_precision(self, retrieved_context, ground_truth_context):
        # Calculate average precision for a single query retrieval

        # Calculate average precision for each relevant chunk
        average_precision = 0
        relevant_chunks = 0

        for i, chunk in enumerate(retrieved_context):
            for ground_truth_chunk in ground_truth_context:
                if self.matching_strategy.is_relevant(chunk, ground_truth_chunk):
                    relevant_chunks += 1
                    average_precision += relevant_chunks / (i + 1)
                    break

        return average_precision / relevant_chunks if relevant_chunks else 0

    def calculate_reciprocal_rank(self, retrieved_context, ground_truth_context):
        # Calculate reciprocal rank for a single query retrieval

        # Calculate reciprocal rank for each relevant chunk
        for i, chunk in enumerate(retrieved_context):
            for ground_truth_chunk in ground_truth_context:
                if self.matching_strategy.is_relevant(chunk, ground_truth_chunk):
                    return 1 / (i + 1)

        # If no relevant chunk is found, return 0
        return 0

    def calculate_normalized_discounted_cumulative_gain(self, retrieved_context, ground_truth_context):
        # Calculate normalized discounted cumulative gain for a single query retrieval

        # Calculate discounted cumulative gain
        dcg = 0
        matched_ground_truths = set()

        for i, chunk in enumerate(retrieved_context):
            for ground_truth_chunk in ground_truth_context:
                if ground_truth_chunk not in matched_ground_truths and self.matching_strategy.is_relevant(
                    chunk, ground_truth_chunk
                ):
                    # Calculate relevance score (relevant gain = 1)
                    dcg += 1 / log(i + 2, 2)
                    matched_ground_truths.add(ground_truth_chunk)
                    break

        # Calculate ideal discounted cumulative gain
        idcg = 0
        for i in range(len(ground_truth_context)):
            idcg += 1 / log(i + 2, 2)

        return dcg / idcg
