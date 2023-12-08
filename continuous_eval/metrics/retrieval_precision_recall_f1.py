from nltk.tokenize import sent_tokenize

from continuous_eval.metrics.base import Metric
from continuous_eval.metrics.retrieval_matching_strategy import (
    MatchingStrategy,
    is_relevant,
)


class PrecisionRecallF1(Metric):
    def __init__(self, matching_strategy: MatchingStrategy) -> None:
        super().__init__()
        self.matching_strategy = matching_strategy
        assert self.matching_strategy in MatchingStrategy

    def calculate(self, retrieved_contexts, ground_truth_contexts, **kwargs):
        # Calculate precision, recall and f1 based on different matching strategies.
        # These metrics do not consider the order or rank of relevant information in the retrieval.

        if self.matching_strategy in (
            MatchingStrategy.EXACT_CHUNK_MATCH,
            MatchingStrategy.ROUGE_CHUNK_MATCH,
        ):
            relevant_chunks = sum(
                [
                    is_relevant(chunk, ground_truth_chunk, self.matching_strategy)
                    for ground_truth_chunk in ground_truth_contexts
                    for chunk in retrieved_contexts
                ]
            )
            precision = (
                relevant_chunks / len(retrieved_contexts) if retrieved_contexts else 0
            )
            recall = (
                relevant_chunks / len(ground_truth_contexts)
                if ground_truth_contexts
                else 0
            )
        elif self.matching_strategy in (
            MatchingStrategy.EXACT_SENTENCE_MATCH,
            MatchingStrategy.ROUGE_SENTENCE_MATCH,
        ):
            retrieval_sentences = [
                sentence
                for chunk in retrieved_contexts
                for sentence in sent_tokenize(chunk)
            ]
            ground_truth_sentences = [
                sentence
                for chunk in ground_truth_contexts
                for sentence in sent_tokenize(chunk)
            ]
            relevant_sentences = sum(
                [
                    is_relevant(retrieved_sentence, gt_sentence, self.matching_strategy)
                    for gt_sentence in ground_truth_sentences
                    for retrieved_sentence in retrieval_sentences
                ]
            )
            precision = (
                relevant_sentences / len(retrieval_sentences)
                if retrieval_sentences
                else 0.0
            )
            recall = (
                relevant_sentences / len(ground_truth_sentences)
                if ground_truth_sentences
                else 0.0
            )

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
