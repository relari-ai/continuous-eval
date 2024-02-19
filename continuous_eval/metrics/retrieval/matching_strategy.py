from abc import ABC, abstractmethod
from enum import Enum, auto

from rouge import Rouge

_DEFAULT_ROUGE_CHUNK_MATCH_THRESHOLD = 0.7
_DEFAULT_ROUGE_SENTENCE_MATCH_THRESHOLD = 0.8


class MatchingStrategyType(Enum):
    CHUNK_MATCH = auto()
    SENTENCE_MATCH = auto()


class MatchingStrategy(ABC):
    @property
    @abstractmethod
    def type(self):
        pass

    @abstractmethod
    def is_relevant(self, retrieved_component, ground_truth_component):
        pass


class ExactChunkMatch(MatchingStrategy):
    @property
    def type(self):
        return MatchingStrategyType.CHUNK_MATCH

    def is_relevant(self, retrieved_component, ground_truth_component):
        return retrieved_component == ground_truth_component


class ExactSentenceMatch(MatchingStrategy):
    @property
    def type(self):
        return MatchingStrategyType.SENTENCE_MATCH

    def is_relevant(self, retrieved_component, ground_truth_component):
        return retrieved_component == ground_truth_component


class RougeChunkMatch(MatchingStrategy):
    def __init__(self, threshold=_DEFAULT_ROUGE_CHUNK_MATCH_THRESHOLD) -> None:
        super().__init__()
        self.threshold = threshold

    @property
    def type(self):
        return MatchingStrategyType.CHUNK_MATCH

    def is_relevant(self, retrieved_component, ground_truth_component):
        return Rouge().get_scores(retrieved_component, ground_truth_component)[0]["rouge-l"]["r"] > self.threshold


class RougeSentenceMatch(MatchingStrategy):
    def __init__(self, threshold=_DEFAULT_ROUGE_SENTENCE_MATCH_THRESHOLD) -> None:
        super().__init__()
        self.threshold = threshold

    @property
    def type(self):
        return MatchingStrategyType.SENTENCE_MATCH

    def is_relevant(self, retrieved_component, ground_truth_component):
        return Rouge().get_scores(retrieved_component, ground_truth_component)[0]["rouge-l"]["r"] >= self.threshold
