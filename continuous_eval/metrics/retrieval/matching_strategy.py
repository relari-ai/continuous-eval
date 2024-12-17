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

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class ExactSentenceMatch(MatchingStrategy):
    @property
    def type(self):
        return MatchingStrategyType.SENTENCE_MATCH

    def is_relevant(self, retrieved_component, ground_truth_component):
        return retrieved_component == ground_truth_component

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class RougeChunkMatch(MatchingStrategy):
    _rouge = Rouge()
    # def __new__(cls, *args, **kwargs):
    #     # Always initialize _rouge during object creation
    #     instance = super().__new__(cls)
    #     instance._rouge = Rouge()  # type: ignore
    #     return instance

    def __init__(self, threshold=_DEFAULT_ROUGE_CHUNK_MATCH_THRESHOLD) -> None:
        super().__init__()
        self.threshold = threshold

    @property
    def type(self):
        return MatchingStrategyType.CHUNK_MATCH

    def is_relevant(self, retrieved_component, ground_truth_component):
        try:
            score = RougeChunkMatch._rouge.get_scores(
                retrieved_component, ground_truth_component, ignore_empty=True
            )
        except Exception:
            return False
        try:
            return score[0]["rouge-l"]["r"] >= self.threshold
        except Exception:
            return False

    def __getstate__(self):
        return {"threshold": self.threshold}

    def __setstate__(self, state):
        self.threshold = state["threshold"]


class RougeSentenceMatch(MatchingStrategy):
    _rouge = Rouge()
    # def __new__(cls, *args, **kwargs):
    #     # Always initialize _rouge during object creation
    #     instance = super().__new__(cls)
    #     instance._rouge = Rouge()  # type: ignore
    #     return instance

    def __init__(
        self, threshold=_DEFAULT_ROUGE_SENTENCE_MATCH_THRESHOLD
    ) -> None:
        super().__init__()
        self.threshold = threshold

    @property
    def type(self):
        return MatchingStrategyType.SENTENCE_MATCH

    def is_relevant(self, retrieved_component, ground_truth_component):
        try:
            score = RougeSentenceMatch._rouge.get_scores(
                retrieved_component, ground_truth_component, ignore_empty=True
            )
        except Exception:
            return False
        try:
            return score[0]["rouge-l"]["r"] >= self.threshold
        except Exception:
            return False

    def __getstate__(self):
        return {"threshold": self.threshold}

    def __setstate__(self, state):
        self.threshold = state["threshold"]
