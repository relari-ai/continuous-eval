from enum import Enum, auto

from rouge import Rouge

ROUGE_CHUNK_MATCH_THRESHOLD = 0.7
ROUGE_SENTENCE_MATCH_THRESHOLD = 0.8


class MatchingStrategy(Enum):
    EXACT_CHUNK_MATCH = auto()
    EXACT_SENTENCE_MATCH = auto()
    ROUGE_CHUNK_MATCH = auto()
    ROUGE_SENTENCE_MATCH = auto()


def is_relevant(retrieved_component, ground_truth_component, matching_strategy):
    if (
        matching_strategy == MatchingStrategy.EXACT_CHUNK_MATCH
        or matching_strategy == MatchingStrategy.EXACT_SENTENCE_MATCH
    ):
        return retrieved_component == ground_truth_component
    elif matching_strategy == MatchingStrategy.ROUGE_CHUNK_MATCH:
        return (
            Rouge().get_scores(retrieved_component, ground_truth_component)[0]["rouge-l"]["r"]
            > ROUGE_CHUNK_MATCH_THRESHOLD
        )
    elif matching_strategy == MatchingStrategy.ROUGE_SENTENCE_MATCH:
        return (
            Rouge().get_scores(retrieved_component, ground_truth_component)[0]["rouge-l"]["r"]
            > ROUGE_SENTENCE_MATCH_THRESHOLD
        )
    else:
        raise ValueError(f"Unknown matching strategy {matching_strategy} .")
