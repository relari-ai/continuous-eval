from enum import Enum, auto

from rouge import Rouge


class MatchingStrategy(Enum):
    EXACT_CHUNK_MATCH = auto()
    EXACT_SENTENCE_MATCH = auto()
    ROUGE_CHUNK_MATCH = auto()
    ROUGE_SENTENCE_MATCH = auto()


def is_relevant(retrieved_component, ground_truth_component, matching_strategy):
    match matching_strategy:
        case MatchingStrategy.EXACT_CHUNK_MATCH:
            return retrieved_component == ground_truth_component
        case MatchingStrategy.EXACT_SENTENCE_MATCH:
            return retrieved_component == ground_truth_component
        case MatchingStrategy.ROUGE_CHUNK_MATCH:
            rouge = Rouge()
            score = rouge.get_scores(retrieved_component, ground_truth_component)[0][
                "rouge-l"
            ]["r"]
            return score > 0.7
        case MatchingStrategy.ROUGE_SENTENCE_MATCH:
            rouge = Rouge()
            score = rouge.get_scores(retrieved_component, ground_truth_component)[0][
                "rouge-l"
            ]["r"]
            return score > 0.8
        case _:
            raise ValueError(f"Matching strategy {matching_strategy} not found.")
