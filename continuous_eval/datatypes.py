from enum import Enum


class DatumField(Enum):
    """
    Enum class for data fields.
    """

    QUESTION = "question"  # user question
    RETRIEVED_CONTEXTS = "retrieved_contexts"
    GROUND_TRUTH_CONTEXTS = "ground_truth_contexts"
    ANSWER = "answer"  # generated answer
    GROUND_TRUTH_ANSWER = "ground_truths"  # ground truth answers
