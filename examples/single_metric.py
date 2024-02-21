from continuous_eval import Dataset
from continuous_eval.metrics.retrieval import PrecisionRecallF1

# A dataset is just a list of dictionaries containing the relevant information
datum = {
    "question": "What is the capital of France?",
    "retrieved_context": [
        "Paris is the capital of France and its largest city.",
        "Lyon is a major city in France.",
    ],
    "ground_truth_context": ["Paris is the capital of France."],
    "answer": "Paris",
    "ground_truths": ["Paris"],
}

# Let's initialize the metric
metric = PrecisionRecallF1()

# Let's calculate the metric for the first datum
print(metric(**datum))
