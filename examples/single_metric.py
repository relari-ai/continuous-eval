from continuous_eval import Dataset
from continuous_eval.metrics import PrecisionRecallF1, RougeChunkMatch

# Let's create a dataset
q = {
    "question": "What is the capital of France?",
    "retrieved_contexts": [
        "Paris is the capital of France and its largest city.",
        "Lyon is a major city in France.",
    ],
    "ground_truth_contexts": ["Paris is the capital of France."],
    "answer": "Paris",
    "ground_truths": ["Paris"],
}
dataset = Dataset([q])

# Let's initialize the metric
metric = PrecisionRecallF1(RougeChunkMatch())

# Let's calculate the metric for the first datum
print(metric.calculate(**dataset.datum(0)))  # alternatively `metric.calculate(**q)`
