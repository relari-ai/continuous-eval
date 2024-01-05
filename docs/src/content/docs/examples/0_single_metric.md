---
title: Single Metric
---

In this example, we show how to calculate a metric on a sinle data point.

```python
from continuous_eval import Dataset
from continuous_eval.metrics import PrecisionRecallF1

# A dataset is just a list of dictionaries containing the relevant information
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
metric = PrecisionRecallF1()

# Let's calculate the metric for the first datum
print(metric.calculate(**dataset.datum(0)))  # alternatively `metric.calculate(**q)`
```