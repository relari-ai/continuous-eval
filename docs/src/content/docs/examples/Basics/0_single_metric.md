---
title: Single Metric
---

In this example, we show how to calculate a metric on a single data point.

```python
from continuous_eval.metrics.retrieval import PrecisionRecallF1

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

metric = PrecisionRecallF1()

print(metric(**datum))
```