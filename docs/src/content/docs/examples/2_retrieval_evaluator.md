---
title: Retrieval Evaluator
---

In this example, we show how to run generation evaluation over an example dataset.

```python
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import PrecisionRecallF1, RankedRetrievalMetrics

# Let's download the retrieval dataset example
dataset = example_data_downloader("retrieval")

# Setup the evaluator
evaluator = RetrievalEvaluator(
    dataset=dataset,
    metrics=[
        PrecisionRecallF1(),
        RankedRetrievalMetrics(),
    ],
)

# Run the eval!
evaluator.run(k=2, batch_size=1)

# Peaking at the results
print(evaluator.aggregated_results)

# Saving the results for future use
evaluator.save("retrieval_evaluator_results.jsonl")
```