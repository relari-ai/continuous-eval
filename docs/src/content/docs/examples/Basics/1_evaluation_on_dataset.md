---
title: Run evaluation over a dataset
---

In this example, we show how to run evaluation over a dataset.

```python
from pathlib import Path

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import Dataset, EvaluationRunner, SingleModulePipeline
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics

# Let's download the retrieval dataset example
dataset_jsonl = example_data_downloader("retrieval") # 300 samples in the retrieval dataset
dataset = Dataset(dataset_jsonl)

pipeline = SingleModulePipeline(
    dataset=dataset,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=dataset.retrieved_contexts,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
        RankedRetrievalMetrics().use(
            retrieved_context=dataset.retrieved_contexts,
            ground_truth_context=dataset.ground_truth_contexts,
        ),
    ],
)

# We start the evaluation runner and run the metrics over the downloaded dataset
evalrunner = EvaluationRunner(pipeline)
metrics = evalrunner.evaluate(dataset)
print(metrics.aggregate())
```
