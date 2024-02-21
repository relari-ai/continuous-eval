---
title: Quick Start
description: Quick Start
---

If you haven't installed continuous-eval, go [here](../installation/).

## Run a single metric

Import the metric of your choice ([see all metrics](../../metrics/overview/)) and get the results.

```python
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
```

## Run evalulation over a dataset

In the following code example, we load an example evaluation dataset `retrieval`, create an `RetrievalEvaluator`, and selected two metric groups `PrecisionRecallF1`, `RankedRetrievalMetrics`. 

The aggregated results are printed in the terminal and the results per datum is saved at `retrieval_evaluator_results.jsonl`.

```python
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.evaluators import RetrievalEvaluator
from continuous_eval.metrics import PrecisionRecallF1, RankedRetrievalMetrics

# Build a dataset: create a dataset from a list of dictionaries containing question/answer/context/etc.
# Or download one of the of the examples... 
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
evaluator.run()
# Peaking at the results
print(evaluator.aggregated_results)
# Saving the results for future use
evaluator.save("retrieval_evaluator_results.jsonl")
```

Learn more about the <a href="/dataset/dataset">dataset</a> class and the <a href="/dataset/evaluator">`Evaluator`</a> class.

## Curate the dataset

**We recommend AI teams invest in manually curating a high-quality golden dataset** (curated domain experts and checked against user data) to properly evaluate and improve the LLM pipeline. The evaluation golden dataset should be diverse enough to capture unique design requirements in each LLM pipeline.

**If you don't have a golden dataset, you can use `SimpleDatasetGenerator` to create a "silver dataset" as a starting point, upon which you can modify and improve.**

Checkout the guide to create a "silver dataset" using <a href="/dataset/simple_dataset_genetator">`SimpleDatasetGenerator`</a>.
