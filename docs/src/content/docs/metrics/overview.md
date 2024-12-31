---
title: Overview of Metrics
description: Overview of different types of metrics
sidebar:
  badge:
    text: beta
    variant: tip
---

## Metric Categories

The `continuous-eval` package offers three categories of metrics based on how they are computed:

- **Deterministic metrics:** calculated based on statistical formulas
- **Semantic:** calculated using smaller models
- **Probabilistic:** calculated by an Evaluation LLM with curated prompts

All the metrics comes with pros and cons and there's not a one-size-fits-all evaluation pipeline that's optimal for every use case. We aim to provide a wide range of metrics for you to choose from.

## Using a metric

There are two ways to use a metric: Directly or through a pipeline.

### 1. Directly

Each metric has a `__call__` method that takes a dictionary of data and returns a dictionary of results.

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
    "ground_truth_answers": ["Paris"],
}

metric = PrecisionRecallF1()

print(metric(**datum))
```

Additionally, each metric has a `args`, `schema` and `help` properties that describe the metric.
The property `args` is a dictionary of arguments that can be passed to the metric

```text
>> print(metric.args)
{
  'retrieved_context': Arg(type=typing.List[str], description='', is_required=True, default=None), 
  'ground_truth_context': Arg(type=typing.List[str], description='', is_required=True, default=None)
}
```

The property `schema` is a dictionary of arguments that can be passed to the metric

```text
>> print(metric.schema)
{
  {'context_precision': Field(type=<class 'float'>, limits=(0.0, 1.0), internal=False, description=None), 
  'context_recall': Field(type=<class 'float'>, limits=(0.0, 1.0), internal=False, description=None), 
  'context_f1': Field(type=<class 'float'>, limits=(0.0, 1.0), internal=False, description=None)
}
```

And finally, the property `help` is a string that describes the metric

```text
>> print(metric.help)
"Calculate the precision, recall, and f1 score for the retrieved context given the ground truth context."
```

### 2. Through a pipeline

This example shows how to use a metric through a pipeline, which is the recommended way when you want to evaluate over a dataset.

```python
from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import EvaluationRunner, SingleModulePipeline
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics

if __name__ == "__main__":
    # Let's download the retrieval dataset example
    dataset = example_data_downloader("retrieval") 

    # Define the pipeline (system under test)
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

Note that it is important to place the code that uses the metric inside the `__main__` block, otherwise the multiprocessing evaluation will not work (and will fall back to the single-process evaluation).;

## More examples

You can find more examples in the [examples folder](https://github.com/relari-ai/continuous-eval/tree/main/examples) or in the [example repository](https://github.com/relari-ai/examples).
