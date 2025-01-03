---
title: Quick Start
description: Quick Start
---

If you haven't installed continuous-eval, go [here](../../getting-started/installation/).

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

## Run evaluation over a dataset

In the following code example, we load an example evaluation dataset `retrieval`, create a pipeline with one module, and selected two metric groups `PrecisionRecallF1`, `RankedRetrievalMetrics`. 

The aggregated results are printed in the terminal.

```python
from time import perf_counter

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import Dataset, EvaluationRunner, SingleModulePipeline
from continuous_eval.eval.tests import GreaterOrEqualThan
from continuous_eval.metrics.retrieval import (
    PrecisionRecallF1,
    RankedRetrievalMetrics,
)


def main():
    # Let's download the retrieval dataset example
    dataset_jsonl = example_data_downloader("retrieval")
    dataset = Dataset(dataset_jsonl) 

    pipeline = SingleModulePipeline(
        dataset=dataset,
        eval=[
            PrecisionRecallF1().use(
                retrieved_context=dataset.retrieved_contexts,  # type: ignore
                ground_truth_context=dataset.ground_truth_contexts,  # type: ignore
            ),
            RankedRetrievalMetrics().use(
                retrieved_context=dataset.retrieved_contexts,  # type: ignore
                ground_truth_context=dataset.ground_truth_contexts,  # type: ignore
            ),
        ],
        tests=[
            GreaterOrEqualThan(
                test_name="Recall", metric_name="context_recall", min_value=0.8
            ),
        ],
    )

    # We start the evaluation manager and run the metrics
    tic = perf_counter()
    runner = EvaluationRunner(pipeline)
    eval_results = runner.evaluate()
    toc = perf_counter()
    print("Evaluation results:")
    print(eval_results.aggregate())
    print(f"Elapsed time: {toc - tic:.2f} seconds\n")

    print("Running tests...")
    test_results = runner.test(eval_results)
    print(test_results)


if __name__ == "__main__":
    # It is important to run this script in a new process to avoid
    # multiprocessing issues
    main()

```

Continuous-eval is designed to support multi-module evaluation. In this case we instead suppose the system is composed by one single module (the retriever) so we can use the `SingleModulePipeline` class to setup the pipeline.

In the pipeline we added both metrics (i.e., `PrecisionRecallF1` and `RankedRetrievalMetrics`) and tests (i.e., `GreaterOrEqualThan` on the recall metric). Read more about this in the [Metrics and Tests](../../pipeline/metrics_and_tests) page.

## Curate a golden dataset

**We recommend AI teams invest in curating a high-quality golden dataset** (curated domain experts and checked against user data) to properly evaluate and improve the LLM pipeline. The evaluation golden dataset should be diverse enough to capture unique design requirements in each LLM pipeline.

**Relari offers more custom synthetic dataset generation / augmentation as a service.** We have generated granular pipeline-level datasets for SEC Filing, Company Transcript, Coding Agents, Dynamic Tool Use, Enterprise Search, Sales Contracts, Company Wiki, Slack Conversation, Customer Support Tickets, Product Docs, etc. [Contact us](mailto:founders@relari.ai) if you are interested.