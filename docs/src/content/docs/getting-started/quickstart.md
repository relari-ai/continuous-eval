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

In the following code example, we load an example evaluation dataset `retrieval`, create a pipeline with one module, and selected two metric groups `PrecisionRecallF1`, `RankedRetrievalMetrics`. 

The aggregated results are printed in the terminal and the results per datum is saved at `metrics_results_retr.jsonl`.

```python
from pathlib import Path

from continuous_eval.data_downloader import example_data_downloader
from continuous_eval.eval import Dataset, SingleModulePipeline
from continuous_eval.eval.manager import eval_manager
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics

# Let's download the retrieval dataset example
dataset_jsonl = example_data_downloader("retrieval")
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

# We start the evaluation manager and run the metrics
eval_manager.set_pipeline(pipeline)
eval_manager.evaluation.results = dataset.data
eval_manager.run_metrics()
eval_manager.metrics.save(Path("metrics_results_retr.json"))

print(eval_manager.metrics.aggregate())
```

## Curate a golden dataset

**We recommend AI teams invest in curating a high-quality golden dataset** (curated domain experts and checked against user data) to properly evaluate and improve the LLM pipeline. The evaluation golden dataset should be diverse enough to capture unique design requirements in each LLM pipeline.

**If you don't have a golden dataset, you can use `SimpleDatasetGenerator` to create a "silver dataset" as a starting point, upon which you can modify and improve.**

**Relari offers more custom synthetic dataset generation / augmentation as a service.** We have generated granular pipeline-level datasets for SEC Filing, Company Transcript, Coding Agents, Dynamic Tool Use, Enterprise Search, Sales Contracts, Company Wiki, Slack Conversation, Customer Support Tickets, Product Docs, etc. [Contact us](mailto:founders@relari.ai) if you are interested.