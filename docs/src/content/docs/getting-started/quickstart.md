---
title: Quick Start
description: Quick Start
---

## Installation

```bash
python3 -m pip install continuous-eval
```

## Run a metric

```python
from continuous_eval.metrics import PrecisionRecallF1, RougeChunkMatch

datum = {
    "question": "What is the capital of France?",
    "retrieved_contexts": [
        "Paris is the capital of France and its largest city.",
        "Lyon is a major city in France.",
    ],
    "ground_truth_contexts": ["Paris is the capital of France."],
    "answer": "Paris",
    "ground_truths": ["Paris"],
}

metric = PrecisionRecallF1(RougeChunkMatch())
print(metric.calculate(**datum))
```

## Run eval on a dataset

**Load a Golden Dataset**

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
evaluator.run(k=2, batch_size=1)
# Peaking at the results
print(evaluator.aggregated_results)
# Saving the results for future use
evaluator.save("retrieval_evaluator_results.jsonl")
```

For generation you can instead use the `GenerationEvaluator`.

### Curate an eval dataset for your application

**We recommend AI teams invest in manually curating a high-quality golden dataset** (created by users, or domain experts) to properly evaluate and improve the LLM pipeline.
Every (RAG-based) LLM application is different in functionalities and requirements, and the evaluation golden dataset should be diverse enough to capture different design requirements.

If you don't have a golden dataset, you can use `SimpleDatasetGenerator` to create a silver dataset as a starting point, upon which you can modify and improve.

```python
from continuous_eval.simple_dataset_generator import SimpleDatasetGenerator

dataset = SimpleDatasetGenerator(VectorStoreIndex, num_questions=10)
```