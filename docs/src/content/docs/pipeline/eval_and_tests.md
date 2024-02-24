---
title: Evaluators and Tests
sidebar:
  badge:
    text: new
    variant: tip
---

## Definitions

You can optionally add `eval` and `tests` to the modules you want to measure the performance of.


#### `eval`: select relevant evaluation metrics
- Select the metrics and specify the input according to the data fields required for each metric. `MetricName().use(data_fields)`.
- Metric inputs can be referenced using items from two sources:
    -   **From `dataset`**: e.g. `ground_truth_context = dataset.ground_truth_context`
    -   **From current module**: e.g. `answer = ModuleOutput()`
    -   **From prior modules**: e.g. `retrieved_context = ModuleOutput(DocumentsContent, module=reranker)`, where
        `DocumentsContent = ModuleOutput(lambda x: [z["page_content"] for z in x])` to select specific items from the prior module's output


#### `tests`: define specific performance criteria
- Select testing class `GreaterOrEqualThan` or `MeanGreaterOrEqualThan` to run test over each datapoint or the mean of the aggregate dataset
- Define `test_name`, `metric_name` (must be part of the metric_name that `eval` calculates), and `min_value`.


## Example

We will expand the example defined in pipeline with metrics and tests.

Evaluation Metrics:
-   `PrecisionRecallF1` to evaluate the Retriever
-   `RankedRetrievalMetrics` to evaluate the Reranker
-   `FleschKincaidReadability`, `DebertaAnswerScores`, and `LLMBasedFaithfulness` to evaluate the Generator

Tests:
-   Mean of `context_recall`, a metric calculated by `PrecisionRecallF1`, needs to be >= 0.8 to pass
-   Mean of `average_precision`, a metric calculated by `RankedRetrievalMetrics`, needs to be >= 0.7 to pass
-   Mean of `deberta_entailment`, a metric calculated by `DebertaAnswerScores` needs to be >= 0.5 to pass


```d2
direction: right
Dataset: Eval Dataset
Dataset.shape: oval
Dataset -> Retriever
Retriever -> Reranker -> Generator
```




```python title="pipeline.py"
from continuous_eval.eval import Module, Pipeline, Dataset, ModuleOutput
from continuous_eval.metrics.retrieval import PrecisionRecallF1, RankedRetrievalMetrics # Deterministic metrics
from continuous_eval.metrics.generation.text import (
    FleschKincaidReadability, # Deterministic metric
    DebertaAnswerScores, # Semantic metric
    LLMBasedFaithfulness, # LLM-based metric
)
from typing import List, Dict
from continuous_eval.eval.tests import MeanGreaterOrEqualThan

dataset = Dataset("data/eval_golden_dataset")

Documents = List[Dict[str, str]]
DocumentsContent = ModuleOutput(lambda x: [z["page_content"] for z in x])

retriever = Module(
    name="retriever",
    input=dataset.question,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Average Precision", metric_name="context_recall", min_value=0.8
        ),
    ],
)

reranker = Module(
    name="reranker",
    input=retriever,
    output=Documents,
    eval=[
        RankedRetrievalMetrics().use(
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Context Recall", metric_name="average_precision", min_value=0.7
        ),
    ],
)

llm = Module(
    name="llm",
    input=reranker,
    output=str,
    eval=[
        FleschKincaidReadability().use(answer=ModuleOutput()),
        DebertaAnswerScores().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truths
        ),
        LLMBasedFaithfulness().use(
            answer=ModuleOutput(),
            retrieved_context=ModuleOutput(DocumentsContent, module=reranker),
            question=dataset.question,
        ),
    ],
    tests=[
        MeanGreaterOrEqualThan(
            test_name="Deberta Entailment", metric_name="deberta_answer_entailment", min_value=0.5
        ),
    ],
)

pipeline = Pipeline([retriever, reranker, llm], dataset=dataset)

print(pipeline.graph_repr())
```