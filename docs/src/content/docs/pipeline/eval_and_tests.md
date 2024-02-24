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

Below is a full example of a two-step pipeline, where we added some metrics and tests to the pipeline.

Evaluation Metrics:
-   `PrecisionRecallF1` is added to evaluate the Retriever
-   `FleschKincaidReadability`, `DebertaAnswerScores`, and `LLMBasedFaithfulness` are added to evaluate the Generator

Tests:
-   `context_recall`, a metric calculated by `PrecisionRecallF1` needs to be >= 0.9 to pass
-   `deberta_entailment`, a metric calculated by `DebertaAnswerScores` needs to be >= 0.8 to pass


```d2
direction: right
Retriever -> Generator
```


```python title="pipeline.py"
from continuous_eval.eval import Module, Pipeline, Dataset, ModuleOutput
from continuous_eval.metrics.retrieval import PrecisionRecallF1 # Deterministic metric
from continuous_eval.metrics.generation.text import (
    FleschKincaidReadability, # Deterministic metric
    DebertaAnswerScores, # Semantic metric
    LLMBasedFaithfulness, # LLM-based metric
)
from typing import List, Dict
from continuous_eval.eval.tests import GreaterOrEqualThan
dataset = Dataset("data/eval_golden_dataset")

Documents = List[Dict[str, str]]
DocumentsContent = ModuleOutput(lambda x: [z["page_content"] for z in x])

retriever = Module(
    name="retriever",
    input=dataset.question,
    output=Documents,
    eval=[
        PrecisionRecallF1().use( # Reference-based metric that compares the Retrieved Context with the Ground Truths
            retrieved_context=DocumentsContent,
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
    tests=[
        GreaterOrEqualThan( # Set a test using context_recall, a metric calculated by PrecisionRecallF1()
            test_name="Context Recall", metric_name="context_recall", min_value=0.9
        ),
    ],
)

llm = Module(
    name="answer_generator",
    input=retriever,
    output=str,
    eval=[
        FleschKincaidReadability().use( # Reference-free metric that only uses the output of the module
            answer=ModuleOutput()
        ),
        DebertaAnswerScores().use( # Reference-based metric that compares the Answer with the Ground Truths
            answer=ModuleOutput(), ground_truth_answers=dataset.ground_truths
        ),
        LLMBasedFaithfulness().use( # Reference-free metric that uses output from a prior module (Retrieved Context) to evaluate the answer
            answer=ModuleOutput(),
            retrieved_context=ModuleOutput(DocumentsContent, module=retriever), # DocumentsContent from the reranker module
            question=dataset.question,
        ),
    ],
    tests=[
        GreaterOrEqualThan( # Compares each result in the dataset against the min_value, and outputs the mean
            test_name="Deberta Entailment", metric_name="deberta_answer_entailment", min_value=0.8
        ),
    ],
)

pipeline = Pipeline([retriever, llm], dataset=dataset)
print(pipeline.graph_repr()) # visualize the pipeline in Mermaid graph format
```