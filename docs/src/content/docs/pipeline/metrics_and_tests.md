---
title: Metrics and Tests
sidebar:
  badge:
    text: new
    variant: tip
---

## Metrics

When defining modules in the pipeline you can also specify metrics to evaluate the module outputs. The metrics are defined in the `eval` attribute of the module definition.

To specify the input to each metric you can use the `use` method.
For example, suppose we have a retriever on which we want to use the `PrecisionRecallF1` metric.
We can define the retriever as follows:

```python
from continuous_eval.eval import Module, ModuleOutput
from continuous_eval.metrics.retrieval import PrecisionRecallF1

retriever = Module(
    name="retriever",
    input=dataset.question,
    output=Documents,
    eval=[
        PrecisionRecallF1().use(
            retrieved_context=ModuleOutput(),
            ground_truth_context=dataset.ground_truth_context,
        ),
    ],
)
```

The `PrecisionRecallF1` metric expects two inputs: `retrieved_context` and `ground_truth_context`.
To use it to evaluate the module we specify that the `retrieved_context` is the module's output while the `ground_truth_context` is the dataset's ground truth context (here we used the dataset field).

The ModuleOutput class is flexible and allows for custom selectors.
Since `PrecisionRecallF1` expect a `List[str]` as input for both arguments, by specifying `ModuleOutput` we assume the module is actually returning a list of strings. Suppose instead that it returns a list of dictionaries where `"page_content"` is the key for the text we want to evaluate.
We could specify the output as follows:

```python
PrecisionRecallF1().use(
    retrieved_context=ModuleOutput(lambda x: [z["page_content"] for z in x]),
    ground_truth_context=dataset.ground_truth_context,
),
```

the evaluation runner will take care of extracting the relevant data from the module output and the dataset.

## Tests

Each module can also have tests to ensure the module is working as expected.
The tests are defined in the `tests` attribute of the module definition.

Suppose we want to make sure the average precision of the retriever is greater than 0.8.

```python
from continuous_eval.eval import Module, ModuleOutput
from continuous_eval.metrics.retrieval import PrecisionRecallF1
from continuous_eval.eval.tests import MeanGreaterOrEqualThan

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
```

The `MeanGreaterOrEqualThan` test expects the name of the metric to test, the minimum value, and the name of the test.
The evaluation runner will run the test and report the results.

```python
evalrunner = EvaluationRunner(pipeline)
metrics = evalrunner.evaluate()

print("\nTests results:")
tests = evalrunner.test(metrics)
for module_name, test_results in tests.results.items():
    print(f"{module_name}")
    for test_name in test_results:
        print(f" - {test_name}: {'PASS' if test_results[test_name] else 'FAIL'}")
```

The output will be:

```text
Tests results:
retriever
 - Average Precision: PASS
```
