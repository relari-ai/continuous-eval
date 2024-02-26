---
title: Custom Metrics
sidebar:
  order: 1
---

## Create your own metric

To define your own metrics, you only need to extend the [Metric](https://github.com/relari-ai/continuous-eval/blob/main/continuous_eval/metrics/base.py) class implementing the `__call__` method.

Optional methods are `batch` (if it is possible to implement optimizations for batch processing) and `aggregate` (to aggregate metrics results over multiple samples_).

Check out [Metric Folder](https://github.com/relari-ai/continuous-eval/tree/main/continuous_eval/metrics) for examples of how various types of metrics are implemented.

## Example

```python title="CustomMetric.py"
from continuous_eval.metrics.base import Metric

class CustomMetric(Metric):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, input_from_dataset, **kwargs):
        
        # implement metric calculation
        score = ...

        return {"custom_metric_score": score}
```

## Add additional LLM Interface

If you want to use a different LLM endpoint, you can augment the `LLMFactory` to add the model interface and parameters.
Check out [LLMFactory](https://github.com/relari-ai/continuous-eval/blob/main/continuous_eval/llm_factory.py) for details.
