---
title: Metric Classes
---

### Base Class

The `Metric` class is the base class for all metrics. It provides a common interface for all metrics, and it is used to create new metrics.

A valid metric must implement the following methods:

- `compute`: compute the metric
- `args`: return the arguments of the metric
- `schema`: return the schema of the metric

Let's see an example: consider (a simplified version of) the `TokenCount` metric.

```python
class TokenCount(Metric):
    def __init__(self -> None:
        super().__init__(is_cpu_bound=True)

    def compute(self, retrieved_context, **kwargs):
        ctx = "\n".join(retrieved_context)
        num_tokens = int(len(ctx) / 4.0)
        return {"num_tokens": num_tokens}

    @property
    def args(self):
        return {"retrieved_context": Arg(type=List[str], is_ground_truth=False)}

    @property
    def schema(self):
        return {"num_tokens": Field(type=int)}
```

# TokenCount Class Explanation

The provided code defines a Python class named `TokenCount`, which is a subclass of a base class called `Metric`. This class is designed to compute a specific metric related to token counting in a given context. Here’s a step-by-step explanation of the code:

## Step-by-Step Explanation

### Class Definition

```python
class TokenCount(Metric):
```

This line defines a new class `TokenCount` that inherits from the `Metric` class. This means `TokenCount` will have access to all methods and properties of `Metric`.

### Constructor Method

```python
def __init__(self) -> None:
    super().__init__(is_cpu_bound=True)
```

The `__init__` method is the constructor for the `TokenCount` class. It initializes a new instance of the class, in particular it will inherit the `batch` processing method.

The method supports both CPU-bound and GPU-bound processing. `super().__init__(is_cpu_bound=True)` calls the constructor of the parent class (`Metric`) and passes an argument `is_cpu_bound=True`, indicating that this metric may be CPU-bound. The performance of the metric is heavily influenced by the `is_cpu_bound` flag, so make sure to set it to `True` if the metric is CPU-bound.

You can also disable multi-processing by setting `disable_multiprocessing=True`. Alternatively, you can implement your own `batch` method.

### Compute Method

```python
def compute(self, retrieved_context, **kwargs):
    ctx = "\n".join(retrieved_context)
    num_tokens = int(len(ctx) / 4.0)
    return {"num_tokens": num_tokens}
```

The `compute` method is responsible for calculating the metric. It takes `retrieved_context` as an input, which is expected to be a list of strings. It is **mandatory** to implement this method.

The method returns a dictionary containing the computed number of tokens: `{"num_tokens": num_tokens}`.

### Args Property

The `args` property defines the expected arguments for the metric, which we can infer the arguments of the `compute` method.

```python
@property
def args(self):
    return {"retrieved_context": Arg(type=List[str], is_ground_truth=False)}
```

It returns a dictionary where the key is `"retrieved_context"` and the value is an instance of `Arg`.

### Schema Property

```python
@property
def schema(self):
    return {"num_tokens": Field(type=int)}
```

The `schema` property defines the output structure of the metric. This can be inferred from the `compute` method as well.

`Field(type=int)` indicates that the value associated with `"num_tokens"` is expected to be of type integer.

## Default Metrics

Continuous-eval provides a set of default metrics that are useful for evaluating the performance of a model. These metrics are implemented in the `metrics` module.

