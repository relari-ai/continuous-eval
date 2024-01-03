---
title: Dataset Class
---

### `Dataset` Class 

**The `Dataset` class takes a list of dictionaries or a pandas dataframe and creates a dataset used for the `Evaluators`.**

It can be loaded from a jsonl file.

```python
dataset = Dataset.from_jsonl("data/retrieval.jsonl")
```

or exported as a jsonl file.

```python
dataset.to_jsonl("data/retrieval.jsonl")
```

It expects at least one of the following columns combinations:

- `answer`, `question`
- `answer`, `ground_truths`
- `answer`, `retrieved_contexts`
- `question`, `retrieved_contexts`
- `retrieved_contexts`, `ground_truth_contexts`

:::tip
Check out the **[Data Dependency Table]()** in Overview of Metrics, to make sure your dataset contains the columns need to compute the metrics of your choice.
:::