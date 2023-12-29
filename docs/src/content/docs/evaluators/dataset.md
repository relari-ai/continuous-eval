---
title: Dataset
---

The dataset takes a list of dictionaries or a pandas dataframe and creates a dataset used for the evaluators.

It can be loaded from a jsonl file.

```python
dataset = Dataset.from_jsonl("data/retrieval.jsonl")
```

or exported as a jsonl file.

```python
dataset.to_jsonl("data/retrieval.jsonl")
```

It expects at least one of the following columns combinations:

- ANSWER, QUESTION
- ANSWER, GROUND_TRUTH_ANSWER
- ANSWER, RETRIEVED_CONTEXTS
- QUESTION, RETRIEVED_CONTEXTS
- RETRIEVED_CONTEXTS, GROUND_TRUTH_CONTEXTS
