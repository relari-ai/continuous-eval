---
title: Overview
sidebar:
  badge:
    text: beta
    variant: tip
---

## Evaluation Pipeline

In order to evaluate your pipeline, you first need to define it.
A pipeline is a sequence of steps that transform data from one format to another.

The basic component of a pipeline is a `Module`.
A module is a named component with specific inputs and outputs.

Consider the following example:

```d2
direction: right
Retriever -> LLM
```

In the example above, the pipeline consists of two modules: a retriever and a language model (LLM).

```python
from continuous_eval.eval import Module, Pipeline, Dataset
from typing import List, Dict

dataset = Dataset("dataset_folder")

retriever = Module(
    name="Retriever",
    input=dataset.question,
    output=List[Dict[str, str]],
)

llm = Module(
    name="LLM",
    input=retriever,
    output=str,
)

pipeline = Pipeline([retriever, llm], dataset=dataset)
```

> We will talk about the dataset later
