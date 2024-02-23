---
title: Pipeline Overview
sidebar:
  badge:
    text: new
    variant: tip
---

## Definition

In order to evaluate your AI application pipeline, you first need to define it.
A pipeline is a sequence of steps that transform data from one format to another.

The basic component of a pipeline is a `Module`.
A module is a named component with specific inputs and outputs.

## Example

Consider the following example:

```d2
direction: right
Retriever -> Reranker -> Generator
```

In the example above, the pipeline consists of three simple modules: a retriever and an LLM generator.

```python title="pipeline.py"
from continuous_eval.eval import Module, Pipeline, Dataset
from typing import List, Dict

dataset = Dataset("dataset_folder") # This is the dataset you will use you evaluate the pipeline module.

retriever = Module(
    name="Retriever",
    input=dataset.question,
    output=List[Dict[str, str]],
)

reranker = Module(
    name="Reranker",
    input=retriever,
    output=List[Dict[str, str]],
)

llm = Module(
    name="LLM",
    input=reranker,
    output=str,
)

pipeline = Pipeline([retriever, reranker, llm], dataset=dataset)
print(pipeline.graph_repr()) # visualize the pipeline in Mermaid graph format
```

