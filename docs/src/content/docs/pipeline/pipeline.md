---
title: Pipeline Overview
sidebar:
  badge:
    text: new
    variant: tip
---

## Definition

To evaluate your custom AI application pipeline, you first need to define it in using the `Pipeline` classes.
<br>

A pipeline is a sequence of steps that transform data from one format to another. In a typical AI application, it usually starts with a user instruction, then goes through a series of steps (`Module`) to return an answer.

The basic component of a pipeline is a `Module`. A module is a named component with specific inputs and outputs.

## Example

Consider the following pipeline example:

```d2
direction: right
Retriever -> Reranker -> Generator
```

This Retrieval-Augmented Generation (RAG) pipeline consists of three simple modules. A Retriever that fetches the relevant documents, a Reranker that reorders and filters the documents, and a Generator that uses LLM to generate a response based on information in the documents.

```python title="pipeline.py"
from continuous_eval.eval import Module, Pipeline, Dataset
from typing import List, Dict

dataset = Dataset("dataset_folder") # Evaluation dataset that contains all the questions and optionall the expected module outputs

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

