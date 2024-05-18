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

The basic component of a pipeline is a `Module`.
A module is a named component with specific inputs and outputs.
It can be a simple function or a complex model that takes some input and returns some output.

To define a module you need to specify the following:

- `name`: a unique name for the module
- `input`: the input of the module, can be a dataset field (`DatasetField`, see dataset page) another module or nothing (`None`)
- `output` the output type (e.g., `str`, `List[str]`, `Dict[str, str]`, etc.)
- `description`: Optional string describing the field
- `eval`: an optional list of metrics (see next page)
- `tests`: an optional list of tests (see next page)

Through the `Pipeline` class, you can define a sequence of modules that represent your application pipeline.

### Example

Consider the following pipeline example:

```d2
direction: right
Dataset: Eval Dataset
Dataset.shape: oval
Dataset -> Retriever
Retriever -> Reranker -> Generator
```

This Retrieval-Augmented Generation (RAG) pipeline consists of three simple modules. A Retriever that fetches the relevant documents, a Reranker that reorders and filters the documents, and a Generator that uses LLM to generate a response based on information in the documents.

```python title="pipeline.py"
from continuous_eval.eval import Module, Pipeline, Dataset
from typing import List, Dict

dataset = Dataset("dataset_folder") # Evaluation dataset that contains all the questions and optional the expected module outputs

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
