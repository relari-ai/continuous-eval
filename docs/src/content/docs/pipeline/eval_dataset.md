---
title: Evaluation Dataset
---

## Dataset Class

The `Dataset` class is a convenient class that represent a dataset that can be used for evaluation.

The dataset class can be initialized with a path to a folder or a file.
The folder should contain the following files:

- `dataset.jsonl` which contains a collection of query / instructions and corresponding reference outputs by the modules in the pipeline.
- an optional `manifest.yaml` which declares the structure and fields of the dataset, the license and other metadata.

```python
from continuous_eval.eval import Dataset

dataset = Dataset("path_to_folder") # or Dataset("path_to_file.jsonl")
```

Alternatively, you can also create a dataset from a list of dictionaries:

```python
dataset = Dataset.from_data([
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Germany?", "answer": "Berlin"},
])
```

To access the raw data, you can use the `data` attribute:

```python
print(dataset.data[0])
```

### Dataset fields

Suppose you want to reference a dataset field, you can use the `DatasetField` class:

```python
class DatasetField:
    name: str
    type: type = typing.Any  # type: ignore
    description: str = ""
    is_ground_truth: bool = False
```

When you load the dataset, the `Dataset` class will automatically infer the fields from the data.

```python
type(dataset.question)  # DatasetField
```

this will be particularly useful when defining the input and output of the modules in the pipeline.

### Example Data Folder

Here's an example golden dataset that contains `uid`, `question`, `answer` (ground truth answers), and `tool_calls` (the tools that are supposed to be used).

#### Dataset File

```json title="data_folder/dataset.jsonl"
{
  "uuid": "1",
  "question": "What is Uber revenue as of March 2022?",
  "answer": [
    "Uber's revenue as of March 2022 is $6,854 million.",
    "$6,854 million",
    "$6,854M"
  ],
  "tool_calls": [
    {
      "name": "march"
    }
  ]
}{
  "uuid": "2",
  "question": "What is Uber revenue as of Sept 2022?",
  "answer": [
    "Uber's revenue as of September 2022 is $23,270 million.",
    "$23,270 million",
    "$23,270M"
  ],
  "tool_calls": [
    {
      "name": "sept"
    }
  ]
}{
  "uuid": "3",
  "question": "What is Uber revenue as of June 2022?",
  "answer": [
    "Uber's revenue as of September 2022 is $8,073 million.",
    "$8,073 million",
    "$8,073M"
  ],
  "tool_calls": [
    {
      "name": "june"
    }
  ]
}
```

#### Manifest (optional)

```yaml title="data_folder/manifest.yaml"
name: Uber 10Q
description: Uber 10Q filings from 2022
format: jsonl
license: CC0
fields:
  uuid:
    description: Unique identifier for the filing
    type: UUID
  question:
    description: The question asked in the filing
    type: str
  answer:
    description: The answer to the question
    type: List[str]
  tool_calls:
    description: The tools used to extract the question and answer
    type: List[Dict[str, str]]
  ```