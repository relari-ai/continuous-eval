---
title: Evaluation Dataset
---

### `Dataset` Class 

**The `Dataset` class takes a folder that contains a flexible evaluation dataset tailored to your evaluation pipeline.**

A custom dataset needs to contain:
- `dataset.jsonl` which contains a collection of query / instructions and corresponding reference outputs by the modules in the pipeline.
- `manifest.yaml` which declares the structure and fields of the dataset to be used by `Pipeline` and `EvaluationManager` instances.

```python
from continuous_eval.eval import Dataset

dataset = Dataset.from_jsonl("data_folder")
```

### Example Data Folder

Here's an example golden dataset that contains `uuid`, `question`, `answer` (ground truth answers), and `tool_calls` (the tools that are supposed to be used).

#### dataset.jsonl
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
#### manifest.yaml
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