---
title: StringMatch
sidebar:
    order: 1
---

### Definitions

**Code String Match** measures how close the generated code string is to the ground truth code string.

It outputs both the binary exact match score and the fuzzy match score in the range of (0.0 - 1.0).

<br>


### Example Usage

Required data items: `answer`, `ground_truths`

```python
from continuous_eval.metrics import CodeStringMatch

datum = {
    "answer": "def function(x, y):\n  return x + y",
    "ground_truths": [
        "def foo(x, y):\n  return x * y",
        "def foo(x, y):\n  return x + y",
    ],
},

metric = CodeStringMatch()
print(metric(**datum))
```

### Example Output

```JSON
{
    "Exact_Match_Score": 0, 
    "Fuzzy_Match_Score": 0.89
}
```
