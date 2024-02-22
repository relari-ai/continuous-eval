---
title: Python AST Similarity
sidebar:
  badge:
    text: new
    variant: note
---

### Definitions

**Python AST Similarity** compares the structure of two Python programs (generated code string vs. ground truth code string) by analyzing their Abstract Syntax Trees (ASTs). It evaluates how similar these programs are by matching nodes in the trees, considering both the types of statements and their organization. The comparison can involve reordering certain parts for a deeper match and uses a scoring system to quantify similarity. 

<br>

:::note
The metric depends on syntactically correct Python scripts to produce the Abstract Syntax Trees (ASTs). If the scripts contain syntax errors and cannot be parsed, the metric will yield a score of -1.0.
:::

<br>

### Example Usage

Required data items: `answer`, `ground_truths`

```python
from continuous_eval.metrics import PythonASTSimilarity

datum = {
    "answer": "def function(x, y):\n  return x + y",
    "ground_truths": [
        "def foo(x, y):\n  return x * y",
        "def foo(x, y):\n  return x + y",
    ],
},

metric = PythonASTSimilarity()
print(metric(**datum))
```

### Example Output

```JSON
{
    "Python_AST_Similarity": 1.0
}
```
