---
title: SQL Syntax Match
sidebar:
  order: 2
---

## Definitions

**SQL Syntax Match** evaluates the syntactic equivelance between generated SQL queries and a set of ground truth queries. The strict comparison can tolerate formatting changes.

## Example Usage

Required data items: `answer`, `ground_truth_answers`

```python
from continuous_eval.metrics.code import SQLSyntaxMatch

sql_syntax_match = SQLSyntaxMatch()

datum = {
    "answer": "SELECT * FROM users;"",
    "ground_truth_answers": [
        "SELECT  *  from  users;"
    ],
},

metric = SQLSyntaxMatch()
print(metric(**datum))
```

## Example Output

```JSON
{
    "SQL_Syntax_Match": 1.0
}
```