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

You can optionally initialize the metric to use optimized SQL queries using the [sqlglot optimizer](https://github.com/tobymao/sqlglot?tab=readme-ov-file#sql-optimizer) and optionally pass in the schema. For example:
```python
schema={"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
sql_syntax_match_optimized = SQLSyntaxSimilarity(optimized=True, schema=schema)
```

## Example Output

```JSON
{
    "SQL_Syntax_Match": 1.0
}
```