---
title: SQL AST Similarity
sidebar:
  order: 1
---

### Definitions

**SQL AST Similarity** compares the structure of two SQL queries by analyzing their Abstract Syntax Trees (ASTs). This metric assesses similarity by matching the nodes within these trees, taking into account the statement types and their arrangement. Different types of tree differences (such as insert, remove, update, move, etc.) are weighted differently to calculate the final similarity score.

<br>

$$
\text{SQL AST Similarity} = 1 - \frac{\text{Total Weight Changes}}{\text{Maximum Possible Nodes}}
$$

<br>

:::note
The metric depends on syntactically correct SQL queries to produce the Abstract Syntax Trees (ASTs). If the scripts contain syntax errors and cannot be parsed, the metric will yield a score of -1.0.
:::

<br>

### Example Usage

Required data items: `answer`, `ground_truth_answers`

```python
from continuous_eval.metrics.code.sql import SQLASTSimilarity

datum = {
    "answer": "SELECT name, age FROM customers",
    "ground_truth_answers": "SELECT age, name FROM customers",
}

metric = SQLASTSimilarity()
print(metric(**datum))
```

You can optionally initialize the metric to use optimized SQL queries using the [sqlglot optimizer](https://github.com/tobymao/sqlglot?tab=readme-ov-file#sql-optimizer) and optionally pass in the schema. For example:

```python
schema={"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
sql_syntax_match_optimized = SQLASTSimilarity(optimized=True, schema=schema)
```

You can also customize weights to different types of nodes in the AST diff.
Higher weights indicate more significant changes, which are expected to have a greater impact on query semantics.

```python
from continuous_eval.metrics.code.sql.deterministic import ASTDiffWeightConfig

weights = ASTDiffWeightConfig(
    keep=0.0,
    update=2,
    insert=1.0,
    remove=1.5,
    move=0,
    default=0,
)
metric = SQLASTSimilarity(diff_weights=weights)
```

### Example Output

```JSON
{
    "SQL_AST_Similarity": 0.9375
}
```
