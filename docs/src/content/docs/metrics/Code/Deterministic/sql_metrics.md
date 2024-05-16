---
title: SQL Syntax Match
sidebar:
  order: 2
---

## Definitions

**SQL Syntax Match** evaluates the syntactic similarity between generated SQL queries and a set of ground truth queries. It compares the structure and syntax of SQL statements to determine how closely they match, considering the order and type of clauses, keywords, and conditions.

:::note
The metric requires syntactically correct SQL queries to function properly. If the queries contain syntax errors and cannot be parsed, the metric will yield a score of 0.0.
:::

## Example Usage

Required data items: `answer`, `ground_truth_answers`

```python
from continuous_eval.metrics.code.sql.sql_deterministic_metrics import SQLSyntaxMatch

# Instantiate the metric
sql_syntax_match = SQLSyntaxMatch()

# Evaluate syntactic similarity
result = sql_syntax_match(answer="SELECT * FROM users;", ground_truth_answers=["SELECT * FROM users;"])
print(result)  # Output: {"SQL_Syntax_Match_Score": 1.0}
```

## Example Output

```JSON
{
    "SQL_Syntax_Match_Score": 1.0
}
```

The `SQLSyntaxMatch` class returns a dictionary with a single key-value pair. The key is `SQL_Syntax_Match_Score`, and the value is a float representing the syntactic match score. A score of `1.0` indicates an exact match, while a score of `0.0` indicates no match.

For more detailed examples and advanced usage, please refer to the test cases in the `code_metrics_test.py` file in the `tests` directory.
