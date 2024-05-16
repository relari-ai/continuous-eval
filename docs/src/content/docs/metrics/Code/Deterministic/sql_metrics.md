# SQL Metrics Documentation

## Overview
The `SQLSyntaxMatch` class is part of the SQL metrics implementation in the `continuous-eval` repository. It is designed to evaluate the syntactic similarity between generated SQL queries and a set of ground truth queries.

## Usage
To use the `SQLSyntaxMatch` metric, you need to import the class from the `sql_deterministic_metrics.py` file and instantiate it. Then, you can call the instance with an SQL query and a ground truth query or a list of ground truth queries.

Here is an example of how to use the `SQLSyntaxMatch` class:

```python
from continuous_eval.metrics.code.sql.sql_deterministic_metrics import SQLSyntaxMatch

# Instantiate the metric
sql_syntax_match = SQLSyntaxMatch()

# Evaluate syntactic similarity
result = sql_syntax_match(answer="SELECT * FROM users;", ground_truth_answers=["SELECT * FROM users;"])
print(result)  # Output: {"SQL_Syntax_Match_Score": 1.0}
```

## Output
The `SQLSyntaxMatch` class returns a dictionary with a single key-value pair. The key is `SQL_Syntax_Match_Score`, and the value is a float representing the syntactic match score. A score of `1.0` indicates an exact match, while a score of `0.0` indicates no match.

## Integration with Continuous Evaluation
The SQL metrics, including the `SQLSyntaxMatch` class, are integrated into the continuous evaluation framework. They can be used to evaluate SQL code generation models and other applications where SQL query evaluation is required.

For more detailed examples and advanced usage, please refer to the test cases in the `code_metrics_test.py` file in the `tests` directory.
