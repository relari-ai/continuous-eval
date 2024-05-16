# Add SQL Metrics Implementation

This pull request introduces the `SQLSyntaxMatch` class, which is designed to evaluate the syntactic similarity between generated SQL queries and a set of ground truth queries. The implementation uses the `sqlparse` library to format and compare SQL queries, ensuring a consistent and standardized comparison.

## Changes
- A new file `sql_deterministic_metrics.py` has been added to the `continuous_eval/metrics/code/sql/` directory.
- The `SQLSyntaxMatch` class extends the `Metric` base class and overrides the `__call__` method to perform the evaluation.
- The `sqlparse` library is utilized to format the SQL queries before comparison.

## Usage
The `SQLSyntaxMatch` metric can be used as follows:
```python
from continuous_eval.metrics.code.sql.sql_deterministic_metrics import SQLSyntaxMatch

# Instantiate the metric
sql_syntax_match = SQLSyntaxMatch()

# Evaluate syntactic similarity
result = sql_syntax_match(answer="SELECT * FROM users;", ground_truth_answers=["SELECT * FROM users;"])
print(result)  # Output: {"SQL_Syntax_Match_Score": 1.0}
```

This metric provides a foundational step towards building out the SQL metrics under Code Generation in the continuous-eval repository. Further refinement and testing are planned to ensure robustness and integration with the existing metrics system.
