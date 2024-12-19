---
title: SQL Correctness
sidebar:
    order: 1
---

### Definition

**SQL Correctness** evaluates the alignment between the generated SQL query and the ground truth SQL query with respect to the intended functionality of the query.

## Example Usage

Required data items: `answer`, `ground_truth_answers`

```python
from continuous_eval.metrics.code.sql import SQLCorrectness

datum = {
    "question": "What's the name of the user with email 'john@example.com'?",
    "answer": "SELECT * FROM users where email = 'john@example.com';",
    "ground_truth_answers": "SELECT name FROM users where email = 'john@example.com';",
}

metric = SQLCorrectness()
print(metric(**datum))
```

### Example Output

```python
{
    "reasoning": "The generated SQL query retrieves data from the 'users' table based on the email 'john@example.com', but it uses 'SELECT *', returning all columns instead of just the 'name' as required.",
    "score": 0.5761831599,
}
```

## Schema

You can optionally pass the database schema.

```python
from continuous_eval.metrics.code.sql import SQLCorrectness

datum = {
    "question": "What's the name of the user with email 'john@example.com'?",
    "answer": "SELECT * FROM users where email = 'john@example.com';",
    "ground_truth_answers": "SELECT name FROM users where email = 'john@example.com';",
    "schema": {
        "users": ["id", "name", "email"],
        "orders": ["id", "user_id", "amount"],
    },
}

metric = SQLCorrectness()
print(metric(**datum))
```

