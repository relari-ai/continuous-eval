from continuous_eval.metrics.code.python.code_deterministic_metrics import (
    CodeStringMatch,
    PythonASTSimilarity,
)
from continuous_eval.metrics.code.sql.deterministic import (
    ASTDiffWeightConfig,
    SQLASTSimilarity,
    SQLSyntaxMatch,
)
from continuous_eval.metrics.code.sql.llm import SQLCorrectness
from tests.helpers import example_datum
from tests.helpers.utils import all_close, validate_metric_metadata


def test_code_string_match():
    expected_results = [
        {"Exact_Match_Score": 0, "Fuzzy_Match_Score": 0.89},
        {"Exact_Match_Score": 0, "Fuzzy_Match_Score": 0.73},
        {"Exact_Match_Score": 0, "Fuzzy_Match_Score": 0.67},
        {"Exact_Match_Score": 0, "Fuzzy_Match_Score": 0.21},
        {"Exact_Match_Score": 0, "Fuzzy_Match_Score": 0.9},
        {"Exact_Match_Score": 0, "Fuzzy_Match_Score": 0.71},
    ]
    metric = CodeStringMatch()
    results = [
        metric(
            answer=datum["answer"], ground_truth_answers=datum["ground_truths"]
        )
        for datum in example_datum.PYTHON_CODE_EXAMPLES
    ]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(res, expected)
        for res, expected in zip(results, expected_results)
    )


def test_python_ast_similarity():
    expected_results = [
        {"Python_AST_Similarity": 1.0},
        {"Python_AST_Similarity": 0.0},
        {"Python_AST_Similarity": 0.0224},
        {"Python_AST_Similarity": 0.0},
        {"Python_AST_Similarity": -1.0},
        {"Python_AST_Similarity": 0.0937},
    ]
    metric = PythonASTSimilarity()
    results = [
        metric(
            answer=datum["answer"], ground_truth_answers=datum["ground_truths"]
        )
        for datum in example_datum.PYTHON_CODE_EXAMPLES
    ]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(res, expected)
        for res, expected in zip(results, expected_results)
    )


def test_sql_syntax_match():
    expected_results = [
        {"SQL_Syntax_Match": 1.0},
        {"SQL_Syntax_Match": 0},
        {"SQL_Syntax_Match": 0},
    ]
    metric = SQLSyntaxMatch()
    results = [
        metric(
            answer=datum["answer"], ground_truth_answers=datum["ground_truths"]
        )
        for datum in example_datum.SQL_CODE_EXAMPLES
    ]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(res, expected)
        for res, expected in zip(results, expected_results)
    )


def test_sql_ast_similarity():
    expected_results = [
        {"SQL_AST_Similarity": 1.0},
        {"SQL_AST_Similarity": 0.9375},
        {"SQL_AST_Similarity": 0.8},
    ]
    metric = SQLASTSimilarity()
    results = [
        metric(
            answer=datum["answer"], ground_truth_answers=datum["ground_truths"]
        )
        for datum in example_datum.SQL_CODE_EXAMPLES
    ]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(res, expected)
        for res, expected in zip(results, expected_results)
    )


def test_sql_optimized_ast_similarity():
    expected_results = [
        {"SQL_AST_Similarity": 1.0},
        {"SQL_AST_Similarity": 1.0},
        {"SQL_AST_Similarity": 0.75},
    ]
    weights = ASTDiffWeightConfig(
        keep=0.0,
        update=2,
        insert=1.0,
        remove=1.5,
        move=0,
        default=0,
    )
    metric = SQLASTSimilarity(optimize=True, diff_weights=weights)
    results = [
        metric(
            answer=datum["answer"], ground_truth_answers=datum["ground_truths"]
        )
        for datum in example_datum.SQL_CODE_EXAMPLES
    ]
    validate_metric_metadata(metric, results)
    assert all(
        all_close(res, expected)
        for res, expected in zip(results, expected_results)
    )


def test_sql_correctness():
    metric = SQLCorrectness()
    datum = {
        "question": "Retrieve the names and email addresses of all customers who have not made any purchases in the last 6 months.",
        "answer": "SELECT name, email FROM customers WHERE last_purchase_date < DATE_SUB(CURDATE(), INTERVAL 6 MONTH);",
        "ground_truth_answers": """SELECT c.name, c.email
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
WHERE o.order_id IS NULL;""",
    }
    results = metric(**datum)
    validate_metric_metadata(metric, results)

    datum = {
        "question": "Retrieve the names of customers who have placed orders totaling more than $1000.",
        "answer": """SELECT c.name
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
HAVING SUM(o.total_amount) > 1000;""",
        "ground_truth_answers": """SELECT name
FROM customers
WHERE customer_id IN (
    SELECT customer_id
    FROM orders
    GROUP BY customer_id
    HAVING SUM(total_amount) > 1000
);""",
        "schema": {
            "customers": {
                "customer_id": "INT PRIMARY KEY",
                "name": "VARCHAR",
                "email": "VARCHAR",
                "phone": "VARCHAR",
            },
            "orders": {
                "order_id": "INT PRIMARY KEY",
                "customer_id": "INT",
                "order_date": "DATE",
                "total_amount": "DECIMAL",
            },
        },
    }
    results = metric(**datum)
    validate_metric_metadata(metric, results)
