import pytest

from continuous_eval.metrics.code.python.code_deterministic_metrics import CodeStringMatch, PythonASTSimilarity
from continuous_eval.metrics.code.sql.sql_deterministic_metrics import SQLSyntaxMatch
from tests.helpers import example_datum
from tests.helpers.utils import all_close


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
    assert all(
        all_close(
            metric(answer=datum["answer"], ground_truth_answers=datum["ground_truths"]),  # type: ignore
            expected,
        )
        for datum, expected in zip(example_datum.PYTHON_CODE_EXAMPLES, expected_results)
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
    assert all(
        all_close(
            metric(answer=datum["answer"], ground_truth_answers=datum["ground_truths"]),  # type: ignore
            expected,  # type: ignore
        )
        for datum, expected in zip(example_datum.PYTHON_CODE_EXAMPLES, expected_results)
    )

# SQLSyntaxMatch tests using pytest
def test_sql_syntax_exact_match():
    metric = SQLSyntaxMatch()
    answer = "SELECT * FROM users;"
    ground_truth = "SELECT * FROM users;"
    result = metric(answer, ground_truth)
    assert result["SQL_Syntax_Match_Score"] == 1.0

def test_sql_syntax_case_insensitive_match():
    metric = SQLSyntaxMatch()
    answer = "select * from users;"
    ground_truth = "SELECT * FROM users;"
    result = metric(answer, ground_truth)
    assert result["SQL_Syntax_Match_Score"] == 1.0

def test_sql_syntax_whitespace_insensitive_match():
    metric = SQLSyntaxMatch()
    answer = "SELECT * FROM users;"
    ground_truth = "SELECT  *  FROM  users;"
    result = metric(answer, ground_truth)
    assert result["SQL_Syntax_Match_Score"] == 1.0

def test_sql_syntax_no_match():
    metric = SQLSyntaxMatch()
    answer = "SELECT * FROM orders;"
    ground_truth = "SELECT * FROM users;"
    result = metric(answer, ground_truth)
    assert result["SQL_Syntax_Match_Score"] == 0.0

@pytest.mark.skip(reason="Partial match scoring not implemented yet")
def test_sql_syntax_partial_match():
    metric = SQLSyntaxMatch()
    answer = "SELECT id, name FROM users;"
    ground_truth = "SELECT * FROM users;"
    result = metric(answer, ground_truth)
    # Assuming a hypothetical partial match score of 0.5
    assert result["SQL_Syntax_Match_Score"] == 0.5
