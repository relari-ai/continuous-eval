import pytest

from continuous_eval.metrics.code.python.code_deterministic_metrics import CodeStringMatch, PythonASTSimilarity
from continuous_eval.metrics.code.sql.deterministic import ASTDiffWeightConfig, SQLASTSimilarity, SQLSyntaxMatch
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
            metric(answer=datum["answer"], ground_truth_answers=datum["ground_truths"]),
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
            metric(answer=datum["answer"], ground_truth_answers=datum["ground_truths"]),
            expected,
        )
        for datum, expected in zip(example_datum.PYTHON_CODE_EXAMPLES, expected_results)
    )


def test_sql_syntax_match():
    expected_results = [{'SQL_Syntax_Match': 1.0}, {'SQL_Syntax_Match': 0}, {'SQL_Syntax_Match': 0}]
    metric = SQLSyntaxMatch()
    assert all(
        all_close(
            metric(answer=datum["answer"], ground_truth_answers=datum["ground_truths"]),
            expected,
        )
        for datum, expected in zip(example_datum.SQL_CODE_EXAMPLES, expected_results)
    )


def test_sql_ast_similarity():
    expected_results = [{'SQL_AST_Similarity': 1.0}, {'SQL_AST_Similarity': 0.9375}, {'SQL_AST_Similarity': 0.8}]
    metric = SQLASTSimilarity()
    assert all(
        all_close(
            metric(answer=datum["answer"], ground_truth_answers=datum["ground_truths"]),
            expected,
        )
        for datum, expected in zip(example_datum.SQL_CODE_EXAMPLES, expected_results)
    )


def test_sql_optimized_ast_similarity():
    expected_results = [{'SQL_AST_Similarity': 1.0}, {'SQL_AST_Similarity': 1.0}, {'SQL_AST_Similarity': 0.75}]
    weights = ASTDiffWeightConfig(
        keep=0.0,
        update=2,
        insert=1.0,
        remove=1.5,
        move=0,
        default=0,
    )
    metric = SQLASTSimilarity(optimize=True, diff_weights=weights)
    assert all(
        all_close(
            metric(answer=datum["answer"], ground_truth_answers=datum["ground_truths"]),
            expected,
        )
        for datum, expected in zip(example_datum.SQL_CODE_EXAMPLES, expected_results)
    )
