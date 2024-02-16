import pytest

from continuous_eval.metrics import CodeStringMatch, PythonASTSimilarity
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
        all_close(metric.calculate(**datum), expected)
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
        all_close(metric.calculate(**datum), expected)
        for datum, expected in zip(example_datum.PYTHON_CODE_EXAMPLES, expected_results)
    )
