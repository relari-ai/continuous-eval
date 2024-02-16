import pytest

from continuous_eval.metrics import CodeStringMatch, PythonASTSimilarity
from tests.helpers import example_datum
from tests.helpers.utils import all_close


def test_code_string_match():
    data = [example_datum.PYTHON_FUNCTION_EXAMPLE]
    expected_results = [
        {
            "Exact_Match_Score": 0.0,
            "Fuzzy_Match_Score": 0.88,
        },
    ]
    metric = CodeStringMatch()
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))


def test_python_ast_similarity():
    data = [example_datum.PYTHON_FUNCTION_EXAMPLE]
    expected_results = [{'Python_AST_Similarity': 1.0}]
    metric = PythonASTSimilarity()
    assert all(all_close(metric.calculate(**datum), expected) for datum, expected in zip(data, expected_results))
