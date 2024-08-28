import unittest
from sqlglot import parse_one
from .code_deterministic_metrics import SQLASTSimilarity

class TestSQLASTSimilarity(unittest.TestCase):
    def setUp(self):
        self.metric = SQLASTSimilarity()

    def test_exact_match(self):
        query1 = "SELECT a, b, c FROM table"
        query2 = "SELECT a, b, c FROM table"
        result = self.metric(query1, query2)
        self.assertAlmostEqual(result["SQL_AST_Similarity"], 1.0)

    def test_different_queries(self):
        query1 = "SELECT a, b, c FROM table"
        query2 = "SELECT x, y, z FROM table"
        result = self.metric(query1, query2)
        self.assertLess(result["SQL_AST_Similarity"], 1.0)

    def test_similar_queries(self):
        query1 = "SELECT a, b, c FROM table"
        query2 = "SELECT a, b, c FROM table WHERE a > 10"
        result = self.metric(query1, query2)
        self.assertGreater(result["SQL_AST_Similarity"], 0.0)
        self.assertLess(result["SQL_AST_Similarity"], 1.0)

    def test_invalid_query(self):
        query1 = "SELECT a, b, c FROM table"
        query2 = "INVALID SQL QUERY"
        result = self.metric(query1, query2)
        self.assertEqual(result["SQL_AST_Similarity"], -1.0)

if __name__ == "__main__":
    unittest.main()
