import unittest
from continuous_eval.metrics.code.sql.sql_deterministic_metrics import SQLSyntaxMatch

class TestSQLSyntaxMatch(unittest.TestCase):

    def setUp(self):
        self.metric = SQLSyntaxMatch()

    def test_exact_match(self):
        answer = "SELECT * FROM users;"
        ground_truth = "SELECT * FROM users;"
        result = self.metric(answer, ground_truth)
        self.assertEqual(result["SQL_Syntax_Match_Score"], 1.0)

    def test_case_insensitive_match(self):
        answer = "select * from users;"
        ground_truth = "SELECT * FROM users;"
        result = self.metric(answer, ground_truth)
        self.assertEqual(result["SQL_Syntax_Match_Score"], 1.0)

    def test_whitespace_insensitive_match(self):
        answer = "SELECT * FROM users;"
        ground_truth = "SELECT  *  FROM  users;"
        result = self.metric(answer, ground_truth)
        self.assertEqual(result["SQL_Syntax_Match_Score"], 1.0)

    def test_no_match(self):
        answer = "SELECT * FROM orders;"
        ground_truth = "SELECT * FROM users;"
        result = self.metric(answer, ground_truth)
        self.assertEqual(result["SQL_Syntax_Match_Score"], 0.0)

if __name__ == '__main__':
    unittest.main()
