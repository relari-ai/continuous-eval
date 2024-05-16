from typing import List, Union

import sqlparse

from continuous_eval.metrics.base import Metric

class SQLSyntaxMatch(Metric):
    """
    This metric evaluates the syntactic similarity between the generated SQL query and a set of ground truth queries.
    It uses the sqlparse library to format and compare the SQL queries.
    """

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str]):
        if isinstance(ground_truth_answers, str):
            ground_truth_answers = [ground_truth_answers]

        # Format the answer and ground truth answers using sqlparse for consistent comparison
        formatted_answer = sqlparse.format(answer, reindent=True, keyword_case="upper")
        formatted_ground_truths = [
            sqlparse.format(gt, reindent=True, keyword_case="upper")
            for gt in ground_truth_answers
        ]

        # Initialize the maximum match score
        max_match_score = 0

        # Compare the formatted answer with each formatted ground truth answer
        for formatted_gt in formatted_ground_truths:
            # Replace simple string comparison with AST comparison
            match_score = float(self.compare_ast(formatted_answer, formatted_gt))
            if match_score > max_match_score:
                max_match_score = match_score

        # Return the maximum match score
        return {"SQL_Syntax_Match_Score": max_match_score}

    def compare_ast(self, query1: str, query2: str) -> float:
        # Parse the queries into ASTs
        ast1 = sqlparse.parse(query1)
        ast2 = sqlparse.parse(query2)

        # Compare the structure of the ASTs
        # This is a placeholder and would need to be replaced with a real implementation
        return float(ast1 == ast2)
