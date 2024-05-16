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
            # Simple string comparison for now, can be improved with more sophisticated methods
            match_score = float(formatted_answer == formatted_gt)
            if match_score > max_match_score:
                max_match_score = match_score

        # Return the maximum match score
        return {"SQL_Syntax_Match_Score": max_match_score}
