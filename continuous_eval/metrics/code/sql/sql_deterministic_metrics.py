from typing import List, Union
import sqlparse
from sqlglot import diff, parse_one
from sqlglot.diff import Keep
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

class SQLASTSimilarity(Metric):
    """
    Compare SQL queries using AST similarity.
    """

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str], **kwargs):
        if isinstance(ground_truth_answers, str):
            ground_truth_answers = [ground_truth_answers]

        try:
            answer_tree = parse_one(answer)
            ground_truth_trees = [parse_one(gt) for gt in ground_truth_answers]
        except Exception as e:
            return {"SQL_AST_Similarity": -1.0}

        similarity_scores = [
            self._calculate_similarity(answer_tree, ground_truth_tree)
            for ground_truth_tree in ground_truth_trees
        ]

        return {
            "SQL_AST_Similarity": max(similarity_scores),
        }

    def _calculate_similarity(self, tree1, tree2):
        diff_result = diff(tree1, tree2)
        total_changes = len([change for change in diff_result if not isinstance(change, Keep)])
        total_nodes = len(list(tree1.walk())) + len(list(tree2.walk()))
        similarity_score = 1 - (total_changes / total_nodes)
        return similarity_score
