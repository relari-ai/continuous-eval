from typing import List, Union

import sqlparse
from sqlglot import diff, parse_one
from sqlglot.diff import Insert, Keep, Move, Remove, Update

from continuous_eval.metrics.base import Metric


class SQLSyntaxMatch(Metric):
    """
    This metric evaluates the syntactic similarity between the generated SQL query and a set of ground truth queries.
    It uses the sqlparse library to format and compare the SQL queries.
    """

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str]):
        if isinstance(ground_truth_answers, str):
            ground_truth_answers = [ground_truth_answers]

        formatted_answer = sqlparse.format(answer, reindent=True, keyword_case="upper")
        formatted_ground_truths = [
            sqlparse.format(gt, reindent=True, keyword_case="upper") for gt in ground_truth_answers
        ]

        max_match_score = 0

        for formatted_gt in formatted_ground_truths:
            match_score = float(formatted_answer == formatted_gt)
            if match_score > max_match_score:
                max_match_score = match_score

        return {"SQL_Syntax_Match": max_match_score}


class SQLASTSimilarity(Metric):
    """
    Compare SQL queries using AST similarity, considering different types of changes differently and improving normalization.
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
            self._calculate_similarity(answer_tree, ground_truth_tree) for ground_truth_tree in ground_truth_trees
        ]

        return {
            "SQL_AST_Similarity": max(similarity_scores) if similarity_scores else -1.0,
        }

    def _calculate_similarity(self, tree1, tree2):
        diff_result = diff(tree1, tree2)
        total_changes = sum(self._change_weight(change) for change in diff_result)
        max_nodes = max(len(list(tree1.walk())), len(list(tree2.walk())))
        similarity_score = 1 - (total_changes / max_nodes) if max_nodes > 0 else 1
        return similarity_score

    def _change_weight(self, change):
        """
        Assign weights to different types of changes based on their expected impact on query semantics.
        """
        if isinstance(change, Keep):
            return 0
        elif isinstance(change, Update):
            return 1.5  # Updates are significant as they imply a modification in function or value.
        elif isinstance(change, Insert) or isinstance(change, Remove):
            return 1  # Inserts and Removes affect the structure and content but are simpler than updates.
        elif isinstance(change, Move):
            return 0.5  # Moves are generally less impactful as they simply change the order.
        return 1  # Default weight for other types of changes
