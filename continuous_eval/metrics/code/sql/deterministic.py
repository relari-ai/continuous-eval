from dataclasses import dataclass
from typing import List, Union

from sqlglot import diff, parse_one, transpile
from sqlglot.diff import Insert, Keep, Move, Remove, Update
from sqlglot.optimizer import optimize

from continuous_eval.metrics.base import Metric


class SQLMetric(Metric):
    def __init__(self, optimize: bool = False, schema=None):
        self.optimize = optimize
        self.schema = schema

    def prepare_query(self, sql: str):
        """
        Parse, transpile, and optionally optimize a SQL query.
        """
        formatted_sql = transpile(sql, pretty=True, comments=False)[0]
        if self.optimize:
            try:
                optimized_sql = optimize(parse_one(formatted_sql), schema=self.schema).sql(pretty=True)
                return optimized_sql
            except Exception as e:
                print(f"Failed to optimize SQL query given schema: {e}. Using unoptimized query.")
                return formatted_sql
        return formatted_sql


class SQLSyntaxMatch(SQLMetric):
    """
    This metric evaluates the syntactic similarity between the generated SQL query and a set of ground truth queries.
    It uses the sqlglot library to format and compare the SQL queries.
    """

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str]):

        transformed_answer = self.prepare_query(answer)
        transformed_ground_truths = [self.prepare_query(gt) for gt in ground_truth_answers]

        max_match_score = 0

        for transformed_gt in transformed_ground_truths:
            match_score = float(transformed_answer == transformed_gt)
            if match_score > max_match_score:
                max_match_score = match_score

        return {"SQL_Syntax_Match": max_match_score}


@dataclass(frozen=True)
class ASTDiffWeightConfig:
    keep_weight: float = 0
    update_weight: float = 1.5  # Updates are significant as they imply a modification in function or value.
    insert_weight: float = 1.0  # Inserts affect the structure and content but are simpler than updates.
    remove_weight: float = 1.0  # Removes affect the structure and content but are simpler than updates.
    move_weight: float = 0.5  # Moves are generally less impactful as they simply change the order.
    default_weight: float = 1.0  # Default weight for other types of changes


class SQLASTSimilarity(SQLMetric):
    """
    Compare SQL queries using AST similarity, considering different types of changes differently and improving normalization.
    """

    def __init__(self, optimize: bool = False, schema=None, diff_weights: ASTDiffWeightConfig = ASTDiffWeightConfig()):
        super().__init__(optimize=optimize, schema=schema)
        self.diff_weights = diff_weights

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str], **kwargs):

        transformed_answer = self.prepare_query(answer)
        transformed_ground_truths = [self.prepare_query(gt) for gt in ground_truth_answers]

        try:
            answer_tree = parse_one(transformed_answer)
            ground_truth_trees = [parse_one(gt) for gt in transformed_ground_truths]
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
            return self.diff_weights.keep_weight
        elif isinstance(change, Update):
            return self.diff_weights.update_weight
        elif isinstance(change, Insert):
            return self.diff_weights.insert_weight
        elif isinstance(change, Remove):
            return self.diff_weights.remove_weight
        elif isinstance(change, Move):
            return self.diff_weights.move_weight
        else:
            return self.diff_weights.default_weight
