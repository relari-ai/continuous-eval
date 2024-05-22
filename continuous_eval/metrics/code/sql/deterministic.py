import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from sqlglot import diff, parse_one, transpile
from sqlglot.diff import Insert, Keep, Move, Remove, Update
from sqlglot.optimizer import optimize

from continuous_eval.metrics.base import Metric

logger = logging.getLogger("metrics")


@dataclass(frozen=True)
class ASTDiffWeightConfig:
    """
    Configuration for assigning weights to different types of changes in the AST diff.
    Higher weights indicate more significant changes, which are expected to have a greater impact on query semantics.
    """

    keep: float = 0.0
    # Updates are significant as they imply a modification in function or value.
    update: float = 1.5
    # Inserts affect the structure and content but are simpler than updates.
    insert: float = 1.0
    # Removes affect the structure and content but are simpler than updates.
    remove: float = 1.0
    # Moves are generally less impactful as they simply change the order.
    move: float = 0.5
    # Default weight for other types of changes
    default: float = 1.0


class _SQLMetric:
    def __init__(self, optimize: bool = False, schema: Optional[Dict] = None):
        self._optimize = optimize
        self._schema = schema

    def _prepare_query(self, sql: str):
        """
        Parse, transpile, and optionally optimize a SQL query.
        """
        formatted_sql = transpile(sql, pretty=True, comments=False)[0]
        if self._optimize:
            try:
                optimized_sql = optimize(parse_one(formatted_sql), schema=self._schema).sql(pretty=True)
                return optimized_sql
            except Exception as e:
                logger.warning(f"Failed to optimize SQL query given schema: {e}. Using unoptimized query.")
                return formatted_sql
        return formatted_sql


class SQLSyntaxMatch(Metric, _SQLMetric):
    """
    This metric evaluates the syntactic similarity between the generated SQL query and a set of ground truth queries.
    It uses the sqlglot library to format and compare the SQL queries.
    """

    def __init__(self, optimize: bool = False, schema: Optional[Dict] = None):
        super(SQLSyntaxMatch, self).__init__()
        _SQLMetric.__init__(self, optimize=optimize, schema=schema)

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str]):

        transformed_answer = self._prepare_query(answer)
        transformed_ground_truths = [self._prepare_query(gt) for gt in ground_truth_answers]

        max_match_score = 0.0

        for transformed_gt in transformed_ground_truths:
            match_score = float(transformed_answer == transformed_gt)
            if match_score > max_match_score:
                max_match_score = match_score

        return {"SQL_Syntax_Match": max_match_score}


class SQLASTSimilarity(Metric, _SQLMetric):
    """
    Compare SQL queries using AST similarity, considering different types of changes differently and improving normalization.
    """

    def __init__(
        self,
        optimize: bool = False,
        schema: Optional[Dict] = None,
        diff_weights: ASTDiffWeightConfig = ASTDiffWeightConfig(),
    ):
        super(SQLASTSimilarity, self).__init__()
        _SQLMetric.__init__(self, optimize=optimize, schema=schema)
        self._diff_weights = diff_weights

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str], **kwargs):

        transformed_answer = self._prepare_query(answer)
        transformed_ground_truths = [self._prepare_query(gt) for gt in ground_truth_answers]

        try:
            answer_tree = parse_one(transformed_answer)
            ground_truth_trees = [parse_one(gt) for gt in transformed_ground_truths]
        except Exception:
            return {"SQL_AST_Similarity": -1.0}

        similarity_scores = [
            self._calculate_similarity(answer_tree, ground_truth_tree) for ground_truth_tree in ground_truth_trees
        ]

        return {
            "SQL_AST_Similarity": max(similarity_scores) if similarity_scores else -1.0,
        }

    def _calculate_similarity(self, tree1, tree2):
        diff_result = diff(tree1, tree2)
        total_changes = sum(self._apply_weights(change) for change in diff_result)
        max_nodes = max(len(list(tree1.walk())), len(list(tree2.walk())))
        similarity_score = 1 - (total_changes / max_nodes) if max_nodes > 0 else 1
        return similarity_score

    def _apply_weights(self, change):
        """
        Assign weights to different types of changes based on their expected impact on query semantics.
        """
        if isinstance(change, Keep):
            return self._diff_weights.keep
        elif isinstance(change, Update):
            return self._diff_weights.update
        elif isinstance(change, Insert):
            return self._diff_weights.insert
        elif isinstance(change, Remove):
            return self._diff_weights.remove
        elif isinstance(change, Move):
            return self._diff_weights.move
        else:
            return self._diff_weights.default
