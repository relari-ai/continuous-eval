import ast
from typing import List, Union

from munkres import Munkres
from thefuzz import fuzz

from continuous_eval.metrics.base import Metric


class CodeStringMatch(Metric):
    def __call__(self, answer: str, ground_truth_answers: List[str]):
        max_exact_match = 0
        max_similarity_score = 0
        for gt in ground_truth_answers:
            exact_match = float(answer == gt)
            similarity_score = fuzz.ratio(answer, gt) / 100
            if exact_match > max_exact_match:
                max_exact_match = exact_match
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
        return {
            "Exact_Match_Score": max_exact_match,
            "Fuzzy_Match_Score": max_similarity_score,
        }


class PythonASTSimilarity(Metric):
    """
    The following functions are adapted from python-ast-comparison by Pedro Salazar Paredes
    Copyright (c) 2023 Pedro Salazar Paredes
    Licensed under the MIT License
    Source: https://github.com/PedroSalazarParedes/python-ast-comparison
    Modifications: Adjusted to be used in the context of generated code evaluation
    """

    def _compare_ASTs(self, ast_a: ast.AST, ast_b: ast.AST, reorder_depth: int) -> int:
        """Compare two ASTs corresponding to python programs.

        Args:
            ast_a: The first program AST to compare.
            ast_b: The first program AST to compare.
            reorder_depth: The maximum children reorder depth for better
            performance.

        Returns:
            The number of matching nodes in the ASTs.
        """
        children_a = list(ast.iter_child_nodes(ast_a))
        children_b = list(ast.iter_child_nodes(ast_b))
        if (type(ast_a) == type(ast_b)) and len(list(children_a)) == 0 and len(list(children_b)) == 0:
            return 1

        if (type(ast_a) != type(ast_b)) or (len(children_a) != len(children_b)):
            return 0

        if reorder_depth == 0:
            match_index = sum(
                map(
                    lambda pairs: self._compare_ASTs(pairs[0], pairs[1], reorder_depth),
                    zip(children_a, children_b),
                )
            )
            return match_index + 1

        elif reorder_depth > 0:
            match_index = self._reorder_children_compare(ast_a, ast_b, reorder_depth - 1)
            return match_index + 1

        return 0

    def _reorder_children_compare(self, ast_a: ast.AST, ast_b: ast.AST, reorder_depth: int) -> int:
        """Reorders child nodes and compares them.

        Args:
            ast_a: The first AST for child comparison.
            ast_b: The second AST for child comparison.
            reorder_depth: The maximum children reorder depth for better
            performance.

        Returns:
            True if there is a way to match 1-1 every child node of ast_a
            with every child node of ast_b, otherwise False.
        """
        comparison_matrix = []
        cost_matrix = []
        best_match_value = 0
        children_a = list(ast.iter_child_nodes(ast_a))
        children_b = list(ast.iter_child_nodes(ast_b))

        if len(children_a) <= 1 or len(children_b) <= 1:
            for child_a in children_a:
                for child_b in children_b:
                    best_match_value += self._compare_ASTs(child_a, child_b, reorder_depth)
        else:
            for child_a in children_a:
                row = []
                cost_row = []
                for child_b in children_b:
                    similarity = self._compare_ASTs(child_a, child_b, reorder_depth)
                    row.append(similarity)
                    cost_row.append(10000000 - similarity)

                comparison_matrix.append(row)
                cost_matrix.append(cost_row)

            m = Munkres()
            indices = m.compute(cost_matrix)  # type: ignore

            for row, col in indices:
                best_match_value += comparison_matrix[row][col]

        return best_match_value

    def _compare_subtrees(self, sig_subtrees_p1: list, sig_subtrees_p2: list, reorder_depth: int) -> tuple:
        """Compare two significant subtree lists reordering up to a certain depth.

        Args:
            sig_subtrees_p1: The first significant AST list for comparison.
            sig_subtrees_p2: The second significant AST list for comparison.
            reorder_depth: The maximum children reorder depth for better
            performance.

        Returns:
            A tuple with the ratio of matching to non-matching nodes of the
            significant subtrees, and a list with the best matching of subtrees.
        """
        comparison_matrix = []
        cost_matrix = []
        best_match = []
        best_match_value = 0
        best_match_weight = 0
        children_a = sig_subtrees_p1.copy()
        children_b = sig_subtrees_p2.copy()

        if len(children_a) <= 1 or len(children_b) <= 1:
            for child_a in children_a:
                best_match += [child_a]
                for child_b in children_b:
                    best_match_value += self._compare_ASTs(child_a, child_b, reorder_depth)
                    best_match += [child_b]
        else:
            for child_a in children_a:
                row = []
                cost_row = []
                for child_b in children_b:
                    similarity = self._compare_ASTs(child_a, child_b, reorder_depth)
                    row.append(similarity)
                    cost_row.append(10000000 - similarity)

                comparison_matrix.append(row)
                cost_matrix.append(cost_row)

            m = Munkres()
            indices = m.compute(cost_matrix)  # type: ignore

            for row, col in indices:
                best_match_weight += self._apply_weights_to_subtrees_mult(
                    comparison_matrix[row][col],
                    sig_subtrees_p1[row],
                    sig_subtrees_p2[col],
                )
                best_match += [sig_subtrees_p1[row], sig_subtrees_p2[col]]

        all_subtrees_weight = sum(
            map(
                lambda tree: self._apply_weights_to_subtrees(self._get_num_nodes(tree), tree),
                sig_subtrees_p1,
            )
        ) + sum(
            map(
                lambda tree: self._apply_weights_to_subtrees(self._get_num_nodes(tree), tree),
                sig_subtrees_p2,
            )
        )

        similarity = 2 * best_match_weight / all_subtrees_weight

        return round(similarity, 4), best_match

    def _is_significant(self, root: ast.AST) -> bool:
        """Determine if an AST is significant.

        Args:
            root: The AST whose significance we want.

        Returns:
            True for if it is significant, False otherwise.
        """
        return (
            isinstance(root, ast.Import)
            or isinstance(root, ast.FunctionDef)
            or isinstance(root, ast.If)
            or isinstance(root, ast.ClassDef)
            or isinstance(root, ast.While)
            or isinstance(root, ast.For)
            or isinstance(root, ast.comprehension)
            or isinstance(root, ast.Return)
        )

    def _get_significant_subtrees(self, root: ast.AST) -> list:
        """Find the significant subtrees of an AST.

        Args:
            root: The root of the main AST.

        Returns:
            A list with all the significant subtrees of root.
        """
        significant_subtrees = []
        for node in ast.walk(root):
            if self._is_significant(node):
                significant_subtrees.append(node)
        return significant_subtrees

    def _get_num_nodes(self, root: ast.AST) -> int:
        """Find the number of nodes for a given tree.

        Args:
            root: The root of the tree whose size we want.

        Returns:
            The number of nodes in the tree.
        """
        return len(list(ast.walk(root)))

    def _apply_weights_to_subtrees(self, weight: float, subtree: ast.AST) -> float:
        """Apply weights to subtrees according to the time por their roots.

        Args:
            weight: The number of nodes in the subtree.
            subtree: The subtree.

        Returns:
            The weighed weight of the tree.
        """
        new_weight = weight
        if isinstance(subtree, ast.Import):
            new_weight *= 0.3
        elif isinstance(subtree, ast.Module):
            new_weight *= 1
        elif isinstance(subtree, ast.FunctionDef):
            new_weight *= 1.2
        elif isinstance(subtree, ast.If):
            new_weight *= 0.5
        elif isinstance(subtree, ast.ClassDef):
            new_weight *= 1
        elif isinstance(subtree, ast.While):
            new_weight *= 1
        elif isinstance(subtree, ast.For):
            new_weight *= 1
        elif isinstance(subtree, ast.comprehension):
            new_weight *= 1
        elif isinstance(subtree, ast.Return):
            new_weight *= 1
        return new_weight

    def _apply_weights_to_subtrees_mult(self, weight: float, ast_1: ast.AST, ast_2: ast.AST) -> float:
        """Find the average weight of both trees in order to weigh the comparison.

        Args:
            weight: The weight of the comparison.
            ast_1: The first compared tree.
            ast_2: The second compared tree.

        Returns:
            The average of the subtrees' weights.
        """
        if weight == 0:
            return 0
        else:
            return (self._apply_weights_to_subtrees(weight, ast_1) + self._apply_weights_to_subtrees(weight, ast_2)) / 2

    def _compare_many(self, programs: list) -> list:
        """Compare all of the programs in the list.

        Args:
            programs: A list of strings with python programs.

        Returns:
            A matrix with the similarity rating of between all the programs.
        """
        tree_list = list(map(lambda prog: self._get_significant_subtrees(ast.parse(prog)), programs))

        matrix = []
        for program_1_tree_num in range(0, len(tree_list)):
            for program_2_tree_num in range(program_1_tree_num, len(tree_list)):
                if program_1_tree_num == program_2_tree_num:
                    continue

                subtrees1 = tree_list[program_1_tree_num]
                subtrees2 = tree_list[program_2_tree_num]

                result = self._compare_subtrees(subtrees1, subtrees2, 1000)[0]

                matrix.append((program_1_tree_num, program_2_tree_num, result))
                matrix.append((program_2_tree_num, program_1_tree_num, result))

        return matrix

    def __call__(self, answer: str, ground_truth_answers: Union[List[str], str], **kwargs):
        if isinstance(ground_truth_answers, str):
            ground_truth_answers = [ground_truth_answers]
        try:
            answer_tree = ast.parse(answer, mode="exec")
            ground_truth_trees = [ast.parse(gt, mode="exec") for gt in ground_truth_answers]
        except SyntaxError as e:
            return {"Python_AST_Similarity": -1.0}

        answer_subtree = self._get_significant_subtrees(answer_tree)
        ground_truth_subtrees = [
            self._get_significant_subtrees(ground_truth_tree) for ground_truth_tree in ground_truth_trees
        ]

        similarity_scores = [
            self._compare_subtrees(answer_subtree, ground_truth_subtree, 1000)[0]
            for ground_truth_subtree in ground_truth_subtrees
        ]

        return {
            "Python_AST_Similarity": max(similarity_scores),
        }
