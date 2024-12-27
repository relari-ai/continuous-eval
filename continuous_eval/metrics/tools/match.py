from typing import Any, Dict, List

from continuous_eval.eval.types import ToolCall
from continuous_eval.metrics.base import Arg, Field, Metric


class ToolSelectionAccuracy(Metric):
    """
    Computes the accuracy of tool selection.
    """

    def __init__(self, order_sensitive: bool = False) -> None:
        super().__init__(is_cpu_bound=True)
        self._order_sensitive = order_sensitive

    @property
    def args(self) -> Dict[str, Any]:
        return {
            "answer": Arg(type=List[Dict], is_ground_truth=False),
            "ground_truths": Arg(type=List[Dict], is_ground_truth=True),
        }

    @property
    def schema(self) -> Dict[str, Field]:
        return {
            "score": Field(type=float, limits=(0, 1)),
            "num_correct": Field(type=int),
        }

    def compute(
        self, tools: List[ToolCall], ground_truths: List[ToolCall], **kwargs
    ):
        if self._order_sensitive:
            # When order matters, compare tool executions directly in sequence.
            num_correct = sum(
                1
                for i, tool in enumerate(tools)
                if i < len(ground_truths)
                and tool["name"] == ground_truths[i]["name"]
                and tool["kwargs"] == ground_truths[i]["kwargs"]
            )
        else:
            # Convert ground_truth to a format that's easy to check for "contains"
            use_kwargs = all("kwargs" in tool for tool in ground_truths)
            if use_kwargs:
                ground_truth_set = {
                    frozenset(tool.items())
                    for tool in [
                        {"name": tool["name"], **tool["kwargs"]}
                        for tool in ground_truths
                    ]
                }
            else:
                ground_truth_set = {
                    frozenset(tool.items()) for tool in ground_truths
                }
            # Score
            num_correct, matched_executions = 0, set()
            for tool in tools:
                if use_kwargs:
                    tool_set = frozenset(
                        {"name": tool["name"], **tool["kwargs"]}.items()
                    )
                else:
                    tool_set = frozenset({"name": tool["name"]}.items())
                if (
                    tool_set in ground_truth_set
                    and tool_set not in matched_executions
                ):
                    num_correct += 1
                matched_executions.add(tool_set)

        return {
            "num_correct": num_correct,
            "score": num_correct / len(ground_truths),
        }
