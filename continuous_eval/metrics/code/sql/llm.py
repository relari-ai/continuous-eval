from pathlib import Path
from typing import Dict, List, Optional, Union

from continuous_eval.metrics.base import (
    Arg,
    Field,
    MetricPrompt,
    response_type,
)
from continuous_eval.metrics.base.probabilistic import (
    DEFAULT_MODEL,
    ProbabilisticMetric,
)

_CWD = Path(__file__).parent


class SQLCorrectness(ProbabilisticMetric):
    """
    The LLM based SQL correctness metric.
    Measures whether the generated SQL query is correct and matches the ground truth SQL query considering natural language exp.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        model: str = DEFAULT_MODEL,
    ):
        prompt = MetricPrompt.from_file(
            system_prompt_path=_CWD / "prompts" / "sql_correctness_sys.jinja2",
            user_prompt_path=_CWD / "prompts" / "sql_correctness_user.jinja2",
            response_format=response_type.Integer(ge=0, le=5),  # type: ignore
            description=None,
        )
        super().__init__(
            name="SQLCorrectness",
            prompt=prompt,
            temperature=temperature,
            model=model,
        )

    def compute(
        self,
        question: str,
        answer: str,
        ground_truth_answers: Union[List[str], str],
        schema: Optional[Dict] = None,
        **kwargs,
    ):
        score = super().compute(
            question=question,
            answer=answer,
            ground_truth_answers=ground_truth_answers,
            schema=schema,
        )
        return {
            "reasoning": score["SQLCorrectness_reasoning"],
            "score": self.prompt.response_format.weighted_score(
                score["SQLCorrectness_probabilities"]
            ),
        }

    @property
    def args(self):
        return {
            "question": Arg(
                type=str, description="The question asked to the system"
            ),
            "answer": Arg(type=str, description="The generated SQL query"),
            "ground_truth_answers": Arg(
                type=Union[str, List[str]],
                description="The ground truth SQL query",
            ),
            "schema": Arg(
                type=Dict,
                description="The schema of the database",
                is_required=True,
            ),
        }

    @property
    def schema(self):
        return {
            "reasoning": Field(
                type=str, description="The reasoning for the correctness score."
            ),
            "score": Field(
                type=float, description="The correctness score.", limits=(0, 1)
            ),
        }
