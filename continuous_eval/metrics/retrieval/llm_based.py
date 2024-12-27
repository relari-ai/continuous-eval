from pathlib import Path
from typing import List, Union

import numpy as np

from continuous_eval.metrics.base import (
    Arg,
    Field,
    MetricPrompt,
    response_type,
)
from continuous_eval.metrics.base.llm import LLMMetric
from continuous_eval.metrics.base.probabilistic import (
    DEFAULT_MODEL,
    ProbabilisticMetric,
)

_CWD = Path(__file__).parent


class ContextPrecision(ProbabilisticMetric):
    """Calculate the precision of the retrieved context given the ground truth context."""

    def __init__(
        self,
        use_few_shot: bool = True,
        log_relevance_by_context: bool = False,
        temperature=1.0,
        model: str = DEFAULT_MODEL,
    ):
        prompt = MetricPrompt.from_file(
            system_prompt_path=_CWD
            / "prompts"
            / "context_precision_sys.jinja2",
            user_prompt_path=_CWD / "prompts" / "context_precision_user.jinja2",
            response_format=response_type.YesOrNo,  # type: ignore
            description="Calculate the context precision score for the given datum.",
        )
        super().__init__(
            name="ContextPrecision",
            prompt=prompt,
            temperature=temperature,
            model=model,
        )
        self.use_few_shot = use_few_shot
        self.log_relevance_by_context = log_relevance_by_context

    def compute(self, retrieved_context, question, **kwargs):
        """
        Calculate the context precision score for the given datum.
        """
        scores = list()
        for context in retrieved_context:
            score = super().compute(
                question=question,
                context=context,
                use_few_shot=self.use_few_shot,
            )
            scores.append(score["ContextPrecision_probabilities"]["yes"])
        relevant_count = 0
        mAP = 0
        for i, score in enumerate(scores):
            if score > 0.5:
                relevant_count += 1
                mAP += relevant_count / (i + 1)  # Precision at this rank
        if relevant_count > 0:
            mAP /= relevant_count  # Average precision
        else:
            mAP = 0  # No relevant items found
        ret = {
            "percentage_relevant": relevant_count / len(scores),
            "context_precision": sum(scores) / len(scores),
            "context_mean_average_precision": mAP,
        }
        if self.log_relevance_by_context:
            ret["context_relevance_by_context"] = scores
        return ret

    @property
    def args(self):
        return {
            "question": Arg(description="The question to answer."),
            "context": Arg(
                type=List[str],
                description="The context to use for answering the question.",
            ),
            "answer": Arg(type=str, description="The answer to the question."),
        }

    @property
    def schema(self):
        return {
            "context_precision": Field(
                type=float,
                description="The precision of the retrieved context.",
                limits=(0, 1),
            ),
            "context_mean_average_precision": Field(
                type=float,
                description="The mean average precision of the retrieved context.",
                limits=(0, 1),
            ),
            "percentage_relevant": Field(
                type=float,
                description="The percentage of relevant context chunks.",
                limits=(0, 1),
            ),
        }


class ContextCoverage(LLMMetric):
    def __init__(
        self,
        use_few_shot: bool = True,
        temperature=1.0,
        model: str = DEFAULT_MODEL,
    ):
        prompt = MetricPrompt.from_file(
            system_prompt_path=_CWD / "prompts" / "context_coverage_sys.jinja2",
            user_prompt_path=_CWD / "prompts" / "context_coverage_user.jinja2",
            response_format=response_type.JSON(
                [{"reason": str, "statement": str, "attribution": bool}]
            ),  # type: ignore
            description="Calculate the context coverage score for the given datum.",
        )
        super().__init__(
            name="ContextCoverage",
            prompt=prompt,
            temperature=temperature,
            model=model,
        )
        self.use_few_shot = use_few_shot

    def compute(
        self,
        question: str,
        retrieved_context: List[str],
        ground_truth_answers: Union[List[str], str],
        **kwargs,
    ):
        """
        Calculate the context precision score for the given datum.
        """
        if not isinstance(ground_truth_answers, list):
            ground_truth_answers = [ground_truth_answers]
        scores_by_gt_answer = list()
        for gt in ground_truth_answers:
            score = super().compute(
                question=question,
                context=retrieved_context,
                answer=gt,
                use_few_shot=self.use_few_shot,
            )
            scores_by_gt_answer.append(score)
        scores = [
            np.mean([x["attribution"] for x in score])
            for score in scores_by_gt_answer
        ]
        idx = np.argmax(scores)
        return {
            "context_coverage": scores[idx],
            "statements": [
                score["statement"] for score in scores_by_gt_answer[idx]
            ],
        }

    @property
    def args(self):
        return {
            "question": Arg(type=str, is_ground_truth=False),
            "retrieved_context": Arg(type=List[str], is_ground_truth=False),
            "ground_truth_answers": Arg(
                type=Union[List[str], str], is_ground_truth=True
            ),
        }

    @property
    def schema(self):
        return {
            "context_coverage": Field(type=float, limits=(0, 1)),
            "statements": Field(type=List[str]),
        }
