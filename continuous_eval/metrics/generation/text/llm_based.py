from pathlib import Path
from typing import List, Union

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


class Faithfulness(ProbabilisticMetric):
    """
    The LLM based faithfulness metric.
    Measures whether the generated answer is faithful to the retrieved context.
    """

    def __init__(
        self,
        use_few_shot: bool = True,
        temperature=1.0,
        model: str = DEFAULT_MODEL,
    ):
        prompt = MetricPrompt.from_file(
            system_prompt_path=_CWD / "prompts" / "faithfulness_sys.jinja2",
            user_prompt_path=_CWD / "prompts" / "faithfulness_user.jinja2",
            response_format=response_type.YesOrNo,  # type: ignore
            description="Calculate the faithfulness score for the given answer.",
        )
        super().__init__(
            name="Faithfulness",
            prompt=prompt,
            temperature=temperature,
            model=model,
        )
        self.use_few_shot = use_few_shot

    def __call__(
        self,
        retrieved_context: List[str],
        answer: str,
        **kwargs,
    ):
        score = super().__call__(
            context=retrieved_context,
            statement=answer,
            use_few_shot=self.use_few_shot,
        )
        return {
            "faithfulness": score["Faithfulness_probabilities"]["yes"],
            "reasoning": score["Faithfulness_reasoning"],
        }

    @property
    def args(self):
        return {
            "retrieved_context": Arg(
                type=List[str],
                description="The context to use for answering the question.",
            ),
            "answer": Arg(type=str, description="The answer to evaluate."),
        }

    @property
    def schema(self):
        return {
            "faithfulness": Field(
                type=float, description="The faithfulness score."
            ),
            "reasoning": Field(
                type=str,
                description="The reasoning for the faithfulness score.",
            ),
        }


class AnswerCorrectness(ProbabilisticMetric):
    """
    The LLM based answer correctness metric.
    Measures whether the generated answer is correct compared to the ground truths.
    """

    def __init__(
        self,
        use_few_shot: bool = True,
        temperature=1.0,
        model: str = DEFAULT_MODEL,
    ):
        prompt = MetricPrompt.from_file(
            system_prompt_path=_CWD / "prompts" / "ans_correctness_sys.jinja2",
            user_prompt_path=_CWD / "prompts" / "ans_correctness_user.jinja2",
            response_format=response_type.Integer(ge=1, le=5),  # type: ignore
            description="Calculate the correctness score for the given answer.",
        )
        super().__init__(
            name="AnswerCorrectness",
            prompt=prompt,
            temperature=temperature,
            model=model,
        )
        self.use_few_shot = use_few_shot

    def compute(
        self,
        question: str,
        answer: str,
        ground_truth_answers: Union[List[str], str],
        **kwargs,
    ):
        if not isinstance(ground_truth_answers, list):
            ground_truth_answers = [ground_truth_answers]
        score = super().compute(
            question=question,
            answer=answer,
            ground_truth_answers=ground_truth_answers,
            use_few_shot=self.use_few_shot,
        )
        return {
            "correctness": self.prompt.response_format.weighted_score(  # type: ignore
                score["AnswerCorrectness_probabilities"]
            ),
            "reasoning": score["AnswerCorrectness_reasoning"],
        }

    @property
    def args(self):
        return {
            "question": Arg(type=str, description="The question to answer."),
            "answer": Arg(type=str, description="The answer to evaluate."),
            "ground_truth_answers": Arg(
                type=Union[List[str], str],
                description="The ground truth answers to compare to.",
            ),
        }

    @property
    def schema(self):
        return {
            "correctness": Field(
                type=float, description="The correctness score.", limits=(0, 1)
            ),
            "reasoning": Field(
                type=str, description="The reasoning for the correctness score."
            ),
        }


class AnswerRelevance(ProbabilisticMetric):
    """
    The LLM based answer relevance metric.
    Measures whether the generated answer is relevant to the question.
    """

    def __init__(
        self,
        use_few_shot: bool = True,
        temperature=1.0,
        model: str = DEFAULT_MODEL,
    ):
        prompt = MetricPrompt.from_file(
            system_prompt_path=_CWD / "prompts" / "ans_relevance_sys.jinja2",
            user_prompt_path=_CWD / "prompts" / "ans_relevance_user.jinja2",
            response_format=response_type.Integer(ge=1, le=3),  # type: ignore
            description="Calculate the relevance score for the given answer.",
        )
        super().__init__(
            name="AnswerRelevance",
            prompt=prompt,
            temperature=temperature,
            model=model,
        )
        self.use_few_shot = use_few_shot

    def __call__(self, question: str, answer: str, **kwargs):
        score = super().__call__(
            question=question,
            answer=answer,
            use_few_shot=self.use_few_shot,
        )
        return {
            "relevance": self.prompt.response_format.weighted_score(  # type: ignore
                score["AnswerRelevance_probabilities"]
            ),
            "reasoning": score["AnswerRelevance_reasoning"],
        }

    @property
    def args(self):
        return {
            "question": Arg(type=str, description="The question to answer."),
            "answer": Arg(type=str, description="The answer to evaluate."),
        }

    @property
    def schema(self):
        return {
            "relevance": Field(
                type=float, description="The relevance score.", limits=(0, 1)
            ),
            "reasoning": Field(
                type=str, description="The reasoning for the relevance score."
            ),
        }


class StyleConsistency(ProbabilisticMetric):
    """
    The LLM based style consistency metric.
    Measures whether the generated answer is consistent in style to the ground truth answer.
    """

    def __init__(
        self,
        use_few_shot: bool = True,
        temperature=1.0,
        model: str = DEFAULT_MODEL,
    ):
        prompt = MetricPrompt.from_file(
            system_prompt_path=_CWD
            / "prompts"
            / "style_consistency_sys.jinja2",
            user_prompt_path=_CWD / "prompts" / "style_consistency_user.jinja2",
            response_format=response_type.Integer(ge=1, le=4),  # type: ignore
            description="Calculate the style consistency score for the given answer.",
        )
        super().__init__(
            name="StyleConsistency",
            prompt=prompt,
            temperature=temperature,
            model=model,
        )
        self.use_few_shot = use_few_shot

    def compute(
        self, answer: str, ground_truth_answers: Union[List[str], str], **kwargs
    ):
        score = super().compute(
            answer=answer,
            ground_truth_answers=ground_truth_answers,
            use_few_shot=self.use_few_shot,
        )
        return {
            "consistency": self.prompt.response_format.weighted_score(  # type: ignore
                score["StyleConsistency_probabilities"]
            ),
            "reasoning": score["StyleConsistency_reasoning"],
        }

    @property
    def args(self):
        return {
            "answer": Arg(type=str, description="The answer to evaluate."),
            "ground_truth_answers": Arg(
                type=Union[List[str], str],
                description="The ground truth answers to compare to.",
            ),
        }

    @property
    def schema(self):
        return {
            "consistency": Field(
                type=float, description="The consistency score.", limits=(0, 1)
            ),
            "reasoning": Field(
                type=str, description="The reasoning for the consistency score."
            ),
        }
