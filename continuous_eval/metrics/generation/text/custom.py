from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from continuous_eval.llm_factory import LLMInterface
from continuous_eval.metrics.base import LLMBasedMetric
from continuous_eval.metrics.generation.text.utils import ScoringFunctions


@dataclass
class EvaluationExample:
    input: Union[str, Dict[str, Any]]
    score: Any
    justification: Optional[str] = ""

    def __str__(self):
        in_str = (
            self.input
            if isinstance(self.input, str)
            else "\n".join([f"{k.replace('_', ' ')}: {v}" for k, v in self.input.items()])
        )
        just_str = f"\nJustification: {self.justification}" if self.justification else ""
        return f"Input: {in_str}\nScore: {self.score}{just_str}"

    def todict(self):
        return {k: str(v) for k, v in asdict(self).items()}


class LLMBasedCustomMetric(LLMBasedMetric):
    def __init__(
        self,
        name: str,
        definition: str,
        scoring_rubric: str,
        scoring_function: Callable = ScoringFunctions.Identity,
        model: Optional[LLMInterface] = None,
        model_parameters: Dict[str, Any] = dict(),
        examples: Optional[List[EvaluationExample]] = None,
    ):
        super().__init__(model)
        assert name, "Name is required"
        assert definition, "Definition is required"
        assert scoring_rubric, "Grading prompt is required"
        assert scoring_function is not None, "Scoring function is required"
        self._name = name
        self._definition = definition
        self._scoring_rubric = scoring_rubric
        self._scoring_function = scoring_function
        self._model_parameters = model_parameters
        self._examples = examples

    @property
    def name(self):
        return self._name

    def _build_prompt(self, **kwargs):
        prompt = {"system_prompt": "", "user_prompt": ""}
        prompt[
            "system_prompt"
        ] = "You are are an expert evaluator. The user will provide a description of the criteria and grading instructions, you will apply them with objectivity.\n"
        prompt["user_prompt"] = (
            "CRITERIA: \n" + self._definition + "\n\n" + "GRADING INSTRUCTIONS: \n" + self._scoring_rubric
        )
        if self._examples:
            prompt["user_prompt"] += "\n\nEXAMPLES: \n"
            for example in self._examples:
                prompt["user_prompt"] += str(example)
        prompt["user_prompt"] += "\n\n"
        prompt["user_prompt"] += "Following the instructions, evaluate this:\n"
        for argname, argval in kwargs.items():
            prompt["user_prompt"] += f"{argname}: {argval}\n"
        return prompt

    def __call__(self, **kwargs):
        res = self._llm.run(prompt=self._build_prompt(**kwargs), **self._model_parameters)
        score = self._scoring_function(res)
        return {f"{self.name}_score": score, f"{self.name}_reasoning": res}
