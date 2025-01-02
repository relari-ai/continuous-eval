from pathlib import Path
from typing import Dict, Optional, Union

from jinja2 import BaseLoader, Environment, meta

from continuous_eval.metrics.base import Arg
from continuous_eval.metrics.base.response_type import (
    ResponseFormat,
    ScoringFunction,
    get_response_format,
)


class PromptTemplate:
    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        args: Optional[Dict[str, Arg]] = None,
    ):
        self._env = Environment(loader=BaseLoader())

        self._raw_system_prompt = system_prompt
        self._raw_user_prompt = user_prompt
        self._sys_prompt_template = self._env.from_string(
            self._raw_system_prompt
        )
        self._user_prompt_template = self._env.from_string(
            self._raw_user_prompt
        )
        self._vars = self._get_vars(self._raw_user_prompt) | self._get_vars(
            self._raw_system_prompt
        )
        self._args = args or {var: Arg() for var in self._vars}
        self._validate()

    def serialize(self):
        return {
            "system_prompt": self._raw_system_prompt,
            "user_prompt": {
                "format": "jinja",
                "template": self._raw_user_prompt,
            },
            "args": {k: v.to_dict() for k, v in self._args.items()},
        }

    def __getstate__(self) -> object:
        return self.serialize()

    def __setstate__(self, state: Dict):
        self.__init__(
            state["system_prompt"],
            state["user_prompt"]["template"],
            {k: Arg.from_dict(v) for k, v in state["args"].items()},
        )

    @classmethod
    def deserialize(cls, data: Dict):
        if data["user_prompt"]["format"] != "jinja":
            raise ValueError("Only jinja prompts are supported")
        if "template" not in data["user_prompt"]:
            raise ValueError("User prompt must contain a 'template'")
        args = None
        if "args" in data:
            args = {k: Arg.from_dict(v) for k, v in data["args"].items()}
        return cls(
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"]["template"],
            args=args,
        )

    def _validate(self):
        if not self._raw_system_prompt:
            raise ValueError("System prompt is required")
        if not self._raw_user_prompt:
            raise ValueError("User prompt is required")
        delta_var = self._vars - set(self._args)
        if delta_var:
            raise ValueError(f"Missing arguments in prompt: {delta_var}")
        delta_var = set(self._args) - self._vars
        if delta_var:
            raise ValueError(f"Extra arguments in prompt: {delta_var}")

    def _get_vars(self, prompt: str):
        ast = self._env.parse(prompt)
        return meta.find_undeclared_variables(ast)

    def system_prompt(self, **kwargs):
        return self._sys_prompt_template.render(**kwargs)

    def user_prompt(self, **kwargs):
        return self._user_prompt_template.render(**kwargs)

    def render(self, **kwargs):
        return {
            "system_prompt": self.system_prompt(**kwargs),
            "user_prompt": self.user_prompt(**kwargs),
        }

    @property
    def args(self):
        return self._args

    def get_identifiers(self):
        return set(self._args.keys())

    @classmethod
    def from_file(
        cls,
        system_prompt_path: Path,
        user_prompt_path: Path,
        args: Optional[Dict[str, Arg]] = None,
    ):
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        with open(user_prompt_path, "r") as f:
            user_prompt = f.read()
        return cls(system_prompt, user_prompt, args)


class MetricPrompt(PromptTemplate):
    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Union[ResponseFormat, ScoringFunction],
        description: Optional[str] = None,
        args: Optional[Dict[str, Arg]] = None,
    ):
        """
        A prompt for a custom metric.

        Args:
            system_prompt (str): The system prompt to be used.
            user_prompt (str): The user prompt to be used (supports jinja2 templating).
            response_format (Union[ResponseFormat, ScoringFunction]): The expected response format or scoring function.
            description (str, optional): A description of the prompt. Defaults to "No description provided".
            args (Dict[str, Arg], optional): A dictionary of argument properties. Defaults to None.
        """
        super().__init__(system_prompt, user_prompt, args)
        self.response_format = response_format
        self.description = description

    @classmethod
    def from_file(
        cls,
        system_prompt_path: Path,
        user_prompt_path: Path,
        response_format: Union[ResponseFormat, ScoringFunction],
        description: Optional[str] = None,
        args: Optional[Dict[str, Arg]] = None,
    ):
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        with open(user_prompt_path, "r") as f:
            user_prompt = f.read()
        return cls(
            system_prompt, user_prompt, response_format, description, args
        )

    def serialize(self):
        return {
            **super().serialize(),
            "response_format": self.response_format.serialize(),
            "description": self.description,
        }

    @classmethod
    def deserialize(cls, data: Dict):
        if data["user_prompt"]["format"] != "jinja":
            raise ValueError("Only jinja prompts are supported")
        if "template" not in data["user_prompt"]:
            raise ValueError("User prompt must contain a 'template'")
        args = None
        if "args" in data:
            args = {k: Arg.from_dict(v) for k, v in data["args"].items()}
        return cls(
            system_prompt=data["system_prompt"],
            user_prompt=data["user_prompt"]["template"],
            response_format=get_response_format(data["response_format"]),  # type: ignore
            description=data.get("description", ""),
            args=args,
        )
