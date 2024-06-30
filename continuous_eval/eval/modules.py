from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from continuous_eval.eval.dataset import DatasetField
from continuous_eval.eval.tests import Test
from continuous_eval.eval.utils import type_hint_to_str
from continuous_eval.metrics import Metric


def _serialize_input_type(obj):
    if isinstance(obj, DatasetField):
        return {"__class__": obj.__class__.__name__, "name": obj.name}
    elif isinstance(obj, Module):
        return {"__class__": obj.__class__.__name__, "name": obj.name}
    elif isinstance(obj, type):
        return type_hint_to_str(obj)
    elif isinstance(obj, (list, tuple)):
        return [_serialize_input_type(x) for x in obj]
    elif obj is None:
        return "None"
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not serializable")


@dataclass(frozen=True, eq=True)
class Tool:
    name: str
    args: Dict[str, Type]
    out_type: Type
    description: Optional[str] = field(default=None)


@dataclass(frozen=True, eq=True)
class Module:
    name: str
    input: Union[DatasetField, "Module", None]
    output: Type
    description: Optional[str] = field(default=None)
    eval: Optional[List[Metric]] = field(default=None)
    tests: Optional[List[Test]] = field(default=None)

    def __post_init__(self):
        if self.name == "":
            raise ValueError(f"Module name cannot be empty")
        if self.tests is not None:
            test_names = {test.name for test in self.tests}
            assert len(test_names) == len(self.tests), f"Each test name must be unique"
        if self.eval is not None:
            eval_names = {metric.name for metric in self.eval}
            assert len(eval_names) == len(self.eval), f"Each metric name must be unique"

    def asdict(self):
        return {
            "name": self.name,
            "input": _serialize_input_type(self.input),
            "output": type_hint_to_str(self.output),
            "description": self.description,
            "eval": [metric.asdict() for metric in self.eval] if self.eval else None,
            "tests": [test.asdict() for test in self.tests] if self.tests else None,
        }


@dataclass(frozen=True, eq=True)
class AgentModule(Module):
    tools: Optional[List[Tool]] = field(default=None)


def SingleModule(
    name: str = "system",
    description: Optional[str] = "",
    eval: Optional[List[Metric]] = None,
    tests: Optional[List[Test]] = None,
) -> Module:
    return Module(
        name=name,
        input=None,
        output=Any,
        description=description,
        eval=eval,
        tests=tests,
    )
