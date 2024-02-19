from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

from continuous_eval.eval.dataset import DatasetField
from continuous_eval.eval.tests import Test
from continuous_eval.metrics import Metric


@dataclass(frozen=True, eq=True)
class Tool:
    name: str
    args: Dict[str, Type]
    out_type: Type
    description: Optional[str] = field(default=None)


@dataclass(frozen=True, eq=True)
class Module:
    name: str
    input: Union[DatasetField, Type, "Module"]
    output: Type
    description: Optional[str] = field(default=None)
    eval: Optional[List[Metric]] = field(default=None)
    tests: Optional[List[Test]] = field(default=None)

    def __post_init__(self):
        if self.tests is not None:
            test_names = {test.name for test in self.tests}
            assert len(test_names) == len(self.tests), f"Each test name must be unique"

        if self.eval is not None:
            eval_names = {metric.name for metric in self.eval}
            assert len(eval_names) == len(self.eval), f"Each metric name must be unique"


@dataclass(frozen=True, eq=True)
class AgentModule(Module):
    tools: Optional[List[Tool]] = field(default=None)
    reference_tool_calls: Optional[DatasetField] = field(default=None)
    is_recursive: bool = field(default=False)
