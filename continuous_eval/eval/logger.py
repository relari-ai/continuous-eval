import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from continuous_eval.eval.modules import AgentModule
from continuous_eval.eval.pipeline import Pipeline
from continuous_eval.eval.result_types import TOOL_PREFIX
from continuous_eval.eval.utils import instantiate_type
from continuous_eval.utils.telemetry import telemetry_event

logger = logging.getLogger("eval-manager")
Serializable = Any


class LogMode(Enum):
    APPEND = 0
    REPLACE = 1


class PipelineLogger:
    @telemetry_event("logger")
    def __init__(self, pipeline: Optional[Pipeline] = None):
        self._pipeline: Optional[Pipeline] = pipeline
        self.data = dict()

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        return self._pipeline

    def _empty_sample(self):
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        empty_samples = dict()
        for module in self._pipeline.modules:
            empty_samples[module.name] = instantiate_type(module.output)
            if isinstance(module, AgentModule):
                empty_samples[f"{TOOL_PREFIX}{module.name}"] = list()
        return empty_samples

    def log(
        self,
        uid: Serializable,
        module: str,
        value: Any,
        mode: LogMode = LogMode.REPLACE,
        **kwargs,
    ):
        # Make sure everything looks good
        assert uid is not None, "UID cannot be None"
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if uid not in self.data:
            self.data[uid] = self._empty_sample()
        if kwargs and "tool_args" in kwargs:
            key = f"{TOOL_PREFIX}{module}"
            self.data[uid][key].append({"name": value, "kwargs": kwargs["tool_args"]})
        else:
            if mode == LogMode.REPLACE:
                self.data[uid][module] = value
            elif mode == LogMode.APPEND:
                if not isinstance(self.data[uid][module], list):
                    if isinstance(value, list):
                        self.data[uid][module].extend(value)
                    else:
                        self.data[uid][module].append(value)
                else:
                    self.data[uid][module].add(value)

    def save(self, filepath: Union[str, Path]):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        assert filepath.suffix == ".jsonl", "File must be a JSONL file"
        assert self.data, "No samples to save"
        with open(filepath, "w") as f:
            for uid, res in self.data.items():
                line = {**{"__uid": uid}, **res}
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")

    def load(self, filepath: Union[str, Path]):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        assert filepath.suffix == ".jsonl", "File must be a JSONL file"
        with open(filepath, "r") as f:
            for line in f:
                record = json.loads(line)
                uid = record.pop("__uid")
                self.data[uid] = record
