import json
from collections import ChainMap
from copy import deepcopy
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

from continuous_eval.eval.dataset import Dataset
from continuous_eval.eval.modules import AgentModule
from continuous_eval.eval.pipeline import Pipeline
from continuous_eval.eval.utils import instantiate_type

TOOL_PREFIX = "_tool__"


class PipelineResults:
    def __init__(self) -> None:
        self.results: List[Dict] = list()

    @classmethod
    def from_dataset(cls, dataset: Dataset):
        eval_results = cls()
        eval_results.results = dataset.data
        return eval_results

    @classmethod
    def from_logs(cls, logs):
        eval_results = cls()
        for datum in logs.pipeline.dataset.data:
            if datum["uid"] not in logs.data:
                continue
            eval_results.results.append({**datum, **logs.data[datum["uid"]]})
        assert len(eval_results) == len(logs.data), "Could not find some uid in the dataset"
        return eval_results

    def initialize(self, pipeline: Pipeline):
        num_samples = len(pipeline.dataset.data)
        self.results: List[Dict] = [self._build_empty_samples(pipeline) for _ in range(num_samples)]

    def __len__(self):
        return len(self.results)

    def is_empty(self) -> bool:
        return not bool(self.results)

    def _build_empty_samples(self, pipeline: Pipeline):
        if pipeline is None:
            raise ValueError("Pipeline not set")
        empty_samples = dict()
        for module in pipeline.modules:
            empty_samples[module.name] = instantiate_type(module.output)
            if isinstance(module, AgentModule):
                empty_samples[f"{TOOL_PREFIX}{module.name}"] = list()
        return empty_samples

    def save(self, filepath: Path):
        assert filepath.suffix == ".jsonl", "File must be a JSONL file"
        assert self.results, "No samples to save"
        with open(filepath, "w") as f:
            for line in self.results:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")

    def load(self, filepath: Path):
        assert filepath.suffix == ".jsonl", "File must be a JSONL file"
        with open(filepath, "r") as f:
            self.results = [json.loads(line) for line in f]


class MetricsResults:
    def __init__(self, pipeline: Optional[Pipeline]) -> None:
        self.pipeline = pipeline
        self.samples = dict()

    def is_empty(self) -> bool:
        return not bool(self.samples)

    @cached_property
    def results(self) -> Dict:
        """
        Returns a dictionary containing the evaluation results for each module.

        The dictionary is structured as follows:
        - The keys are the names of the modules.
        - The values are lists of dictionaries, where each dictionary represents a result.
        """
        return {
            module_name: [dict(ChainMap(*x)) for x in zip(*eval_res.values())]
            for module_name, eval_res in self.samples.items()
        }

    def to_pandas(self):
        """
        Converts the results to a pandas DataFrame.

        Returns:
            pandas.DataFrame: The results as a pandas DataFrame.
        """
        import pandas as pd

        if len(self.results) > 1:
            flatten = [
                {f"{outer_key}_{key}": value for key, value in inner_dict.items()}
                for outer_key, dict_list in self.results.items()
                for inner_dict in dict_list
            ]
        else:
            flatten = list(*self.results.values())
        return pd.DataFrame(flatten)

    @lru_cache(maxsize=1)
    def aggregate(self):
        """
        Aggregates the metric values for each module and metric name in the samples.

        Returns:
            dict: A dictionary containing the aggregated metric values for each module and metric name.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not set")
        aggregated_samples = dict()
        for module_name, metrics_results in self.samples.items():
            aggregated_samples[module_name] = dict()
            for metric_name, metric_values in metrics_results.items():
                metric = self.pipeline.get_metric(module_name, metric_name)
                aggregated_samples[module_name][metric_name] = metric.aggregate(metric_values)
        actual_results = {
            module_name: dict(ChainMap(*metrics.values())) for module_name, metrics in aggregated_samples.items()
        }
        return actual_results

    def save(self, filepath: Union[str, Path]):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        assert filepath.suffix == ".json", "File must be a JSON file"
        assert self.samples, "No samples to save"
        with open(filepath, "w") as f:
            json.dump(self.samples, f, ensure_ascii=False)

    def load(self, filepath: Union[str, Path]):
        if isinstance(filepath, str):
            filepath = Path(filepath)
        assert filepath.suffix == ".json", "File must be a JSON file"
        with open(filepath, "r") as f:
            self.samples = json.load(f)
        return self


class TestResults:
    def __init__(self) -> None:
        self.results = dict()

    def is_empty(self) -> bool:
        return not bool(self.results)

    def save(self, filepath: Path):
        assert filepath.suffix == ".json", "File must be a JSON file"
        assert self.results, "No samples to save"
        with open(filepath, "w") as f:
            json.dump(self.results, f, ensure_ascii=False)

    def load(self, filepath: Path):
        assert filepath.suffix == ".json", "File must be a JSON file"
        with open(filepath, "r") as f:
            self.results = json.load(f)
