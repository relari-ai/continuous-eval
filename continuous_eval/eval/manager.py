import json
from collections import ChainMap
from pathlib import Path
from typing import Type, get_origin

from loguru import logger

from continuous_eval.eval.dataset import Dataset, DatasetField
from continuous_eval.eval.pipeline import ModuleOutput, Pipeline


def _instantiate_type(type_hint: Type):
    origin = get_origin(type_hint)
    # If the origin is None, it means type_hint is not a generic type
    # and we assume type_hint itself is directly instantiable
    if origin is None:
        origin = type_hint
    try:
        # This only works for types without required arguments in their __init__.
        instance = origin()
    except TypeError as e:
        # If instantiation fails, return an error message or raise a custom exception
        instance = None
    return instance


class EvaluationManager:
    def __init__(self):
        self._pipeline = None
        self._dataset = None
        self._samples = None
        self._eval_results = None
        self._test_results = None
        self._is_running = False

        self._idx = 0

    def _build_empty_samples(self):
        assert self.pipeline is not None, "Pipeline not set"
        empty_samples = dict()
        for module in self.pipeline.modules:
            empty_samples[module.name] = _instantiate_type(module.output)
        return empty_samples

    @property
    def is_complete(self):
        return self._idx == len(self._dataset)

    @property
    def samples(self):
        return self._samples

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def test_results(self):
        return self._test_results

    @property
    def eval_results(self):
        return self._eval_results

    def set_pipeline(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self._dataset = pipeline.dataset

    def is_running(self) -> bool:
        return self.is_running

    def start_run(self):
        self._idx = 0
        self._is_running = True
        self._samples = [self._build_empty_samples() for _ in range(len(self._dataset.data))]

    @property
    def curr_sample(self):
        if self._idx >= len(self._dataset.data):
            return None
        return self._dataset.data[self._idx]

    def next_sample(self):
        if self._idx >= len(self._dataset.data):
            self._is_running = False
        else:
            self._idx += 1
        return self.curr_sample

    def log(self, key, value):
        assert type(value) == get_origin(self._pipeline.module_by_name(key).output) or isinstance(
            value, self._pipeline.module_by_name(key).output
        ), f"Value {value} does not match expected type in the pipeline"
        if not self._is_running:
            raise ValueError("Cannot log when not running")
        if key not in self._samples[self._idx]:
            raise ValueError(f"Key {key} not found, review your pipeline")
        if isinstance(self._samples[self._idx][key], list) and not isinstance(value, list):
            self._samples[self._idx][key] = value
        elif isinstance(self._samples[self._idx][key], dict):
            self._samples[self._idx][key].update(value)
        elif isinstance(self._samples[self._idx][key], set) and not isinstance(value, list):
            self._samples[self._idx][key] = value
        else:
            self._samples[self._idx][key] = value

    def save_results(self, filepath: Path):
        assert self._samples is not None, "No samples to save"
        assert self._dataset is not None, "Dataset not set"
        assert len(self._samples) == len(self._dataset.data), "Samples not complete"
        assert filepath.suffix == ".jsonl", "File must be a JSONL file"
        # Save samples to file (JSONL)
        with open(filepath, "w") as f:
            for line in self._samples:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")

    def load_results(self, filepath: Path):
        assert filepath.suffix == ".jsonl", "File must be a JSONL file"
        assert self._dataset is not None, "Dataset not set"
        # Load samples from file (JSONL)
        with open(filepath, "r") as f:
            self._samples = [json.loads(line) for line in f]
        assert len(self._samples) == len(self._dataset.data), "Samples not complete"

    # Evaluate

    def _prepare(self, module, metric):
        kwargs = dict()
        if metric.overloaded_params is not None:
            for key, val in metric.overloaded_params.items():
                if isinstance(val, DatasetField):
                    kwargs[key] = [x[val.name] for x in self._dataset.data]
                elif isinstance(val, ModuleOutput):
                    module_name = module.name if val.module is None else val.module.name
                    kwargs[key] = [val(x[module_name]) for x in self._samples]
                else:
                    raise ValueError(f"Invalid promised parameter {key}={val}")
        return kwargs

    def run_eval(self):
        logger.info("Running evaluation")
        assert self._pipeline is not None, "Pipeline not set"
        assert self._dataset is not None, "Dataset not set"
        assert self._samples is not None, "Samples not set"
        assert len(self._samples) == len(self._dataset.data), "Samples not complete"
        evaluation_results = {
            module.name: {metric.name: metric.batch(**self._prepare(module, metric)) for metric in module.eval}
            for module in self._pipeline.modules
            if module.eval is not None
        }
        results = {
            module_name: [dict(ChainMap(*x)) for x in zip(*eval_res.values())]
            for module_name, eval_res in evaluation_results.items()
        }
        self._eval_results = results
        return self._eval_results

    def save_eval_results(self, filepath: Path):
        assert filepath.suffix == ".json", "File must be a JSON file"
        assert self._eval_results is not None, "No samples to save"
        assert self._dataset is not None, "Dataset not set"
        assert all(
            [len(module_res) == len(self._dataset.data) for module_res in self._eval_results.values()]
        ), "Evaluation is not complete"
        with open(filepath, "w") as json_file:
            json.dump(self._eval_results, json_file, indent=None)

    def load_eval_results(self, filepath: Path):
        assert filepath.suffix == ".json", "File must be a JSON file"
        assert self._dataset is not None, "Dataset not set"
        with open(filepath, "r") as json_file:
            self._eval_results = json.load(json_file)
        assert all(
            [len(module_res) == len(self._dataset.data) for module_res in self._eval_results.values()]
        ), "Evaluation is not complete"

    # Tests

    def run_tests(self):
        logger.info("Running tests")
        assert self._pipeline is not None, "Pipeline not set"
        assert self._dataset is not None, "Dataset not set"
        assert self._eval_results is not None, "Evaluation results not set"
        assert all(
            [len(module_res) == len(self._dataset.data) for module_res in self._eval_results.values()]
        ), "Evaluation is not complete"
        self._test_results = {
            module.name: {test.name: test.run(self._eval_results[module.name]) for test in module.tests}
            for module in self._pipeline.modules
            if module.tests is not None
        }
        return self._test_results

    def save_test_results(self, filepath: Path):
        assert filepath.suffix == ".json", "File must be a JSON file"
        assert self._test_results is not None, "No samples to save"
        with open(filepath, "w") as json_file:
            json.dump(self._test_results, json_file, indent=None)

    def load_test_results(self, filepath: Path):
        assert filepath.suffix == ".json", "File must be a JSON file"
        with open(filepath, "r") as json_file:
            self._test_results = json.load(json_file)

    def test_graph(self):
        pipeline_graph = self._pipeline.graph_repr()
        tests = "\n    %% Tests\n"
        for module, results in self._test_results.items():
            metrics_lines = []
            for metric, passed in results.items():
                status = "Pass" if passed else "Fail"
                metrics_lines.append(f"<i>{metric}: {status}</i>")
            metrics = "<br>".join(metrics_lines)
            tests += f"    {module}[<b>{module}</b><br>---<br>{metrics}]\n"

        return pipeline_graph + tests


eval_manager = EvaluationManager()
