import warnings
from enum import Enum
from typing import Any, List, Optional

from loguru import logger

from continuous_eval.eval.dataset import Dataset, DatasetField
from continuous_eval.eval.pipeline import CalledTools, ModuleOutput, Pipeline
from continuous_eval.eval.result_types import TOOL_PREFIX, EvaluationResults, MetricsResults, TestResults


class LogMode(Enum):
    APPEND = 0
    REPLACE = 1


class EvaluationManager:
    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._eval_results: EvaluationResults = EvaluationResults()
        self._metrics_results: MetricsResults = MetricsResults()
        self._test_results: TestResults = TestResults()
        self._is_running: bool = False

        self._idx = 0

    @property
    def is_complete(self) -> bool:
        if self._pipeline is None:
            return False
        return self._idx == len(self._pipeline.dataset.data)

    @property
    def samples(self) -> List[dict]:
        return self._eval_results.results

    @property
    def evaluation(self) -> EvaluationResults:
        return self._eval_results

    @property
    def metrics(self) -> MetricsResults:
        return self._metrics_results

    @property
    def tests(self) -> TestResults:
        return self._test_results

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        return self._pipeline

    @property
    def dataset(self) -> Dataset:
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if self._pipeline.dataset is None:
            raise ValueError("Dataset not set")
        return self._pipeline.dataset

    def set_pipeline(self, pipeline: Pipeline):
        self._metrics_results.pipeline = pipeline
        self._pipeline = pipeline

    def is_running(self) -> bool:
        return self._is_running

    def start_run(self):
        self._idx = 0
        self._is_running = True
        self._eval_results = EvaluationResults(self._pipeline)

    @property
    def curr_sample(self):
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if self._idx >= len(self.dataset.data):
            return None
        return self.dataset.data[self._idx]

    def next_sample(self):
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if self._idx >= len(self.dataset.data):
            self._is_running = False
        else:
            self._idx += 1
        return self.curr_sample

    # Logging results

    def log(
        self,
        module: str,
        value: Any,
        mode: LogMode = LogMode.REPLACE,
        **kwargs,
    ):
        # Make sure everything looks good
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if not self._is_running:
            raise ValueError("Cannot log when not running")
        if module not in self._eval_results.results[self._idx]:
            raise ValueError(f"module {module} not found, review your pipeline")

        if kwargs and "tool_args" in kwargs:
            key = f"{TOOL_PREFIX}{module}"
            self._eval_results.results[self._idx][key].append({"name": value, "kwargs": kwargs["tool_args"]})
        else:
            if mode == LogMode.REPLACE:
                self._eval_results.results[self._idx][module] = value
            elif mode == LogMode.APPEND:
                if not isinstance(self._eval_results.results[self._idx][module], list):
                    if isinstance(value, list):
                        self._eval_results.results[self._idx][module].extend(value)
                    else:
                        self._eval_results.results[self._idx][module].append(value)
                else:
                    self._eval_results.results[self._idx][module].add(value)

    # Evaluate

    def _prepare(self, module, metric):
        kwargs = dict()
        if metric.overloaded_params is not None:
            for key, val in metric.overloaded_params.items():
                if isinstance(val, DatasetField):
                    kwargs[key] = [x[val.name] for x in self.dataset.data]  # type: ignore
                elif isinstance(val, ModuleOutput):
                    module_name = module.name if val.module is None else val.module.name
                    kwargs[key] = [val(x[module_name]) for x in self._eval_results.results]
                elif isinstance(val, CalledTools):
                    module_name = module.name if val.module is None else val.module.name
                    val_key = f"{TOOL_PREFIX}{module_name}"
                    kwargs[key] = [val(x[val_key]) for x in self._eval_results.results]
                else:
                    raise ValueError(f"Invalid promised parameter {key}={val}")
        return kwargs

    def run_metrics(self):
        logger.info("Running evaluation")
        assert self._pipeline is not None, "Pipeline not set"
        assert len(self._eval_results.results) > 0, "No evaluation samples to run the metrics on"
        if len(self._eval_results.results) != len(self.dataset.data):
            warnings.warn("The number of samples does not match the dataset size")
        self._metrics_results.samples = {
            module.name: {metric.name: metric.batch(**self._prepare(module, metric)) for metric in module.eval}
            for module in self._pipeline.modules
            if module.eval is not None
        }
        return self._metrics_results

    def aggregate_eval_results(self):
        assert self._pipeline is not None, "Pipeline not set"
        assert self._eval_results is not None, "Evaluation results not set"
        assert all(
            [len(module_res) == len(self.dataset.data) for module_res in self._metrics_results.results.values()]
        ), "Evaluation is not complete"
        return self._metrics_results.results

    # Tests

    def run_tests(self):
        logger.info("Running tests")
        assert self._pipeline is not None, "Pipeline not set"
        assert not self._eval_results.is_empty(), "Evaluation results not set"
        assert all(
            [len(module_res) == len(self.dataset.data) for module_res in self._metrics_results.results.values()]
        ), "Evaluation is not complete"
        self._test_results.results = {
            module.name: {test.name: test.run(self._metrics_results.results[module.name]) for test in module.tests}
            for module in self._pipeline.modules
            if module.tests is not None
        }
        return self._test_results

    def test_graph(self):
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if self._test_results is None:
            raise ValueError("Tests not run")
        pipeline_graph = self._pipeline.graph_repr()
        tests = "\n    %% Tests\n"
        for module, results in self._test_results.results.items():
            metrics_lines = []
            for metric, passed in results.items():
                status = "Pass" if passed else "Fail"
                metrics_lines.append(f"<i>{metric}: {status}</i>")
            metrics = "<br>".join(metrics_lines)
            tests += f"    {module}[<b>{module}</b><br>---<br>{metrics}]\n"

        return pipeline_graph + tests


eval_manager = EvaluationManager()
