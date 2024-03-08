import concurrent.futures
import logging
import warnings
from enum import Enum
from typing import Any, List, Optional

from continuous_eval.eval.dataset import Dataset, DatasetField
from continuous_eval.eval.pipeline import CalledTools, ModuleOutput, Pipeline
from continuous_eval.eval.result_types import TOOL_PREFIX, EvaluationResults, MetricsResults, TestResults
from continuous_eval.utils.telemetry import telemetry_event

logger = logging.getLogger("eval-manager")


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
        self._metadata = dict()

        self._idx = 0

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
    def metadata(self) -> dict:
        return self._metadata

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

    def set_metadata(self, metadata: dict):
        self._metadata = metadata

    def is_running(self) -> bool:
        return self._is_running

    @telemetry_event("eval_manager")
    def start_run(self, metadata: dict = dict()):
        self._idx = 0
        self._is_running = True
        self._eval_results = EvaluationResults(self._pipeline)
        self.set_metadata(metadata)

    @property
    def curr_sample(self):
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if self._idx >= len(self.dataset.data):
            self._is_running = False
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

    # Context manager
    @property
    def experiment(self):
        class ExperimentContext:
            def __init__(self, manager):
                self._manager = manager

            def __enter__(self):
                # Initialize the session
                self._manager.start_run()
                return self  # Return the session object itself to be used as an iterator

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Clean up the session
                self._manager._is_running = False
                # Optional: Handle exceptions
                if exc_type:
                    print(f"An error occurred: {exc_val}")
                return False  # Propagate exceptions

            def __iter__(self):
                while self._manager.is_running() and self._manager.curr_sample is not None:
                    yield self._manager.curr_sample
                    self._manager.next_sample()

        return ExperimentContext(self)

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

    @telemetry_event("eval_manager")
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
        # metrics_results = dict()
        # # Use ProcessPoolExecutor to parallelize computation
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # Prepare tasks for modules that have eval
        #     tasks = [executor.submit(self.compute_metric_results, module)
        #             for module in self._pipeline.modules if module.eval is not None]
        #     # Wait for all tasks to complete and collect results
        #     for future in concurrent.futures.as_completed(tasks):
        #         module_name, results = future.result()
        #         metrics_results[module_name] = results
        # self._metrics_results.samples = metrics_results
        return self._metrics_results

    def aggregate_eval_results(self):
        assert self._pipeline is not None, "Pipeline not set"
        assert self._eval_results is not None, "Evaluation results not set"
        assert all(
            [len(module_res) == len(self.dataset.data) for module_res in self._metrics_results.results.values()]
        ), "Evaluation is not complete"
        return self._metrics_results.results

    # Tests
    @telemetry_event("eval_manager")
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
