import logging
from typing import Optional, Union

from continuous_eval.eval.dataset import Dataset, DatasetField, LambdaField
from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.modules import Module, TOOL_PREFIX
from continuous_eval.eval.pipeline import CalledTools, ModuleOutput, Pipeline
from continuous_eval.eval.result_types import (
    MetricsResults,
    PipelineResults,
    TestResults,
)
from continuous_eval.metrics import Metric
from continuous_eval.utils.telemetry import telemetry_event
from copy import deepcopy

logger = logging.getLogger("eval-manager")


class EvaluationRunner:
    def __init__(self, pipeline: Pipeline):
        assert isinstance(pipeline, Pipeline), "Pipeline not set"
        self._pipeline = pipeline

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def dataset(self) -> Dataset:
        if self._pipeline is None:
            raise ValueError("Pipeline not set")
        if self._pipeline.dataset is None:
            raise ValueError("Dataset not set")
        return self._pipeline.dataset

    # Evaluate
    @staticmethod
    def prepare(
        dataset: Dataset,
        eval_results: PipelineResults,
        module: Module,
        metric: Metric,
    ):
        kwargs = dict()
        if metric.overloaded_params is not None:
            for key, val in metric.overloaded_params.items():
                if key == "uid":
                    continue
                if isinstance(val, DatasetField):
                    try:
                        kwargs[key] = [
                            x[module.name][val.name] for x in dataset.data
                        ]  # type: ignore
                    except Exception:
                        kwargs[key] = [x[val.name] for x in dataset.data]  # type: ignore
                elif isinstance(val, LambdaField):
                    kwargs[key] = list()
                    for rx in eval_results.results:
                        uid = rx["uid"]
                        if module.name in rx:
                            kwargs[key].append(val.func(rx[module.name]))
                        else:
                            for x in dataset.data:
                                if x["uid"] == uid:
                                    kwargs[key].append(val.func(x))
                                    break
                elif isinstance(val, ModuleOutput):
                    module_name = getattr(
                        val.module,
                        "name",
                        val.module
                        if isinstance(val.module, str)
                        else module.name,
                    )
                    if module_name is None:
                        raise ValueError(
                            f"Invalid promised parameter {key}={val}"
                        )
                    try:
                        kwargs[key] = [
                            val(x[module_name]) for x in eval_results.results
                        ]
                    except Exception as e:
                        print(f"Oh! {e}")
                elif isinstance(val, CalledTools):
                    module_name = getattr(
                        val.module,
                        "name",
                        val.module
                        if isinstance(val.module, str)
                        else module.name,
                    )
                    if module_name is None:
                        raise ValueError(
                            f"Invalid promised parameter {key}={val}"
                        )
                    val_key = f"{TOOL_PREFIX}{module_name}"
                    kwargs[key] = [
                        val(x[val_key]) for x in eval_results.results
                    ]
                else:
                    raise ValueError(f"Invalid promised parameter {key}={val}")
            return kwargs
        else:
            for item in eval_results.results:
                itr = item[module.name] if module.name in item else item
                for key, value in itr.items():
                    if key not in kwargs:
                        kwargs[key] = []
                    kwargs[key].append(value)
        return kwargs

    @telemetry_event(name="EvaluationRunner.evaluate")
    def evaluate(
        self,
        data: Optional[Union[PipelineResults, PipelineLogger, Dataset]] = None,
    ) -> MetricsResults:
        logger.info("Running evaluation")
        if data is None:
            eval_results = PipelineResults.from_dataset(self.dataset)
        elif isinstance(data, PipelineResults):
            eval_results = deepcopy(data)
        elif isinstance(data, Dataset):
            eval_results = PipelineResults.from_dataset(data)
        elif isinstance(data, PipelineLogger):
            eval_results = PipelineResults.from_logs(data)
        else:
            raise ValueError("Invalid data type")
        assert self._pipeline is not None, "Pipeline not set"
        assert (
            len(eval_results.results) > 0
        ), "No evaluation samples to run the metrics on"
        metrics_results = MetricsResults(self.pipeline)
        metrics_results.samples = {
            module.name: {
                metric.name: metric.batch(
                    **self.prepare(self.dataset, eval_results, module, metric)
                )
                for metric in module.eval
            }
            for module in self._pipeline.modules
            if module.eval is not None
        }
        return metrics_results

    @telemetry_event(name="EvaluationRunner.evaluate")
    def test(self, metrics: MetricsResults) -> TestResults:
        logger.info("Running tests")
        test_results = TestResults()
        test_results.results = {
            module.name: {
                test.name: test.run(metrics.results[module.name])
                for test in module.tests
            }
            for module in self._pipeline.modules
            if module.tests is not None
        }
        return test_results
