from typing import Any, Dict

import dspy

from continuous_eval.metrics.base import Metric


class DspyMetricAdapter(Metric):
    def __init__(self, dspy_module: dspy.Module):
        super().__init__()
        self.dspy_module = dspy_module

    def __call__(self, **kwargs) -> Dict[str, Any]:

        dspy_outputs = self.dspy_module.forward(**kwargs)

        metric_outputs = self._convert_outputs(dspy_outputs)

        return metric_outputs

    def _convert_outputs(self, outputs: Any) -> Dict[str, Any]:

        return {"score": outputs}
