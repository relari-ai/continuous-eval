import random
from dataclasses import dataclass
from pathlib import Path

import dspy
from dspy.datasets import HotpotQA

from continuous_eval.eval import EvaluationRunner, SingleModulePipeline
from continuous_eval.metrics.integrations.dspy import DspyMetricAdapter

# Set up the LM
turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
dspy.settings.configure(lm=turbo)


@dataclass
class HotpotQATone(HotpotQA):
    tone: str = None

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item.tone = random.choice(["positive", "negative"])  # bad, but for testing purposes
        return item


hotpotqa_dataset = HotpotQATone(HotpotQA())


class Tone(dspy.Module):
    def __init__(self):
        super().__init__()

        self._signature = dspy.ChainOfThought(
            prompt_template="""Analyze the tone of the following answer and return if its positive or negative:

Answer:
{answer}

Description:"""
        )

    def forward(self, answer):
        return self._signature(answer=answer)


tone_metric = DspyMetricAdapter(Tone())

pipeline = SingleModulePipeline(
    dataset=hotpotqa_dataset,
    eval=[
        tone_metric.use(
            answer=lambda x: x.answer,
            ground_truth=lambda x: x.tone,
        ),
    ],
)

runner = EvaluationRunner(pipeline)
eval_results = runner.evaluate()
