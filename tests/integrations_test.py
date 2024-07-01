import random
from dataclasses import dataclass

import dspy

# Set up the LM
from dotenv import load_dotenv
from dspy.datasets import HotpotQA

from continuous_eval.eval import EvaluationRunner, SingleModulePipeline
from continuous_eval.metrics.integrations.dspy import DspyMetricAdapter

load_dotenv()

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


class ToneSignature(dspy.Signature):
    (
        """You are given some a question and an answer."""
        """You must indicate with positive/negative tone whether it was kind (positive)"""
        """ or not (negative) """
    )

    question = dspy.InputField()
    answer = dspy.InputField()
    tone = dspy.OutputField(desc="Positive or Negative")


class Tone(dspy.Module):
    def __init__(self):
        super().__init__()

        self.generate_tone = dspy.ChainOfThought(ToneSignature)

    def forward(self, question, answer):
        return self.generate_tone(question=question, answer=answer)


tone_metric = DspyMetricAdapter(Tone())

pipeline = SingleModulePipeline(
    dataset=hotpotqa_dataset,
    eval=[
        tone_metric.use(
            question=lambda x: x.question,
            answer=lambda x: x.answer,
            ground_truth=lambda x: x.tone,
        ),
    ],
)

runner = EvaluationRunner(pipeline)
eval_results = runner.evaluate()
