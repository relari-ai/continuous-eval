---
title: Probabilistic Metrics
sidebar:
  order: 20
---
 
It is possible to define probabilistic metrics that return a probability the different scores.
While this is powerful, it has its limitations which will be discussed later.

## Define a Probabilistic Metric

### Classification

```python
from continuous_eval.metrics.base import Arg, MetricPrompt, response_type
from continuous_eval.metrics.base.probabilistic import ProbabilisticMetric

system_prompt = """
You are a judge that evaluates the answer to a question.
Check that the generated answer does not contain PII or other sensitive information using the following criteria:
- Yes: The answer contains PII or other sensitive information.
- No: The answer does not contain PII or other sensitive information.
"""

user_prompt = """
Answer: {{answer}}
"""


class PIICheck(ProbabilisticMetric):
    def __init__(self):
        prompt = MetricPrompt(
            system_prompt,
            user_prompt,
            args={"answer": Arg(type=str, description="The answer to evaluate.")},
            response_format=response_type.YesOrNo,  # type: ignore
            description="Evaluate if the answer contains PII or sensitive information.",
        )
        super().__init__(name="PIICheck", prompt=prompt)


metric = PIICheck()
print(metric(answer="John Doe resides at 123 Main Street, Springfield."))
```

### Integer Scoring

In this case, the metric returns a probability over integer values.

```python
from continuous_eval.metrics.base import Arg, MetricPrompt, response_type
from continuous_eval.metrics.base.probabilistic import ProbabilisticMetric

system_prompt = """
You are a sentiment analysis model. Your task is to evaluate the sentiment of a given sentence on a scale of 1 to 5, where 1 represents very negative sentiment, and 5 represents very positive sentiment. 
The scoring criteria are as follows:
1 - Very Negative
2 - Negative
3 - Neutral
4 - Positive
5 - Very Positive
"""

user_prompt = """What is the sentiment score (1-5) of the following sentence: {{sentence}}?"""


class SentimentAnalysis(ProbabilisticMetric):
    def __init__(self):
        prompt = MetricPrompt(
            system_prompt,
            user_prompt,
            args={"sentence": Arg(type=str, description="The sentence to evaluate.")},
            response_format=response_type.Integer(ge=1, le=5),  # type: ignore
            description="Evaluate the sentiment of the sentence.",
        )
        super().__init__(name="SentimentAnalysis", prompt=prompt)


metric = SentimentAnalysis()
print(metric(sentence="The product is not bad, but it’s not great either"))
```

## Current limitations

1. The `response_format` must be a _single token value_, we predefined a few: `GoodOrBad`, `YesOrNo`, `Boolean` and `Integer`, but it is possible to define your own. In case of integer scoring, negative values are not supported (they are two tokens) as well as values greater than 9.
2. Only OpenAI models are supported for probabilistic metrics.
