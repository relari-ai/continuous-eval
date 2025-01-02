---
title: Probabilistic LLM Metrics
sidebar:
  order: 20
---
 
Probabilistic LLM metrics are LLM-as-a-Judge metrics that provide score distributions with their associated confidence levels, enabling assessment of model certainty in its evaluations. These distributions are derived from the model's token-level log probabilities.

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

#### Example Output

```json
{
    "PIICheck_score": "yes",
    "PIICheck_reasoning": "The answer contains a full name (John Doe) and a complete address (123 Main Street, Springfield), both of which are considered personally identifiable information (PII). Therefore, it falls under the category of containing sensitive information.",
    "PIICheck_probabilities": {
        "yes": 1.0,
        "no": 0.0
    }
}
```

### Integer Scoring

In this case, the metric returns a probability over integer values. In addition to the score distribution, the metric can output the weighted score directly using `weighted_score` method of `response_format`.

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
            response_format=response_type.Integer(ge=1, le=5),  # Greater than or equal to 1, less than or equal to 5
            description="Evaluate the sentiment of the sentence.",
        )
        super().__init__(name="SentimentAnalysis", prompt=prompt)


metric = SentimentAnalysis()
result = metric(sentence="The product is not bad, but it’s not great either")
print(result)

# Optionally, you can output the weighted score directly, which is normalized based on the upper and lower bounds of the response format
print({"weighted_score": metric.prompt.response_format.weighted_score(result['SentimentAnalysis_probabilities'])})
```

#### Example Output

```json
{
    "SentimentAnalysis_score": 3,
    "SentimentAnalysis_reasoning": "The sentence expresses a moderate view of the product, suggesting it has some positive qualities ('not bad'), but it also indicates a lack of strong enthusiasm or satisfaction ('not great either'). This makes the sentiment neutral, as it neither strongly opposes nor strongly endorses the product.",
    "SentimentAnalysis_probabilities": {
        "1": 0.0,
        "2": 1.250152864552272e-09,
        "3": 0.9999999985326031,
        "4": 2.1724399318911855e-10,
        "5": 0.0
    }
}

# Weighted Score Output
{
    "weighted_score": 0.499999953716648  // Returns a value between 0 and 1
}
```

## Current limitations

1. The `response_format` must be a _single token value_, we predefined a few: `GoodOrBad`, `YesOrNo`, `Boolean` and `Integer`, but it is possible to define your own. In case of integer scoring, negative values are not supported (they are two tokens) as well as values greater than 9.
2. Only OpenAI models are supported for probabilistic metrics.
