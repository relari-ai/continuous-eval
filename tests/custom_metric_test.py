from continuous_eval.metrics.base.metric import Arg, Field
from continuous_eval.metrics.custom import CustomMetric, Example
from tests.helpers.utils import validate_args, validate_schema


def test_llm_based_metric():
    metric = CustomMetric(
        name="ProfessionalTone",
        criteria="Professional tone in the answer",
        rubric="""1: The answer is not professional.
2: The answer is somewhat professional.
3: The answer is very professional.""",
        arguments={
            "answer": Arg(type=str, description="The answer to evaluate."),
            "context": Arg(type=str, description="The context of the answer."),
        },
        response_format={
            "explanation": Field(
                type=str,
                description="The explanation for the score given to the answer",
            ),
            "score": Field(
                type=int, description="The score of the answer between 1 and 3"
            ),
        },
        examples=[
            Example(
                input={
                    "answer": "Hello, you good toray?",
                    "context": "Toray means today in Japanese.",
                },
                output={
                    "explanation": "The answer is not professional.",
                    "score": 1,
                },
            ),
            Example(
                input={
                    "answer": "Good morning, how are you today?",
                    "context": "None",
                },
                output={
                    "explanation": "The answer is somewhat professional.",
                    "score": 2,
                },
            ),
        ],
    )
    ret = metric(
        answer="Hello, you good toray?",
        context="Toray means today in Japanese.",
    )
    assert validate_schema(metric.schema, ret)
    assert metric.help is not None
    assert validate_args(metric.args)
