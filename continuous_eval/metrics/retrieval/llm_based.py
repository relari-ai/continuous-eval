import re
from typing import Optional

from continuous_eval.llm_factory import LLMInterface
from continuous_eval.metrics.base import LLMBasedMetric


class LLMBasedContextPrecision(LLMBasedMetric):
    def __init__(
        self,
        model: Optional[LLMInterface] = None,
        use_few_shot: bool = True,
        log_relevance_by_context: bool = False,
    ):
        super().__init__(model)
        self.use_few_shot = use_few_shot
        self.log_relevance_by_context = log_relevance_by_context

    def __str__(self):
        return f"LLMBasedContextPrecision(model={self._llm}, use_few_shot={self.use_few_shot})"

    def __call__(self, retrieved_context, question, **kwargs):
        """
        Calculate the context precision score for the given datapoint.
        """
        scores = []
        for context in retrieved_context:
            few_shot_prompt = (
                """Example 1:
Question: What is the capital of France?
Context: Paris is the largest city and the capital of France. It has many historical monuments.
Response: Yes
Reasoning: The context states that Paris is the capital of France.
Example 2:
Question: What is the capital of France?
Context: Lyon is a major city in France. It is known for its culinary arts.
Response: No
Reasoning: The context does not mention any city that is the capital of France.
Now evaluate the following:"""
                if self.use_few_shot
                else ""
            )

            prompt = {
                "system_prompt": (
                    """
Given the following question and context, verify if the information in the given context is useful in answering the question. Respond with either Yes or No, followed by reasoning.\n
"""
                    + few_shot_prompt
                ),
                "user_prompt": ("Question: " + question + "\nContext: " + context + "\nResponse:"),
            }

            content = self._llm.run(prompt)
            try:
                score = "yes" in content.lower()
            except Exception:
                score = False
            scores.append(score)

        relevant_chunks = 0
        average_precision = 0
        for i, score in enumerate(scores):
            if score:
                relevant_chunks += 1
                average_precision += relevant_chunks / (i + 1)
        average_precision = average_precision / relevant_chunks if relevant_chunks else 0
        precision = relevant_chunks / len(scores)

        if self.log_relevance_by_context:
            return {
                "LLM_based_context_precision": precision,
                "LLM_based_context_average_precision": average_precision,
                "LLM_based_context_relevance_by_context": scores,
            }
        else:
            return {
                "LLM_based_context_precision": precision,
                "LLM_based_context_average_precision": average_precision,
            }


class LLMBasedContextCoverage(LLMBasedMetric):
    def __init__(self, model: Optional[LLMInterface] = None, use_few_shot: bool = True):
        super().__init__(model)
        self.use_few_shot = use_few_shot

    def __str__(self):
        return f"LLMBasedContextCoverage(model={self._llm}, use_few_shot={self.use_few_shot})"

    def __call__(self, question, retrieved_context, ground_truth_answers, **kwargs):
        """
        Calculate the context coverage score for the given datapoint.
        """
        """
        Calculate the context coverage score for the given datapoint.
        """
        context = "\n".join(retrieved_context)

        few_shot_prompt = (
            """Example:
question: What are the main characteristics of Jupiter?
context: Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined, but less than one-thousandth the mass of the Sun. Jupiter is known for its prominent Great Red Spot, a giant storm larger than Earth that has been ongoing for hundreds of years.
answer: Jupiter is the largest planet in our Solar System and has a giant storm known as the Great Red Spot.
classification:
[
    {{
        "statement_1":"Jupiter is the largest planet in the Solar System.",
        "reason": "This is directly stated in the context.",
        "Attributed": 1
    }},
    {{
        "statement_2":"Jupiter is closer to the Sun than Earth.",
        "reason": "The context contradicts this, stating Jupiter is the fifth planet from the Sun, while Earth is the third.",
        "Attributed": 0
    }}
]"""
            if self.use_few_shot
            else ""
        )

        scores = []
        for gt in ground_truth_answers:
            prompt = {
                "system_prompt": (
                    """
    Given a question, context, and answer, analyze each statement in the answer and classify if the statement can be attributed to the given context or not. Output JSON strictly in the following format.
    """
                    + few_shot_prompt
                ),
                "user_prompt": ("question: " + question + "\ncontext: " + context + "\nanswer: " + gt),
            }

            content = self._llm.run(prompt)

            try:
                coverage = self.extract_attributed_from_broken_json(content)
            except Exception as e:
                print(f"{type(e).__name__} Error: {content}, skipping")
                scores.append(
                    {
                        "LLM_based_context_coverage": -1.0,
                        "LLM_based_context_statements": content,
                    }
                )
            else:
                scores.append(
                    {
                        "LLM_based_context_coverage": coverage,
                        "LLM_based_context_statements": content,
                    }
                )

        return max(scores, key=lambda x: x["LLM_based_context_coverage"])

    @staticmethod
    def extract_attributed_from_broken_json(statements):
        pattern = r'"Attributed":\s*(\d+)'
        attributed_numbers = re.findall(pattern, statements, re.IGNORECASE)
        try:
            attributed_numbers = [int(num) for group in attributed_numbers for num in group if num]
        except Exception as e:
            print(f"{type(e).__name__} Error: {attributed_numbers}, skipping")
            return None
        coverage = sum(attributed_numbers) / len(attributed_numbers) if attributed_numbers else None
        return coverage
