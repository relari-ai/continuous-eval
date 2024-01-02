from continuous_eval.llm_factory import DefaultLLM, LLMInterface
from continuous_eval.metrics.base import LLMBasedMetric
from continuous_eval.metrics.retrieval_LLM_based_metrics import LLMBasedContextCoverage


class LLMBasedFaithfulness(LLMBasedMetric):
    """
    The LLM based faithfulness metric.
    Measures whether the generated answer is faithful to the retrieved context.
    """

    def __init__(
        self,
        model: LLMInterface = DefaultLLM,
        use_few_shot: bool = True,
        classify_by_statement: bool = False,
    ):
        super().__init__(model)
        self.use_few_shot = use_few_shot
        self.classify_by_statement = classify_by_statement

    def __str__(self):
        return f"LLMBasedFaithfulness(model={self.model}, use_few_shot={self.use_few_shot}, classify_by_statement={self.classify_by_statement})"

    def calculate(self, question, retrieved_contexts, answer, **kwargs):
        """
        Calculate the faithfulness score for the given datapoint.
        """
        if self.classify_by_statement:
            # Context coverage uses the same prompt as faithfulness because it calculates how what proportion statements in the answer can be attributed to the context.
            # The difference is that faithfulness uses the generated answer, while context coverage uses ground truth answer (to evaluate context).
            context_coverage = LLMBasedContextCoverage(use_few_shot=self.use_few_shot)
            results = context_coverage.calculate(question, retrieved_contexts, answer)
            score = results["LLM_based_context_coverage"]
            reasoning = results["LLM_based_context_statements"]
        else:
            context = "\n".join(retrieved_contexts)
            if self.use_few_shot:
                few_shot_prompt = """
Example 1:
Context: The Eiffel Tower, a wrought-iron lattice tower on the Champ de Mars in Paris, France, is one of the most famous landmarks in the world. It was designed by Gustave Eiffel and completed in 1889.
Statement: The Eiffel Tower can be found in the center of London, near the Thames River.
Response:
No
The statement contradicts with the context, which states that Eiffel Tower is in Paris, as opposed to the center of London.

Example 2:
Context: Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organisms' activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.
Statement: Photosynthesis in plants primarily involves the conversion of light energy into chemical energy stored in forms such as sugar.
Response:
Yes
The statement is supported by the context, which states that photosynthesis converts light energy into chemical energy and that the chemical energy is stored in carbohydrate molecules, such as sugars.
"""
            else:
                few_shot_prompt = ""
            prompt = {
                "system_prompt": (
                    "You are tasked to evaluate whether the statement is fully supported by the context. Respond with either Yes or No, followed by your reasoning in a new line.\n"
                    + few_shot_prompt
                ),
                "user_prompt": ("Context: " + context + r"\Statement: " + answer),
            }

            response = self._llm.run(prompt)
            score_txt, reasoning = response.split("\n", 1)
            score = bool("yes" in score_txt.lower())

        return {
            "LLM_based_faithfulness": score,
            "LLM_based_faithfulness_reasoning": reasoning,
        }


class LLMBasedAnswerCorrectness(LLMBasedMetric):
    """
    The LLM based answer correctness metric.
    Measures whether the generated answer is correct compared to the ground truths.
    """

    def __init__(self, model: LLMInterface = DefaultLLM, use_few_shot: bool = True):
        super().__init__(model)
        self.use_few_shot = use_few_shot

    def __str__(self):
        return f"LLMBasedAnswerCorrectness(model={self.model}, use_few_shot={self.use_few_shot})"

    def calculate(self, question, answer, ground_truths, **kwargs):
        """
        Calculate the faithfulness score for the given datapoint.
        """
        gt_answers = "\n".join(ground_truths)
        if self.use_few_shot:
            few_shot_prompt = """Example Response:
3.5
The answer is relevant to the question and similar to the ground truth answer but misses some information.
"""
        else:
            few_shot_prompt = ""
        prompt = {
            "system_prompt": (
                """
You are an expert evaluator system for a question answering system.
You need to evaluate the quality of the generated answer based on the question and reference ground truth answer.
Output a score and the reasoning for your score in a new line.
Use the following guidelines for evaluation:
* You should output a single score between 1 to 5.
* 1 means that the answer is completely irrelevant to the question.
* 2 means that the answer is relevant to the question but contains major errors.
* 3 means that the answer is relevant to the question and is partially correct.
* 4 means that the answer is relevant to the question and is correct.
* 5 means that the answer is relevant to the question and is correct and complete.
"""
                + few_shot_prompt
            ),
            "user_prompt": (
                "Question: " + question + "\nAnswer: " + answer + r"\Ground truth reference answer(s): " + gt_answers
            ),
        }

        response = self._llm.run(prompt)
        score_txt, reasoning = response.split("\n", 1)
        score = float(score_txt.split(":")[-1].strip())

        return {
            "LLM_based_answer_correctness": score,
            "LLM_based_answer_correctness_reasoning": reasoning,
        }
