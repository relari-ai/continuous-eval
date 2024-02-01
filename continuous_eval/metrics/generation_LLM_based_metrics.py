from typing import Optional

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
        model: Optional[LLMInterface] = None,
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

    def __init__(self, model: Optional[LLMInterface] = None, use_few_shot: bool = True):
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
        normalized_score = (score - 1) / 4

        return {
            "LLM_based_answer_correctness": normalized_score,
            "LLM_based_answer_correctness_reasoning": reasoning,
        }


class LLMBasedAnswerRelevance(LLMBasedMetric):
    """
    The LLM based answer relevance metric.
    Measures whether the generated answer is relevant to the question.
    """

    def __init__(self, model: Optional[LLMInterface] = None, use_few_shot: bool = True):
        super().__init__(model)
        self.use_few_shot = use_few_shot

    def __str__(self):
        return f"LLMBasedAnswerRelevance(model={self.model}, use_few_shot={self.use_few_shot})"

    def calculate(self, question, answer, **kwargs):
        """
        Calculate the faithfulness score for the given datapoint.
        """
        if self.use_few_shot:
            few_shot_prompt = """
Example:
Question:
What is the process of photosynthesis?
Answer:
Photosynthesis is an important process of all plants.
Response:
2
The answer acknowledges the importance of photosynthesis for plants, which is partially relevant. However, it fails to explain the process of photosynthesis, thereby only partially answering the question.

"""
        else:
            few_shot_prompt = ""
        prompt = {
            "system_prompt": (
                """
You are an expert evaluator system for a question answering system.
You need to evaluate the relevance and completeness of the generated answer based on the question.
Output a score and the reasoning for your score in a new line.
Use the following guidelines for evaluation:
* You should output a single score between 1 to 3.
* 1 means that the answer is completely irrelevant to the question.
* 2 means that the answer is partially relevant to the question or it only partially answers the question.
* 3 means that the answer is relevant to the question and completely answers the question.
"""
                + few_shot_prompt
            ),
            "user_prompt": ("Question: " + question + "\nAnswer: " + answer),
        }

        response = self._llm.run(prompt)
        score_txt, reasoning = response.split("\n", 1)
        score = float(score_txt.split(":")[-1].strip())
        normalized_score = (score - 1) / 2

        return {
            "LLM_based_answer_relevance": normalized_score,
            "LLM_based_answer_relevance_reasoning": reasoning,
        }


class LLMBasedStyleConsistency(LLMBasedMetric):
    """
    The LLM based answer relevance metric.
    Measures whether the generated answer is relevant to the question.
    """

    def __init__(self, model: Optional[LLMInterface] = None, use_few_shot: bool = True):
        super().__init__(model)
        self.use_few_shot = use_few_shot

    def __str__(self):
        return f"LLMBasedStyleConsistency(model={self.model}, use_few_shot={self.use_few_shot})"

    def calculate(self, answer, ground_truths, **kwargs):
        """
        Calculate the faithfulness score for the given datapoint.
        """
        gt_answers = "\n".join(ground_truths)
        if self.use_few_shot:
            few_shot_prompt = """
Example:
Generated Answer:
Got it, can you tell me more about it?
Reference Answer(s):
I apologize for the difficulties you're facing. To assist you better, could you please provide more details about the problem?
Response:
2.5
The generated answer is more brief and doesn't have the formality and empathetic tone in the reference answer.
"""
        else:
            few_shot_prompt = ""
        prompt = {
            "system_prompt": (
                """
You are an expert evaluator system for a question answering system.
You need to evaluate the style of the generated answer based on some reference answers.
Assess style aspects such as tone, verbosity, formality, complexity, use of terminology, etc.
Output a score and the reasoning for your score in a new line.
Use the following guidelines for evaluation:
* You should output a single score between 1 to 4.
* 1 means that the answer is in a completely different style as the reference answer(s).
* 2 means that the answer is barely in the same style as the reference answer(s), with noticable differences.
* 3 means that the answer is largely in the same style as the reference answer(s) but there's a slight difference in some aspects.
* 4 means that there's no dicernable style difference between the generated answer and reference answer(s).
"""
                + few_shot_prompt
            ),
            "user_prompt": ("Answer: " + answer + r"\Ground truth reference answer(s): " + gt_answers),
        }

        response = self._llm.run(prompt)
        score_txt, reasoning = response.split("\n", 1)
        score = float(score_txt.split(":")[-1].strip())
        normalized_score = (score - 1) / 3

        return {
            "LLM_based_style_consistency": normalized_score,
            "LLM_based_style_consistency_reasoning": reasoning,
        }
