from continuous_eval.metrics.base import EVAL_LLM, LLMBasedMetric


class LLMBasedFaithfulness(LLMBasedMetric):
    """
    The LLM based faithfulness metric.
    Measures whether the generated answer is faithful to the retrieved context.
    """

    def __init__(self, model: str = EVAL_LLM, use_few_shot: bool = True):
        super().__init__(model)
        self.use_few_shot = use_few_shot

    def calculate(self, retrieved_contexts, answer, **kwargs):
        """
        Calculate the faithfulness score for the given datapoint.
        """
        context = "\n".join(retrieved_contexts)
        if self.use_few_shot:
            few_shot_prompt = """
Example 1:
Context: The Eiffel Tower, a wrought-iron lattice tower on the Champ de Mars in Paris, France, is one of the most famous landmarks in the world. It was designed by Gustave Eiffel and completed in 1889.
Statement: The Eiffel Tower can be found in the center of London, near the Thames River.
Response: No
Example 2:
Context: Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organisms' activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.
Statement: Photosynthesis in plants primarily involves the conversion of light energy into glucose, a simple sugar.
Response: Yes
"""
        else:
            few_shot_prompt = ""
        prompt = {
            "system_prompt": (
                "You are tasked to evaluate whether the statement is fully supported by the context. Respond with either Yes or No.\n"
                + few_shot_prompt
            ),
            "user_prompt": ("Context: " + context + "\Statement: " + answer),
        }

        response = self._llm_response(prompt)
        score = "yes" in response.lower()

        return {"LLM_based_faithfulness_score": score}


class LLMBasedAnswerCorrectness(LLMBasedMetric):
    """
    The LLM based answer relevance metric.
    Measures whether the generated answer is relenvat to the question.
    """

    def __init__(self, model: str = EVAL_LLM, use_few_shot: bool = True):
        super().__init__(model)
        self.use_few_shot = use_few_shot

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
                "Question: "
                + question
                + "\nAnswer: "
                + answer
                + "\Ground truth reference answer(s): "
                + gt_answers
            ),
        }

        response = self._llm_response(prompt)
        score_txt, reasoning = response.split("\n", 1)
        score = float(score_txt)

        return {"LLM_based_answer_correctness": score}
