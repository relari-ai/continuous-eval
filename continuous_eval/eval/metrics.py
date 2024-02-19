from continuous_eval.eval import Metric


class Correctness(Metric):
    def __init__(self):
        super().__init__("correctness")

    def evaluate(self, expected, actual):
        return 1 if expected == actual else 0


class ToolCallCorrectness(Metric):
    def __init__(self):
        super().__init__("tool_call_correctness")

    def evaluate(self, expected, actual):
        return 1 if expected == actual else 0
