from typing import Dict

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def eval_prediction(y, y_hat, average="binary", labels=None) -> Dict:
    accuracy = accuracy_score(y, y_hat)
    precision = precision_score(y, y_hat, average=average)
    recall = recall_score(y, y_hat, average=average)
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = float("nan")
    return dict(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        confusion_matrix=confusion_matrix(y, y_hat, labels=labels),
    )
