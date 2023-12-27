import numpy as np
import pandas as pd


def dummy_results(num_samples, cols, prob_detection=0.9, prob_false_alarm=0.1, seed=42):
    np.random.seed(seed=seed)
    X_values = list()
    y_values = np.random.randint(0, 2, num_samples)
    prob_fcn = lambda x: (1 - prob_detection) ** x * (1 - prob_false_alarm) ** (1 - x)
    outcome_fcn = lambda x: int(np.random.random() > prob_fcn(x))
    X_values = [{col: outcome_fcn(x) for col in cols} for x in y_values]
    return pd.DataFrame(X_values), np.array(y_values)
