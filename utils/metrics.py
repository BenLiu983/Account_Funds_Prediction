import pandas as pd
import numpy as np

# Gini function to calculate Gini coefficient
def gini(model_array, truth_array):
    """
    Calculate the Gini coefficient for a given model's predictions and true values.

    Args:
    model_array (numpy.array): Predicted values from the model
    truth_array (numpy.array): True values

    Returns:
    float: Gini coefficient
    """
    d = pd.DataFrame({'pred': model_array, 'truth': truth_array})
    d = d.sort_values(by='pred')
    d['truth'] = d['truth'] / np.sum(d['truth'])
    gini_score = 1 - 2 * np.sum(np.cumsum(d['truth'].to_numpy())) / len(model_array)
    return gini_score


# Power Ratio function to calculate power ratio equivalent to AUC for non-binary targets
def PowerRatio(y_pred, y_true):
    """
    Calculate the Power Ratio (equivalent of AUC for non-binary target).

    Args:
    pred (numpy.array): Predicted values from the model
    true (numpy.array): True values

    Returns:
    float: Adjusted Power Ratio
    """
    numerator = gini(y_pred, y_true)
    denominator = gini(y_true, y_true)
    pr = numerator / denominator
    pr_adjusted = (1 + pr) / 2

    return pr_adjusted
