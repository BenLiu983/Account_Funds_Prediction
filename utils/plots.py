import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

def histogram_plot(df, column_name, bins=50, color="skyblue", title="Histogram Plot", xlabel="Feature", ylabel="Frequency"):
    """
    Plots a histogram for a specified column in the DataFrame using plt.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column to plot.
        bins (int): Number of bins for the histogram. Default is 50.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins=bins, color=color, edgecolor="black", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def fea_imp_lgb(lgb_model, max_features=20):
    """
    Plot the feature importance of a LightGBM model.

    Parameters:
    - lgb_model: Trained LightGBM model (e.g., LGBMRegressor, LGBMClassifier, or Booster).
    - max_features: Maximum number of top features to display (default: 20).

    Returns:
    - A matplotlib plot of the feature importances.
    """
    # Access the Booster object if using the scikit-learn API
    if hasattr(lgb_model, 'booster_'):
        booster = lgb_model.booster_
    else:
        booster = lgb_model  # assume it's already a Booster object

    # extract feature importance and corresponding feature names
    feature_importance = booster.feature_importance(importance_type='gain')
    feature_names = booster.feature_name()

    # combine feature names and importance into a list and sort by importance
    feature_importance_data = sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )

    # limit to the top features based on max_features
    top_features = feature_importance_data[:max_features]
    top_feature_names, top_importance = zip(*top_features)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_feature_names[::-1], top_importance[::-1], color='skyblue')
    plt.xlabel('Feature Importance (Gain)')
    plt.ylabel('Features')
    plt.title('Top Feature Importance for LightGBM')
    plt.tight_layout()
    plt.show()
