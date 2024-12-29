import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def histogram_plot(df, column_name, bins=50, color="skyblue", title="Histogram Plot", xlabel="Feature", ylabel="Frequency"):
    """
    Plots a histogram for a specified column in the DataFrame using plt.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column to plot.
        bins (int): Number of bins for the histogram. Default is 50.
        color (str): Color of the histogram bars. Default is 'skyblue'.
        title (str): Title of the histogram. Default is 'Histogram Plot'.
        xlabel (str): Label for the x-axis. Default is 'Feature'.
        ylabel (str): Label for the y-axis. Default is 'Frequency'.
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

