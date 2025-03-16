
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import boxcox
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from data_processing import optimise_memory

def transform_input_data(input_data, data_before_trans, data_after_transf, cont_features, other_features, boxcox_lambdas):
    """
    Transforms new input data using precomputed transformations.

    - Applies Box-Cox transformation for continuous features using precomputed lambdas.
    - Applies Winsorization for 'platelets' based on original dataset percentiles.
    - Standardizes using precomputed mean and std from transformed data.

    Parameters:
    - input_data (pd.DataFrame): New input data to be transformed.
    - data_before_trans (pd.DataFrame): Dataset before transformations (used for Winsorization).
    - data_after_transf (pd.DataFrame): Dataset after transformations (used to extract mean & std for standardization).
    - cont_features (list): List of continuous features to apply Box-Cox.
    - other_features (list): List of features that should remain unchanged.
    - boxcox_lambdas (dict): Precomputed dictionary of Box-Cox lambda values for continuous features.

    Returns:
    - pd.DataFrame: Transformed and standardized input data.
    """
    data_transformed = input_data.copy()

    # Apply Box-Cox transformation to continuous features using precomputed lambdas
    for col in cont_features:
        if col in boxcox_lambdas and col in data_transformed.columns:
            lambda_val = boxcox_lambdas[col]
            data_transformed[col] = boxcox(data_transformed[col] + 1e-6, lambda_val)  # Avoid zero values

    # Apply Winsorization to 'platelets' using percentiles from data_before_trans
    if 'platelets' in data_transformed.columns and 'platelets' in data_before_trans.columns:
        lower_bound = np.percentile(data_before_trans['platelets'], 5)
        upper_bound = np.percentile(data_before_trans['platelets'], 95)
        data_transformed['platelets'] = np.clip(data_transformed['platelets'], lower_bound, upper_bound)

    # Ensure other features remain unchanged
    for col in other_features:
        if col in input_data.columns:
            data_transformed[col] = input_data[col]

    # Extract precomputed means and stds from data_after_transf
    stats = data_after_transf.describe().T
    means = stats["mean"]
    stds = stats["std"]

    # Standardize input data using precomputed mean & std
    for col in data_transformed.columns:
        if col != "DEATH_EVENT" and col in means and col in stds and stds[col] != 0:  # Avoid division by zero
            data_transformed[col] = (data_transformed[col] - means[col]) / stds[col]

    return data_transformed
