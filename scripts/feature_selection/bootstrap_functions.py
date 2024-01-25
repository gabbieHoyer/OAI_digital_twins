# File: bootstrap_functions.py

import pandas as pd
import pickle
import os
import numpy as np
import statsmodels.api as sm
from sklearn.utils import resample, shuffle
from memory_profiler import profile


def filter_covariates(dataframe, target, confounders, slice_start):
    """
    Filters a DataFrame to include specific covariates based on the provided target and confounders.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be filtered.
    target (str): The name of the target variable.
    confounders (list): A list of column names to be excluded as confounders.

    Returns:
    pd.DataFrame: A DataFrame filtered to include the desired covariates.
    """
    # Extracting covariates (excluding the columns specified in confounders)
    # and including the target variable
    covariates = [col for col in dataframe.columns[slice_start:] if col not in confounders] + [target]

    # Creating a new DataFrame with the selected covariates
    filtered_df = dataframe[covariates]

    return filtered_df

def save_checkpoint(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename, features):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            checkpoint_data = pickle.load(f)
            return (checkpoint_data.get('bootstrap_estimators', {}),
                    checkpoint_data.get('feature_selection_frequency', {feature: 0 for feature in features}),
                    checkpoint_data.get('current_iteration', 0))
    else:
        # If the file doesn't exist, return default values
        return {}, {feature: 0 for feature in features}, 0


# @profile
def bootstrap_elastic_net_with_feature_selection(df, X, y, n_bootstrap=1000, alpha=0.01, L1_wt=0.5, maxiter=1000, selection_threshold=1e-5, checkpoint_interval=100):
    # checkpoint_file = 'bootstrap_checkpoint.pkl'
    checkpoint_file = '/data/VirtualAging/users/ghoyer/OAI/digital_trial/mega_bootstrap/better_TKR_checkpoints/bootstrap_checkpoint.pkl'
    
    # Load previous state if checkpoint exists
    bootstrap_estimators, feature_selection_frequency, start_iteration = load_checkpoint(checkpoint_file, X.columns)

    for i in range(start_iteration, n_bootstrap):
        if i % 10 == 0:
            print(f"Iteration: {i}", flush=True)

        # Bootstrap sample
        df_bootstrap = resample(df, replace=True, n_samples=len(df), random_state=i)
        X_bootstrap = df_bootstrap[X.columns]
        y_bootstrap = df_bootstrap[y.name]

        # Add constant to the model
        X_bootstrap_const = sm.add_constant(X_bootstrap)

        # Create GLM model
        glm_model_bootstrap = sm.GLM(y_bootstrap, X_bootstrap_const, family=sm.families.Binomial())
        
        # Fit the model with Elastic Net regularization
        elastic_net_model_bootstrap = glm_model_bootstrap.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=L1_wt, maxiter=maxiter)
        
        # Store the coefficients
        coefficients = elastic_net_model_bootstrap.params
        bootstrap_estimators[i] = coefficients

        # Track feature selection
        for feature, coef in coefficients.items():
            if abs(coef) > selection_threshold:
                feature_selection_frequency[feature] = feature_selection_frequency.get(feature, 0) + 1

        if (i + 1) % checkpoint_interval == 0:
            checkpoint_data = {
                'bootstrap_estimators': bootstrap_estimators, 
                'feature_selection_frequency': feature_selection_frequency, 
                'current_iteration': i + 1
            }
            save_checkpoint(checkpoint_data, checkpoint_file)
            print(f"Checkpoint saved at iteration {i + 1}", flush=True)

    # Final save
    checkpoint_data = {'bootstrap_estimators': bootstrap_estimators, 'feature_selection_frequency': feature_selection_frequency, 'current_iteration': n_bootstrap}
    save_checkpoint(checkpoint_data, checkpoint_file)
    print("Final checkpoint saved.", flush=True)

    # Convert frequency counts to proportions
    for feature in feature_selection_frequency:
        feature_selection_frequency[feature] /= n_bootstrap

    return bootstrap_estimators, pd.DataFrame.from_dict(feature_selection_frequency, orient='index', columns=['Selection Frequency'])
