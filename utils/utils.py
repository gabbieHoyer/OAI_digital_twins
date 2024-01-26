import pandas as pd
import numpy as np

def load_and_merge_data(full_data_path, match_data_path, merge_key, selected_columns=['distance','id']):
    full_df = pd.read_csv(full_data_path)
    match_df = pd.read_csv(match_data_path, usecols=selected_columns)
    merged_df = pd.merge(match_df, full_df, on=merge_key, how='left')
    return merged_df


def filter_covariates(dataframe, target, confounders=['height', 'weight', 'womac_pain', 'womac_adl', 'womac_stiff', 'womac_total']):
    """
    Filters a DataFrame to include specific covariates based on the provided target and confounders.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be filtered.
    target (str): The name of the target variable.
    confounders (list): A list of column names to be excluded as confounders.

    Returns:
    pd.DataFrame: A DataFrame filtered to exclude confounders and include the target variable.
    """
    # Retain columns that are not in confounders and the target variable
    covariates = [col for col in dataframe.columns if col not in confounders or col == target]

    # Creating a new DataFrame with the selected covariates
    filtered_df = dataframe[covariates]

    return filtered_df


def save_model_reports(glm_logit_model, target_variable, output_dir):
    """
    Saves various statistical reports and metrics from a logistic regression model,
    with filenames including the target variable name.

    Parameters:
    glm_logit_model: The logistic regression model from statsmodels.
    target_variable (str): The name of the target variable in the model.
    output_dir (str): The directory path where the reports will be saved.
    """
    # Ensure output_dir ends with a slash
    if not output_dir.endswith('/'):
        output_dir += '/'

    # Save the first summary report
    model_summary_str = str(glm_logit_model.summary())
    summary_filename = f"{output_dir}{target_variable}_logReg_model_summary.txt"
    with open(summary_filename, 'w') as file:
        file.write(model_summary_str)

    # Save the second summary report
    model_summary_str_v2 = str(glm_logit_model.summary2())
    summary_v2_filename = f"{output_dir}{target_variable}_logReg_model_summary_v2.txt"
    with open(summary_v2_filename, 'w') as file:
        file.write(model_summary_str_v2)

    # Save additional metrics
    metrics_filename = f"{output_dir}{target_variable}_logReg_model_metrics.txt"
    with open(metrics_filename, 'w') as file:
        file.write(f"AIC: {glm_logit_model.aic}\n")
        file.write(f"BIC: {glm_logit_model.bic}\n")
        file.write(f"BIC LLF: {glm_logit_model.bic_llf}\n")
        file.write(f"LLF: {glm_logit_model.llf}\n")
        file.write(f"Pearson chi2: {glm_logit_model.pearson_chi2}\n")
        file.write(f"Pseudo R-squared: {glm_logit_model.pseudo_rsquared(kind='cs')}\n")


