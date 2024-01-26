
import pandas as pd
import numpy as np
import scipy.stats

import openpyxl
import xlsxwriter

def calculate_feature_stat_summary(raw_coef_df, sel_freq_df, output_dir):
    """
    Calculate statistical summaries, including mean, standard deviation,
    confidence intervals, selection frequencies, and other metrics for coefficients.

    :param raw_coef_df: DataFrame with coefficients from each iteration.
    :param sel_freq_df: DataFrame with selection frequencies.
    :return: DataFrame with the statistical summary.
    """
    # raw_coef_df is dataframe with coefficients from each iteration
    # df_freq is dataframe with selection frequencies

    # Calculate the mean and standard deviation for each coefficient across the 1000 iterations.
    means = raw_coef_df.mean()
    std_devs = raw_coef_df.std()

    # Calculate the 95% and 99% confidence intervals for each coefficient.
    ci_95 = {}
    ci_99 = {}
    for col in raw_coef_df.columns:
        ci_95[col] = scipy.stats.norm.interval(0.95, loc=means[col], scale=std_devs[col]/np.sqrt(len(raw_coef_df)))
        ci_99[col] = scipy.stats.norm.interval(0.99, loc=means[col], scale=std_devs[col]/np.sqrt(len(raw_coef_df)))

    # Create a DataFrame for the selection frequency.
    df_selection_freq = pd.DataFrame({'variable': sel_freq_df['Unnamed: 0'], 'selection_frequency': sel_freq_df['Selection Frequency']})

    # Merge the selection frequency DataFrame with the average coefficients and confidence intervals.
    df_summary = pd.DataFrame(means, columns=['mean_coef'])
    df_summary['std_dev'] = std_devs
    df_summary['ci_95_lower'], df_summary['ci_95_upper'] = zip(*ci_95.values())
    df_summary['ci_99_lower'], df_summary['ci_99_upper'] = zip(*ci_99.values())
    df_summary = df_summary.merge(df_selection_freq, left_index=True, right_on='variable')

    # Assuming a variable is included in the CI if the CI does not cross 0
    df_summary['included_in_95'] = (df_summary['ci_95_lower'] * df_summary['ci_95_upper']) > 0
    df_summary['included_in_99'] = (df_summary['ci_95_lower'] * df_summary['ci_95_upper']) > 0 # Update with actual 99% CI if available

    # Add a column to indicate whether the confidence interval includes zero
    df_summary['ci_includes_zero'] = (df_summary['ci_95_lower'] <= 0) & (df_summary['ci_95_upper'] >= 0)

    # Effect Size and Confidence Interval Width
    df_summary['ci_width'] = df_summary['ci_95_upper'] - df_summary['ci_95_lower']

    # Calculate the error bars based on the confidence interval width
    df_summary['error'] = (df_summary['ci_95_upper'] - df_summary['ci_95_lower']) / 2

    # Adjusting the calculation of weighted importance using absolute values
    df_summary['abs_mean_coef'] = df_summary['mean_coef'].abs()
    df_summary['weighted_importance'] = df_summary['abs_mean_coef'] * df_summary['selection_frequency']

    # Ranking Variables
    df_summary['rank'] = df_summary['weighted_importance'].rank(ascending=False)

    df_summary.to_excel(f'{output_dir}/bootstrap_coefficient_stats_summary.xlsx', index=False)

    return df_summary


def select_top_features(df_summary, quantile=0.75, mandatory_covariates=['gender', 'BMI', 'age', 'race']):
    """
    Selects the top features based on weighted importance and ensures mandatory covariates are included.

    :param df_summary: DataFrame containing the summary of features with 'weighted_importance'.
    :param quantile: Quantile value to determine the threshold for top features selection.
                     Default is 0.75 for the top 25%.
    :param mandatory_covariates: List of covariates that must be included in the final list.
    :return: List of selected feature names including mandatory covariates.
    """
    # Calculate the threshold for the top 25%
    quantile_threshold = df_summary['weighted_importance'].quantile(quantile)

    # Filter features that meet the threshold
    important_features = df_summary[df_summary['weighted_importance'] >= quantile_threshold]

    # Extract the names of these important features
    selected_feature_names = important_features['variable'].tolist()

    # Add missing mandatory covariates
    for cov in mandatory_covariates:
        if cov not in selected_feature_names:
            selected_feature_names.append(cov)

    return selected_feature_names

def calculate_odds_ratios_and_stats(glm_logit_model, output_dir):
    """
    Calculates odds ratios, confidence intervals, and additional statistics 
    from a logistic regression model.

    Parameters:
    glm_logit_model: The logistic regression model from statsmodels.

    Returns:
    pd.DataFrame: A DataFrame containing the odds ratios, confidence intervals, 
                  p-values, standard errors, and z-values.
    """
    # Calculate odds ratios and 95% CI
    odds_ratios = np.exp(glm_logit_model.params)
    conf_int = np.exp(glm_logit_model.conf_int())
    conf_int['Odds Ratio'] = odds_ratios

    # Add a column with p-values
    conf_int['p-value'] = glm_logit_model.pvalues

    # Add a column for Standard Error
    conf_int['Std.Err.'] = glm_logit_model.bse

    # Add a column for z-values
    conf_int['z-value'] = glm_logit_model.params / glm_logit_model.bse

    # Rename columns for the CI to be more informative
    conf_int.columns = ['CI 2.5%', 'CI 97.5%', 'Odds Ratio', 'p-value', 'Std.Err.', 'z-value']
    
    conf_int.to_excel(f'{output_dir}/odds_ratio_stats_summary.xlsx', index=False)

    return conf_int


def extract_significant_variables(glm_logit_model, alpha=0.05, output_dir=None):
    """
    Extracts coefficients and p-values from a logistic regression model, 
    filters out variables with p-values greater than the specified alpha value, 
    and optionally saves the DataFrame to an Excel file.

    Parameters:
    glm_logit_model: The logistic regression model from statsmodels.
    alpha (float): The significance level for filtering variables (default is 0.05).
    output_dir (str, optional): The directory path where the Excel file will be saved.
                                If None, the DataFrame is not saved to a file.

    Returns:
    pd.DataFrame: A DataFrame containing variables, coefficients, and p-values 
                  for significant variables (p-value < alpha).
    """
    # Extract the coefficients and p-values
    params = glm_logit_model.params
    pvalues = glm_logit_model.pvalues

    # Create DataFrame with variables, coefficients, and p-values
    significant_vars = pd.DataFrame({
        'Variable': params.index, 
        'Coefficient': params.values, 
        'P-value': pvalues.values
    })

    # Filter only those variables with p-values less than alpha
    significant_vars = significant_vars[significant_vars['P-value'] < alpha]

    # Save to Excel file if output directory is provided
    if output_dir:
        if not output_dir.endswith('/'):
            output_dir += '/'
        file_path = f'{output_dir}significant_variables_summary.xlsx'
        significant_vars.to_excel(file_path, index=False)

    return significant_vars


