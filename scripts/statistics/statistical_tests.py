import pandas as pd
import numpy as np
import openpyxl
import xlsxwriter

import statsmodels.api as sm
from scipy.stats import ttest_rel, false_discovery_control
from scipy.stats import shapiro, anderson, wilcoxon, chi2_contingency, contingency, pointbiserialr

from collections import namedtuple

# Set the random seed for reproducibility
np.random.seed(42)

def perform_normality_tests(dataframe, group_column, column_range):
    # Conducting normality tests on your data to check the distribution assumptions
    # Extracting groups
    group_control = dataframe[dataframe[group_column] == 0].reset_index(drop=True)
    group_treatment = dataframe[dataframe[group_column] == 1].reset_index(drop=True)

    # Ensure equal sizes
    assert len(group_control) == len(group_treatment), "Groups are not paired correctly!"

    # Define a DataFrame to hold our results
    results = pd.DataFrame(columns=['variable', 'shapiro_stat', 'shapiro_p', 'anderson_stat', 'anderson_critical_values', 'anderson_significance_level'])

    # Determine the slicing range
    if isinstance(column_range, tuple):
        start, end = column_range
        selected_columns = dataframe.iloc[:, start:end if end is not None else None].columns
    elif isinstance(column_range, slice):
        selected_columns = dataframe.iloc[:, column_range].columns
    else:
        raise ValueError("column_range must be a slice or a tuple")

    # Loop through all variable columns
    for col in selected_columns:
        differences = group_treatment[col] - group_control[col]

        # Drop NA values from differences
        differences = differences.dropna()

        # Shapiro-Wilk Test
        shapiro_stat, shapiro_p = shapiro(differences)

        # Anderson-Darling Test
        anderson_result = anderson(differences)
        anderson_stat = anderson_result.statistic
        anderson_critical_values = anderson_result.critical_values
        anderson_significance_level = anderson_result.significance_level

        # Append results
        results = results.append({
            'variable': col,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'anderson_stat': anderson_stat,
            'anderson_critical_values': anderson_critical_values,
            'anderson_significance_level': anderson_significance_level
        }, ignore_index=True)

    return results


# oa_inc_numerical_results = perform_normality_tests(oa_inc_matched_df, 'oa_prog', (20, -110))
# tkr_numerical_results = perform_normality_tests(tkr_matched_df, 'tkr', (20, -110))

# # Optionally, save the results to Excel files
# oa_inc_numerical_results.to_excel('publish_dataframes/OA_Inc_demos_clinicalFactors_normality_tests_results.xlsx', index=False, engine='openpyxl')
# tkr_numerical_results.to_excel('publish_dataframes/TKR_demos_clinicalFactors_normality_tests_results.xlsx', index=False, engine='openpyxl')

# ************************************************************************* #
# Checking Normality and Homoscedasticity of Twin/Matched Subject imaging biomarker PC modes

# Example usage for oa_inc_matched_df
# oa_inc_results = perform_normality_tests(oa_inc_matched_df, 'oa_prog', (-110, None))

# # Example usage for tkr_matched_df
# tkr_results = perform_normality_tests(tkr_matched_df, 'tkr', (-110, None))

# Optionally, save the results to Excel files
# oa_inc_results.to_excel('publish_dataframes/OA_Inc_Clinical_Twin_PCA_variables_normality_tests_results.xlsx', index=False, engine='openpyxl')
# tkr_results.to_excel('publish_dataframes/TKR_Clinical_Twin_PCA_variables_normality_tests_results.xlsx', index=False, engine='openpyxl')



