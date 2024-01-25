import pandas as pd
import numpy as np
import openpyxl
import xlsxwriter

import statsmodels.api as sm
from scipy.stats import ttest_rel, false_discovery_control
from scipy.stats import shapiro, anderson, wilcoxon, chi2_contingency, contingency, pointbiserialr

from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

def bootstrap_median_difference(treatment_col, control_col, n_bootstrap=1000, confidence_levels=[95, 99], seed=None):
    """
    Performs bootstrap analysis to estimate confidence intervals of the median differences between two columns.

    Parameters:
    treatment_col (array-like): The treatment group column.
    control_col (array-like): The control group column.
    n_bootstrap (int): The number of bootstrap samples to draw.
    confidence_levels (list): The confidence levels for which to compute the intervals.
    seed (int, optional): The seed for the random number generator.

    Returns:
    dict: Confidence intervals for each specified level.
    """
    if seed is not None:
        np.random.seed(seed)

    if len(treatment_col) != len(control_col):
        raise ValueError("Treatment and control columns must be of the same length.")

    bootstrapped_medians = []
    for _ in range(n_bootstrap):
        resampled_oa = np.random.choice(treatment_col, size=len(treatment_col), replace=True)
        resampled_control = np.random.choice(control_col, size=len(control_col), replace=True)
        median_diff = np.median(resampled_oa - resampled_control)
        bootstrapped_medians.append(median_diff)

    ci_bounds = {}
    for level in confidence_levels:
        if not (0 < level < 100):
            raise ValueError("Confidence levels must be between 0 and 100.")
        lower_bound = np.percentile(bootstrapped_medians, (100 - level) / 2)
        upper_bound = np.percentile(bootstrapped_medians, 100 - (100 - level) / 2)
        ci_bounds[level] = (lower_bound, upper_bound)

    return ci_bounds


def calculate_nonparametric_stats_and_save(dataframe, target_column, numerical_cols, categorical_cols, file_name):
    # Extracting groups
    group_control = dataframe[dataframe[target_column] == 0].reset_index(drop=True)
    group_treatment = dataframe[dataframe[target_column] == 1].reset_index(drop=True)

    # Ensure equal sizes
    assert len(group_control) == len(group_treatment), "Groups are not paired correctly!"

    # Define named tuples for results
    WilcoxonTestResults = namedtuple('WilcoxonTestResults', ['column', 'wilcoxon_stat', 'p_value', 'dof', 'ci_95_low', 'ci_95_high', 'ci_99_low', 'ci_99_high', 'test_type', 'median_diff', 'margin_of_err95', 'margin_of_err99', 'point_estimate_ci_95', 'point_estimate_ci_99'])
    CategoricalTestResults = namedtuple('CategoricalTestResults', ['column', 'chi2_stat', 'p_value', 'dof', 'expected', 'observed', 'test_type'])

    numerical_results = []
    categorical_results = []

    for col in numerical_cols + categorical_cols:
        if col in numerical_cols:
            # Drop any NaN values from consideration
            control_col = group_control[col].dropna()
            treatment_col = group_treatment[col].dropna()

            # Ensure equal sizes
            if len(control_col) != len(treatment_col):
                print(f"Column: {col} - Groups are not paired correctly or have missing data!")
                continue

            # Calculate the median difference / point estimate for Wilcoxon test
            median_diff = np.median(treatment_col - control_col)

            # Perform Wilcoxon Signed-Rank Test
            stat, p_value = wilcoxon(treatment_col, control_col)

            # No direct method for confidence intervals in Wilcoxon test, Perform bootstrapping
            ci_bounds = bootstrap_median_difference(treatment_col, control_col, n_bootstrap=1000, confidence_levels=[95, 99])

            # Use the number of non-missing paired observations minus one for df
            df = len(control_col) - 1

            # margin of error calculation:
            margin_of_err95 = (ci_bounds[95][1] - ci_bounds[95][0]) / 2
            margin_of_err99 = (ci_bounds[99][1] - ci_bounds[99][0]) / 2

            # point estimate [CI lower, CI upper]:
            formatted_ci_95 = f"{median_diff}, 95% CI: [{ci_bounds[95][0]}, {ci_bounds[95][1]}]"
            formatted_ci_99 = f"{median_diff}, 99% CI: [{ci_bounds[99][0]}, {ci_bounds[99][1]}]"

            # Append results with bootstrapped confidence intervals
            numerical_results.append(WilcoxonTestResults(col, stat, p_value, df,
                                                        ci_bounds[95][0], ci_bounds[95][1],
                                                        ci_bounds[99][0], ci_bounds[99][1],
                                                        'wilcoxon', median_diff, margin_of_err95, margin_of_err99,
                                                        formatted_ci_95, formatted_ci_99
                                              ))
        elif col in categorical_cols:
            # Categorical columns: Chi-squared test
            contingency_table = pd.crosstab(dataframe[target_column], dataframe[col])  # contingency table of observed frequencies
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Add the observed frequencies to your results - new change here
            categorical_results.append(CategoricalTestResults(col, chi2_stat, p_value, dof, expected.tolist(), contingency_table.values.tolist(), 'chi-squared'))

    # Converting results into DataFrames
    numerical_results_df = pd.DataFrame(numerical_results)
    categorical_results_df = pd.DataFrame(categorical_results)

    # Save the results to an Excel file
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        numerical_results_df.to_excel(writer, sheet_name='Numerical_Statistics')
        categorical_results_df.to_excel(writer, sheet_name='Categorical_Statistics')


# After matching
# OA Incidence example usage
# numerical_cols = oa_inc_matched_df.iloc[:,20:-110].columns.tolist()
# categorical_cols = oa_inc_matched_df.iloc[:,11:20].columns.tolist()
# calculate_nonparametric_stats_and_save(oa_inc_matched_df, 'oa_prog', numerical_cols, categorical_cols, "publish_dataframes/OA_Inc_Twins_nonparametric_statistics_output.xlsx")

# TKR example usage
# numerical_cols = tkr_matched_df.iloc[:,20:-110].columns.tolist()
# categorical_cols = tkr_matched_df.iloc[:,11:20].columns.tolist()
# calculate_nonparametric_stats_and_save(tkr_matched_df, 'tkr', numerical_cols, categorical_cols, "publish_dataframes/TKR_Twins_nonparametric_statistics_output.xlsx")


# **********************************************************#
def perform_statistical_tests(dataframe, group_column, variable_columns_range, non_norm_columns):
    # Extracting groups
    group_control = dataframe[dataframe[group_column] == 0].reset_index(drop=True)
    group_treatment = dataframe[dataframe[group_column] == 1].reset_index(drop=True)

    # Ensure equal sizes
    assert len(group_control) == len(group_treatment), "Groups are not paired correctly!"

    # Define named tuples for results
    TestResults = namedtuple('TestResults', ['column', 't_stat', 'p_value', 'adj_p_value', 'df', 'ci_95_low', 'ci_95_high', 'ci_99_low', 'ci_99_high', 'test_type', 'mean_diff', 'margin_of_err95', 'margin_of_err99', 'point_estimate_ci_95', 'point_estimate_ci_99'])
    WilcoxonTestResults = namedtuple('WilcoxonTestResults', ['column', 'wilcoxon_stat', 'p_value', 'adj_p_value', 'df', 'ci_95_low', 'ci_95_high', 'ci_99_low', 'ci_99_high', 'test_type', 'median_diff', 'margin_of_err95', 'margin_of_err99', 'point_estimate_ci_95', 'point_estimate_ci_99'])

    results = []
    numerical_p_values = []

    # Loop through variable columns
    for col in dataframe.iloc[:, variable_columns_range:].columns:
        control_col = group_control[col].dropna()
        treatment_col = group_treatment[col].dropna()

        # Ensure equal sizes
        if len(control_col) != len(treatment_col):
            print(f"Column: {col} - Groups are not paired correctly or have missing data!")
            continue

        # Determine if we should use t-test or Wilcoxon test
        if col in non_norm_columns:

            # Calculate the median difference / point estimate for Wilcoxon test
            median_diff = np.median(treatment_col - control_col)

            # Perform Wilcoxon Signed-Rank Test instead of t-test
            stat, p_value = wilcoxon(treatment_col, control_col)
            # No direct method for confidence intervals in Wilcoxon test, consider using a bootstrapping method if needed

            # Perform bootstrapping for confidence intervals
            ci_bounds = bootstrap_median_difference(treatment_col, control_col, n_bootstrap=1000, confidence_levels=[95, 99])

            # Use the number of non-missing paired observations minus one for df
            df = len(control_col) - 1

            # Accumulate p-values for Wilcoxon tests
            numerical_p_values.append(p_value)

            # margin of error calculation:
            margin_of_err95 = (ci_bounds[95][1] - ci_bounds[95][0]) / 2
            margin_of_err99 = (ci_bounds[99][1] - ci_bounds[99][0]) / 2

            # point estimate [CI lower, CI upper]:
            formatted_ci_95 = f"{median_diff}, 95% CI: [{ci_bounds[95][0]}, {ci_bounds[95][1]}]"
            formatted_ci_99 = f"{median_diff}, 99% CI: [{ci_bounds[99][0]}, {ci_bounds[99][1]}]"

            # Append results with bootstrapped confidence intervals
            results.append(WilcoxonTestResults(col, stat, p_value, None, df,
                                                        ci_bounds[95][0], ci_bounds[95][1],
                                                        ci_bounds[99][0], ci_bounds[99][1],
                                                        'wilcoxon', median_diff, margin_of_err95, margin_of_err99,
                                                        formatted_ci_95, formatted_ci_99
                                              ))

        else:
            # Calculate the mean difference / point estimate for t-test
            mean_diff = np.mean(treatment_col - control_col)

            # Paired t-test
            l = ttest_rel(treatment_col, control_col)

            df = len(control_col) - 1

            try:
                ci_95 = l.confidence_interval(confidence_level=0.95)
                ci_99 = l.confidence_interval(confidence_level=0.99)
            except AttributeError:
                print(f"Column: {col} - 'confidence_interval' method is not available. Update SciPy or use an alternative method.")
                continue

            # Accumulate p-values for numerical tests
            numerical_p_values.append(l.pvalue)

                    # margin of error calculation:
            margin_of_err95 = (ci_95.high - ci_95.low) / 2
            margin_of_err99 = (ci_99.high - ci_99.low) / 2

            # point estimate [CI lower, CI upper]:
            formatted_ci_95 = f"{mean_diff}, 95% CI: [{ci_95.low}, {ci_95.high}]"
            formatted_ci_99 = f"{mean_diff}, 99% CI: [{ci_99.low}, {ci_99.high}]"


            # Append results with None for adj_p_value
            results.append(TestResults(col, l.statistic, l.pvalue, None, df, ci_95.low, ci_95.high, ci_99.low, ci_99.high, 'ttest', mean_diff, margin_of_err95, margin_of_err99,
                                      formatted_ci_95, formatted_ci_99
                                      ))

    # Accumulate all p-values from both numerical and categorical tests
    all_p_values = numerical_p_values

    # Apply Hochberg correction to the combined p-values
    adj_all_p_values = false_discovery_control(all_p_values, method='bh')

    # Separate the adjusted p-values back into numerical and categorical
    adj_numerical_p_values = adj_all_p_values[:len(numerical_p_values)]

    # Assign the adjusted p-values back to the results
    for i in range(len(results)):
        results[i] = results[i]._replace(adj_p_value=adj_numerical_p_values[i])

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df

# # Example usage for oa_inc_matched_df
# non_norm_columns_oa_inc = oa_inc_matched_df.iloc[:,-110:].columns.tolist()
# results_df_oa_inc = perform_statistical_tests(oa_inc_matched_df, 'oa_prog', -110, non_norm_columns_oa_inc)

# # Example usage for tkr_matched_df
# non_norm_columns_tkr = tkr_matched_df.iloc[:,-110:].columns.tolist()
# results_df_tkr = perform_statistical_tests(tkr_matched_df, 'tkr', -110, non_norm_columns_tkr)

# Optionally, save results to Excel
# results_df_oa_inc.to_excel("publish_dataframes/OA_Inc_meanDifference_wilcoxon_hochbergCorrected_results.xlsx", index=False)
# results_df_tkr.to_excel("publish_dataframes/TKR_meanDifference_wilcoxon_hochbergCorrected_results.xlsx", index=False)
