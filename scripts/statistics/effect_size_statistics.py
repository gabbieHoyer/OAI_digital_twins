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

def calculate_effect_size_stats_and_save(dataframe, target_column, numerical_cols, categorical_cols, file_name):
    # Extracting groups
    group_control = dataframe[dataframe[target_column] == 0].reset_index(drop=True)
    group_treatment = dataframe[dataframe[target_column] == 1].reset_index(drop=True)

    # Ensure equal sizes
    # assert len(group_control) == len(group_treatment), "Groups are not paired correctly!"

    # Define named tuples for results
    NumericalTestResults = namedtuple('NumericalTestResults', ['column', 'biserial_corr', 'p_value', 'test_type'])
    CategoricalTestResults = namedtuple('CategoricalTestResults', ['column', 'association_stat', 'test_type'])

    numerical_results = []
    categorical_results = []

    for col in numerical_cols + categorical_cols:
        if col in numerical_cols:
            # Numerical columns: Point Biserial Correlation Coefficient
            y = dataframe[col].dropna()
            x = dataframe.loc[y.index, target_column]
            biserial_corr, p_value = pointbiserialr(x, y)
            numerical_results.append(NumericalTestResults(col, biserial_corr, p_value, 'biserial_corr'))
        elif col in categorical_cols:
            # Categorical columns: Association
            contingency_table = pd.crosstab(dataframe[target_column], dataframe[col])
            assoc_stat = contingency.association(contingency_table, method='cramer', correction=False)
            categorical_results.append(CategoricalTestResults(col, assoc_stat, 'cramers v'))

    # Converting results into DataFrames
    numerical_results_df = pd.DataFrame(numerical_results)
    categorical_results_df = pd.DataFrame(categorical_results)

    # Save the results to an Excel file
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        numerical_results_df.to_excel(writer, sheet_name='Numerical_Statistics')
        categorical_results_df.to_excel(writer, sheet_name='Categorical_Statistics')


# Before Matching
# OA Incidence example usage
oa_inc_before_match = pd.read_csv('publish_dataframes/oa_inc_multiple_imputation_filled.csv')
numerical_cols = oa_inc_before_match.iloc[:,16:-110].columns.tolist()
categorical_cols = oa_inc_before_match.iloc[:,7:16].columns.tolist()
calculate_effect_size_stats_and_save(oa_inc_before_match, 'oa_prog', numerical_cols, categorical_cols, "publish_dataframes/OA_Inc_Twins_before_matching_effectSize_statistics_output.xlsx")

# TKR example usage
tkr_before_match = pd.read_csv('publish_dataframes/tkr_multiple_imputation_filled.csv')
numerical_cols = tkr_before_match.iloc[:,16:-110].columns.tolist()
categorical_cols = tkr_before_match.iloc[:,7:16].columns.tolist()
calculate_effect_size_stats_and_save(tkr_before_match, 'tkr', numerical_cols, categorical_cols, "publish_dataframes/TKR_Twins_before_matching_effectSize_statistics_output.xlsx")


# After Matching
# OA Incidence example usage
numerical_cols = oa_inc_matched_df.iloc[:,20:-110].columns.tolist()
categorical_cols = oa_inc_matched_df.iloc[:,11:20].columns.tolist()
calculate_effect_size_stats_and_save(oa_inc_matched_df, 'oa_prog', numerical_cols, categorical_cols, "publish_dataframes/OA_Inc_Twins_effectSize_statistics_output.xlsx")

# TKR example usage
numerical_cols = tkr_matched_df.iloc[:,20:-110].columns.tolist()
categorical_cols = tkr_matched_df.iloc[:,11:20].columns.tolist()
calculate_effect_size_stats_and_save(tkr_matched_df, 'tkr', numerical_cols, categorical_cols, "publish_dataframes/TKR_Twins_effectSize_statistics_output.xlsx")
