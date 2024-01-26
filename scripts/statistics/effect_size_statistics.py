import pandas as pd
import numpy as np
import openpyxl
import xlsxwriter

import statsmodels.api as sm
from scipy.stats import ttest_rel, false_discovery_control
from scipy.stats import shapiro, anderson, wilcoxon, chi2_contingency, contingency, pointbiserialr

from collections import namedtuple

def calculate_effect_size_stats_and_save(dataframe, target_column, continuous_cols, categorical_cols, output_dir, match_flag):
    # Extracting groups
    group_control = dataframe[dataframe[target_column] == 0].reset_index(drop=True)
    group_treatment = dataframe[dataframe[target_column] == 1].reset_index(drop=True)

    # Define named tuples for results
    continuousTestResults = namedtuple('ContinuousTestResults', ['column', 'biserial_corr', 'p_value', 'test_type'])
    CategoricalTestResults = namedtuple('CategoricalTestResults', ['column', 'association_stat', 'test_type'])

    continuous_results = []
    categorical_results = []

    for col in continuous_cols + categorical_cols:
        if col in continuous_cols:
            # Continuous variable columns: Point Biserial Correlation Coefficient
            y = dataframe[col].dropna()
            x = dataframe.loc[y.index, target_column]
            biserial_corr, p_value = pointbiserialr(x, y)
            continuous_results.append(continuousTestResults(col, biserial_corr, p_value, 'biserial_corr'))
        elif col in categorical_cols:
            # Categorical variable columns: Association
            contingency_table = pd.crosstab(dataframe[target_column], dataframe[col])
            assoc_stat = contingency.association(contingency_table, method='cramer', correction=False)
            categorical_results.append(CategoricalTestResults(col, assoc_stat, 'cramers v'))

    # Converting results into DataFrames
    continuous_results_df = pd.DataFrame(continuous_results)
    categorical_results_df = pd.DataFrame(categorical_results)

    # Save the results to an Excel file
    file_name = f'{output_dir}/{target_column}_{match_flag}_effect_eval_statistics.xlsx'
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        continuous_results_df.to_excel(writer, sheet_name='Continuous_Statistics')
        categorical_results_df.to_excel(writer, sheet_name='Categorical_Statistics')


