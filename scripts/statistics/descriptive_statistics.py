import pandas as pd
import numpy as np
import openpyxl
import xlsxwriter

def descriptive_group_stats(group_data, continuous_cols, categorical_cols):
    # Continuous Variable Descriptive Statistics
    continuous_desc = group_data[continuous_cols].describe()
    continuous_iqr = group_data[continuous_cols].apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
    continuous_iqr = pd.DataFrame(continuous_iqr, columns=['IQR'])
    continuous_desc.loc['median'] = continuous_desc.loc['50%']
    continuous_desc.loc['IQR'] = continuous_iqr['IQR']

    # Categorical Variable Descriptive Statistics
    combined_categorical_stats = pd.DataFrame(columns=['Variable', 'Category', 'N', '%'])
    for cat_var in categorical_cols:
        counts = group_data[cat_var].value_counts(normalize=False)
        percents = group_data[cat_var].value_counts(normalize=True) * 100
        temp_df = pd.DataFrame({
            'Variable': [cat_var] * len(counts),
            'Category': counts.index,
            'N': counts.values,
            '%': percents.values,
        })
        combined_categorical_stats = pd.concat([combined_categorical_stats, temp_df], ignore_index=True)

    return continuous_desc.loc[['count', 'mean', 'median', 'std', 'IQR']], combined_categorical_stats

def calculate_group_stats_and_save(dataframe, group_column, group_names, continuous_cols, categorical_cols, output_dir, match_flag):
    
    file_name = f'{output_dir}/{group_column}_{match_flag}_descriptive_statistics.xlsx'

    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        # Extracting unique values to define groups
        unique_groups = dataframe[group_column].unique()

        for group in unique_groups:
            subset = dataframe[dataframe[group_column] == group]

            continuous_stats, categorical_stats = descriptive_group_stats(subset, continuous_cols, categorical_cols)

            # Use custom group name if provided, else use the group value as the name
            group_name = group_names.get(group, str(group))

            # Writing continuous and categorical statistics to Excel
            continuous_stats.to_excel(writer, sheet_name=f"{group_name}_Continuous")
            categorical_stats.to_excel(writer, sheet_name=f"{group_name}_Categorical")

        # Adding Overall Statistics
        continuous_stats, categorical_stats = descriptive_group_stats(dataframe, continuous_cols, categorical_cols)
        continuous_stats.to_excel(writer, sheet_name="Overall_Continuous")
        categorical_stats.to_excel(writer, sheet_name="Overall_Categorical")

