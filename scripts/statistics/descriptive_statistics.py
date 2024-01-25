import pandas as pd
import numpy as np
import openpyxl
import xlsxwriter

# Set the random seed for reproducibility
np.random.seed(42)

def calculate_group_stats(group_data, numerical_cols, categorical_cols):
    # Numerical Descriptive Statistics
    numerical_desc = group_data[numerical_cols].describe()
    numerical_iqr = group_data[numerical_cols].apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
    numerical_iqr = pd.DataFrame(numerical_iqr, columns=['IQR'])
    numerical_desc.loc['median'] = numerical_desc.loc['50%']
    numerical_desc.loc['IQR'] = numerical_iqr['IQR']

    # Categorical Descriptive Statistics
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

    return numerical_desc.loc[['count', 'mean', 'median', 'std', 'IQR']], combined_categorical_stats

def calculate_and_save_group_stats(dataframe, group_column, group_names, numerical_cols, categorical_cols, file_name):
    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        # Extracting unique values to define groups
        unique_groups = dataframe[group_column].unique()

        for group in unique_groups:
            subset = dataframe[dataframe[group_column] == group]

            numerical_stats, categorical_stats = calculate_group_stats(subset, numerical_cols, categorical_cols)

            # Use custom group name if provided, else use the group value as the name
            group_name = group_names.get(group, str(group))

            # Writing numerical and categorical statistics to Excel
            numerical_stats.to_excel(writer, sheet_name=f"{group_name}_Numerical")
            categorical_stats.to_excel(writer, sheet_name=f"{group_name}_Categorical")

        # Adding Overall Statistics
        numerical_stats, categorical_stats = calculate_group_stats(dataframe, numerical_cols, categorical_cols)
        numerical_stats.to_excel(writer, sheet_name="Overall_Numerical")
        categorical_stats.to_excel(writer, sheet_name="Overall_Categorical")



# # before matcching
# # OA Incidence example usage
# oa_inc_before_match = pd.read_csv('publish_dataframes/oa_inc_multiple_imputation_filled.csv')
# numerical_cols = oa_inc_before_match.iloc[:,16:-110].columns
# categorical_cols = oa_inc_before_match.iloc[:,7:16].columns
# group_names = {0: 'Control', 1: 'OA_Inc Group'}
# calculate_and_save_group_stats(oa_inc_before_match, 'oa_prog', group_names, numerical_cols, categorical_cols, "publish_dataframes/OA_Inc_before_matching_descriptive_statistics_output.xlsx")

# # TKR example usage
# tkr_before_match = pd.read_csv('publish_dataframes/tkr_multiple_imputation_filled.csv')
# numerical_cols = tkr_before_match.iloc[:,16:-110].columns
# categorical_cols = tkr_before_match.iloc[:,7:16].columns
# group_names = {0: 'Control', 1: 'TKR Group'}
# calculate_and_save_group_stats(tkr_before_match, 'tkr', group_names, numerical_cols, categorical_cols, "publish_dataframes/TKR_before_matching_descriptive_statistics_output.xlsx")

# # after matcching
# # OA Incidence example usage
# numerical_cols = oa_inc_matched_df.iloc[:,20:-110].columns
# categorical_cols = oa_inc_matched_df.iloc[:,11:20].columns
# group_names = {0: 'Control', 1: 'OA_Inc Group'}
# calculate_and_save_group_stats(oa_inc_matched_df, 'oa_prog', group_names, numerical_cols, categorical_cols, "publish_dataframes/OA_Inc_Twins_descriptive_statistics_output.xlsx")

# # TKR example usage
# numerical_cols = tkr_matched_df.iloc[:,20:-110].columns
# categorical_cols = tkr_matched_df.iloc[:,11:20].columns
# group_names = {0: 'Control', 1: 'TKR Group'}
# calculate_and_save_group_stats(tkr_matched_df, 'tkr', group_names, numerical_cols, categorical_cols, "publish_dataframes/TKR_Twins_descriptive_statistics_output.xlsx")
