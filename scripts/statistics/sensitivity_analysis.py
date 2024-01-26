import pandas as pd
from scipy.stats import ks_2samp

def ks_test_by_group_to_excel(original_df, imputed_df, columns_to_test, group_col, excel_file_path):

    # Initialize a list to store test results
    ks_test_results = []

    # Iterate over each group in the specified column
    for name, group in original_df.groupby(group_col):
        group_imputed = imputed_df[imputed_df[group_col] == name]

        for column in columns_to_test:
            original_data = group[column].dropna()  # Original data for the column in the group
            imputed_data = group_imputed[column]    # Imputed data for the column in the group

            # Conduct the Kolmogorov-Smirnov test
            ks_statistic, p_value = ks_2samp(original_data, imputed_data)

            # Append the results to the list
            ks_test_results.append({
                'Group': name,
                'Column': column,
                'KS Statistic': ks_statistic,
                'P-Value': p_value
            })

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(ks_test_results)

    # Save the results to an Excel file
    results_df.to_excel(excel_file_path, index=False)

    return results_df
