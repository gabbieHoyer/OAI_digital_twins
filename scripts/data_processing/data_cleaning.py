import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def process_all_clin(all_clin):
    all_clin[['id', 'side']] = all_clin['Knee'].str.split("_", expand=True)
    all_clin['side'] = all_clin['side'].replace({'R': 'RIGHT', 'L': 'LEFT'})
    all_clin['id'] = pd.to_numeric(all_clin['id'])
    all_clin.drop(columns=['Knee', 'Age', 'Gender', 'BMI', 'Postmeno', 'KL'], inplace=True)
    return all_clin

def filter_data_by_side_and_visit(df, side, visit):
    """
    Filter a DataFrame by specified side and visit, and process the 'total_or_partial' column.

    Args:
    df (pd.DataFrame): The DataFrame to filter.
    side (str): The side to filter by (e.g., 'RIGHT', 'LEFT').
    visit (str): The visit code to filter by (e.g., 'V00', 'V01').

    Returns:
    pd.DataFrame: The filtered and processed DataFrame.
    """
    # Filter the DataFrame by the specified side and visit
    filtered_df = df[(df['side'] == side) & (df['visit'] == visit)]

    return filtered_df


def select_columns(baseline_clean_oai, specific_cols):
    # Pattern-based columns for 'bs', 't2', 'thick'
    pattern_based_cols_1 = [
        f"{prefix}_{part}_pc{num}"
        for prefix in ['bs', 't2', 'thick']
        for part in ['fem', 'pat', 'tib']
        for num in range(1, 11)
    ]
    # Separate pattern for 'med' and 'lat' columns
    pattern_based_cols_2 = [f"{prefix}_pc{num}" for prefix in ['med', 'lat'] for num in range(1, 11)]

    # Combine the lists
    all_columns = specific_cols + pattern_based_cols_1 + pattern_based_cols_2

    # Create the new DataFrame
    oai_particular = baseline_clean_oai[all_columns]
    oai_particular.reset_index(inplace=True, drop=True)
    
    return oai_particular

def drop_rows_missing_pc_columns(df, n):
    columns_to_check = df.iloc[:, -n:]
    df_no_missing = df.dropna(subset=columns_to_check.columns)
    df_with_missing = df[df.iloc[:, -n:].isna().any(axis=1)]
    return df_no_missing, df_with_missing

def drop_columns_with_missing_values(df, columns=None, threshold=5):
    # If no specific columns are provided, check all columns
    if columns is None:
        columns = df.columns

    # Calculate missing percentage for the specified columns
    missing_percentage = df[columns].isnull().mean() * 100
    
    # Identify columns to drop based on the threshold
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    
    # Drop the identified columns and return the modified DataFrame
    return df.drop(columns=columns_to_drop)

def impute_by_group(df, group_col, categorical_columns=None, numerical_columns=None):
    # Default columns for imputation if not provided
    if categorical_columns is None:
        categorical_columns = df.columns[7:16]
    if numerical_columns is None:
        numerical_columns = df.columns[16:-110]

    # Initialize IterativeImputer for numerical and categorical data
    imputer_num = IterativeImputer(estimator=RandomForestRegressor(), initial_strategy='median', max_iter=40, random_state=0)
    imputer_cat = IterativeImputer(estimator=RandomForestClassifier(), initial_strategy='most_frequent', max_iter=40, random_state=0)

    # Empty list to hold imputed parts
    imputed_parts = []

    # Indices to slice the DataFrame
    first_cat_index = df.columns.get_loc(categorical_columns[0])
    last_cat_index = df.columns.get_loc(categorical_columns[-1]) + 1
    first_num_index = df.columns.get_loc(numerical_columns[0])
    last_num_index = df.columns.get_loc(numerical_columns[-1]) + 1

    # Iterate over each group in specified column
    for name, group in df.groupby(group_col):
        # Separate categorical and numerical data
        categorical_data = group.iloc[:, first_cat_index:last_cat_index]
        numerical_data = group.iloc[:, first_num_index:last_num_index]

        # Impute numerical and categorical data
        numerical_data_imputed = imputer_num.fit_transform(numerical_data)
        categorical_data_imputed = imputer_cat.fit_transform(categorical_data)

        # Convert imputed data back to DataFrame
        numerical_data_imputed_df = pd.DataFrame(numerical_data_imputed, columns=numerical_columns, index=group.index)
        categorical_data_imputed_df = pd.DataFrame(categorical_data_imputed, columns=categorical_columns, index=group.index)

        # Combine imputed data with non-imputed data
        combined_data = pd.concat([
            group.iloc[:, :first_cat_index], 
            categorical_data_imputed_df, 
            numerical_data_imputed_df, 
            group.iloc[:, last_num_index:]
        ], axis=1)

        # Append combined data to the list
        imputed_parts.append(combined_data)

    # Concatenate all parts into one DataFrame and sort by index
    return pd.concat(imputed_parts).sort_index()

def scale_columns(df, start_col_idx, end_col_idx):
    """
    Scales specified columns of a dataframe based on given column indices.

    Parameters:
    df (pd.DataFrame): The dataframe to be scaled.
    start_col_idx (int): The starting index of the columns to be scaled.
    end_col_idx (int): The ending index of the columns to be scaled (exclusive).

    Returns:
    pd.DataFrame: A copy of the dataframe with specified columns scaled.
    """
    cols_to_scale = df.columns[start_col_idx:end_col_idx]

    sc = StandardScaler()
    sc.fit(df[cols_to_scale])

    transformed_df = df.copy()
    transformed_df[cols_to_scale] = sc.transform(transformed_df[cols_to_scale])

    return transformed_df

def prepare_dataframe_for_matching(dataframe, target_flag_name, covariates_range, id_column_name):
    """
    Prepares a DataFrame for matching by selecting the relevant columns.

    Parameters:
    dataframe (pd.DataFrame): The original DataFrame.
    target_flag_name (str): The name of the column containing the target flag.
    covariates_range (slice): The range of columns to be used as covariates.
    id_column_name (str): The name of the column containing the ID.

    Returns:
    pd.DataFrame: A new DataFrame with the selected columns.
    """

    # Select the covariates and the target flag
    covariates = dataframe.iloc[:, covariates_range]
    target_flag = dataframe[target_flag_name]
    id_column = dataframe[id_column_name]

    # Create a new DataFrame with only the relevant columns
    prepared_dataframe = pd.concat([id_column, target_flag, covariates], axis=1)

    return prepared_dataframe