import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_all_clin(all_clin):
    all_clin[['id', 'side']] = all_clin['Knee'].str.split("_", expand=True)
    all_clin['side'] = all_clin['side'].replace({'R': 'RIGHT', 'L': 'LEFT'})
    all_clin['id'] = pd.to_numeric(all_clin['id'])
    all_clin.drop(columns=['Knee', 'Age', 'Gender', 'BMI', 'Postmeno', 'KL'], inplace=True)
    return all_clin

def derive_process_tkr(clean_oai):
    tkr = clean_oai[['id', 'side', 'visit', 'total_or_partial']]
    tkr_ids = tkr[tkr['total_or_partial'].notna()]['id']
    tkr_right = tkr[tkr['side'] == 'RIGHT']
    tkr_right_ids = tkr_right[tkr_right['total_or_partial'].notna()]['id']
    return tkr, tkr_ids, tkr_right, tkr_right_ids

def merge_dataframes(clean_oai, all_clin):
    oai_extra = pd.merge(clean_oai, all_clin, on=['id', 'side']).rename(columns={'max_kl': 'oa_prog'})
    return oai_extra

def process_right_side_data(oai_extra):
    right = oai_extra[oai_extra['side'] == 'RIGHT']
    right['total_or_partial'].fillna(0, inplace=True)
    right['tkr'] = right['total_or_partial'].apply(lambda x: 0 if x == 0 else 1)
    return right

def get_baseline_data(right):
    baseline_clean_oai = right[right['visit'] == 'V00']
    return baseline_clean_oai

def select_columns(baseline_clean_oai):
    specific_cols = [...]  # Define your columns here
    pattern_based_cols_1 = [...]  # Define your columns here
    pattern_based_cols_2 = [...]  # Define your columns here
    all_columns = specific_cols + pattern_based_cols_1 + pattern_based_cols_2
    oai_particular = baseline_clean_oai[all_columns]
    oai_particular.reset_index(inplace=True, drop=True)
    return oai_particular

def drop_rows_missing_last_n_columns(df, n):
    columns_to_check = df.iloc[:, -n:]
    df_no_missing = df.dropna(subset=columns_to_check.columns)
    df_with_missing = df[df.iloc[:, -n:].isna().any(axis=1)]
    return df_no_missing, df_with_missing

def drop_columns_with_missing_values(df, threshold=5):
    missing_percentage = df.iloc[:,:-110].isnull().mean() * 100
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    return df.drop(columns=columns_to_drop)

def impute_by_group(df, group_col):
    # (Include your function definition here as it is)
    ...

def scale_columns(df, start_col_idx, end_col_idx):
    # (Include your function definition here as it is)
    ...

# Other preprocessing functions
