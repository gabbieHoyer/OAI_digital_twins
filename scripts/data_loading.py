import pandas as pd

def load_all_clin_data(base_path):
    return pd.read_csv(base_path + 'OAI_all_knees_data.csv')

def load_clean_oai_data(base_path):
    return pd.read_csv(base_path + 'publish_dataframes/pca_modes_and_demos_all_timepoints_12082023.csv')
