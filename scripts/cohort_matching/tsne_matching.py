import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

def perform_tsne(df, tsne_cols, tsne_params):
    """
    Performs t-SNE transformation on the specified columns of the DataFrame.

    :param df: DataFrame to be processed.
    :param tsne_cols: Columns to be used for t-SNE transformation (as a slice object).
    :param tsne_params: Dictionary of parameters for t-SNE.
    :return: DataFrame with t-SNE components added.
    """
    tsne = TSNE(**tsne_params)
    tsne_components = tsne.fit_transform(df.iloc[:, tsne_cols])
    tsne_df = df.copy()
    tsne_df[['tsne-one', 'tsne-two', 'tsne-three']] = tsne_components
    return tsne_df


def calculate_distances(df, id_col):
    """
    Calculates the pairwise Euclidean distances between rows in the DataFrame.

    :param df: DataFrame with t-SNE components.
    :param id_col: The column name that uniquely identifies each row.
    :return: DataFrame of distances.
    """
    dist_df = pd.DataFrame(
        squareform(pdist(df[['tsne-one', 'tsne-two', 'tsne-three']])),
        columns=df[id_col].unique(),
        index=df[id_col].unique()
    )
    return dist_df


def find_closest_pairs(df, target_col, dist_df, id_col):
    """
    Identifies the closest pairs in the dataset.

    :param df: Original DataFrame with target information.
    :param target_col: Target column name.
    :param dist_df: DataFrame with distances.
    :param id_col: The column name that uniquely identifies each row.
    :return: DataFrame with closest pair information.
    """
    target_ids = df[df[target_col] == 1][id_col].tolist()
    progressing_dist = dist_df[target_ids]

    # Creating Long Format DataFrame
    progressing_long = progressing_dist.stack().reset_index().rename(columns={'level_0':'id','level_1':'pt2',0:'distance'})

    # Merging with Target Column Information
    target_vals = df[['id', target_col]]
    target_long = progressing_long.merge(target_vals, on='id').rename(columns={'id':'pt1_id', 'pt2':'id', target_col: target_col + '_pt1'})
    target_2pat = target_long.merge(target_vals, on='id').rename(columns={'id':'pt2_id', target_col: target_col + '_pt2'})

    # Identifying Different Pairs
    min_sort_target = (target_2pat.sort_values(['pt1_id','distance'], ascending=True)[['pt1_id','pt2_id','distance', target_col+'_pt1', target_col+'_pt2']])
    min_sort_target['diff_pair'] = (min_sort_target[target_col+'_pt1'] != min_sort_target[target_col+'_pt2'])

    # Finding the closest different pairs for each 'pt2_id'
    closest_pairs = min_sort_target[min_sort_target['diff_pair']].sort_values('distance').groupby('pt2_id').first().reset_index()

    # Dropping the 'diff_pair' column as it's no longer needed
    closest_pairs = closest_pairs.drop('diff_pair', axis=1)

    # Prepare for melting: Identify columns ending with '_id'
    id_cols = [col for col in closest_pairs.columns if col.endswith('_id')]

    # Melting the DataFrame to long format
    twins_melt = pd.melt(closest_pairs, id_vars='distance', value_vars=id_cols, value_name='id')

    # Assigning 'tkr' values based on the 'variable' column
    twins_melt[target_col] = np.where(twins_melt['variable'] == 'pt1_id', 0, 1)

    # Dropping the 'variable' column as it's no longer needed
    twins_melt = twins_melt.drop('variable', axis=1)

    return twins_melt



#**********
tsne_params = {'n_components': 3, 'perplexity': 50, 'n_iter': 4000}

transformed_oa_inc_control = pd.read_csv('publish_dataframes/control_oa_inc_standardized_df.csv')

df = transformed_oa_inc_control
target_col = 'oa_prog'    # target column
tsne_cols = slice(7, 27)  # column slice for t-SNE

# Perform t-SNE transformation
tsne_df = perform_tsne(df, tsne_cols, tsne_params)

# Calculate distances
dist_df = calculate_distances(tsne_df, 'id')

# Find closest pairs
matched_oa_inc_df = find_closest_pairs(df, target_col, dist_df, 'id')

#**************
tsne_params = {'n_components': 3, 'perplexity': 50, 'n_iter': 4000}

transformed_tkr = pd.read_csv('publish_dataframes/tkr_standardized_df.csv')

df = transformed_tkr
target_col = 'tkr'    # target column
tsne_cols = slice(7, 27)  # column slice for t-SNE

# Perform t-SNE transformation
tsne_df = perform_tsne(df, tsne_cols, tsne_params)

# Calculate distances
dist_df = calculate_distances(tsne_df, 'id')

# Find closest pairs
matched_tkr_df = find_closest_pairs(df, target_col, dist_df, 'id')