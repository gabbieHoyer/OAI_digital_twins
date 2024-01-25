import pandas as pd
from rpy2.robjects import r

def calculate_distance(input_dataframe, method='euclidean'):
    # Add logic to select only covariate columns if needed
    # Example: input_dataframe = input_dataframe.iloc[:, 2:]

    # Convert the DataFrame to CSV for R
    temp_covariates_csv = 'temp_covariates_data.csv'
    input_dataframe.to_csv(temp_covariates_csv, index=False)

    # R script to calculate distance
    r_script = f'''
    df_covariates <- read.csv("{temp_covariates_csv}")
    euclidean_dist_matrix <- as.matrix(dist(df_covariates, method = "{method}"))
    '''

    # Execute R code
    r(r_script)

    # Return the name of the R object holding the distance matrix
    return 'euclidean_dist_matrix'


def perform_matching(input_dataframe, target_column, output_csv, output_matched_csv, method='nearest', params=None,  distance_method=None):
    print("Starting matching process...")

    # Convert IDs to string
    input_dataframe['id'] = input_dataframe['id'].astype('str')

    # Save the DataFrame as a CSV file for R
    temp_csv = 'temp_input_data.csv'
    input_dataframe.to_csv(temp_csv, index=False)
    print("Dataframe saved as CSV for R.")

    # Prepare the parameters string from the dictionary
    params_string = ", ".join([f"{key} = {value}" for key, value in params.items()]) if params else ""

    # Check if a specific distance calculation is needed
    if distance_method:
        distance_matrix_var = calculate_distance(input_dataframe, method=distance_method)
        params_string += f", distance = {distance_matrix_var}" if params_string else f"distance = {distance_matrix_var}"

    # Prepare the R script with dynamic target column and custom parameters
    r_script = f'''
    library(MatchIt)
    library(rgenoud)
    library(Matching)

    print("R libraries loaded. Reading data into R...")

    # Read the data into R
    input_data_r <- read.csv("{temp_csv}")

    print("Data read into R. Starting matching process with method '{method}' and custom parameters...")

    # Perform matching using MatchIt with specified method and custom parameters
    m.out <- matchit({target_column} ~ ., data = input_data_r, method = "{method}", {params_string})

    print("Matching completed. Retrieving matched data...")

    # Retrieve the matched data with IDs
    matched_data <- get_matches(m.out, id = "new_id")

    print("Matched data retrieved. Writing to CSV...")

    # Write the matched data to a CSV file
    write.csv(matched_data, "{output_matched_csv}", row.names = FALSE)
    '''

    # Execute R code including library loading
    print("Executing R code for matching...")
    r(r_script)

    print("R processing completed. Reading matched data back into Pandas...")

    # Read the matched data back into a Pandas DataFrame
    matched_data_df = pd.read_csv(output_matched_csv)
    matched_data_df.to_csv(output_csv, index=False)

    print("Matching process completed.")


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
