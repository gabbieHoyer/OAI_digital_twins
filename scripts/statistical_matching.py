import pandas as pd
from rpy2.robjects import r

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