import pandas as pd
from scripts.data_processing import process_all_clin, filter_data_by_side_and_visit, select_columns
from scripts.data_processing import drop_rows_missing_pc_columns, drop_columns_with_missing_values, impute_by_group, scale_columns
from scripts.data_processing import prepare_dataframe_for_matching
from scripts.statistics.sensitivity_analysis import ks_test_by_group_to_excel
from statistical_matching import perform_matching
from tsne_matching import perform_tsne, calculate_distances, find_closest_pairs  

def main():
    base_path = '/content/drive/MyDrive/Colab_Notebooks/oai/TKR_twin/'
    all_clin = pd.read_csv(base_path + 'OAI_all_knees_data.csv')
    clean_oai = pd.read_csv(base_path + 'publish_dataframes/pca_modes_and_demos_all_timepoints_12082023.csv')

    all_clin = process_all_clin(all_clin)
    oai_extra = pd.merge(clean_oai, all_clin, on=['id', 'side']).rename(columns={'max_kl': 'oa_prog'})

    # prep knee replacement column - random missingness
    oai_extra['total_or_partial'].fillna(0, inplace=True)
    oai_extra['tkr'] = oai_extra['total_or_partial'].apply(lambda x: 0 if x == 0 else 1)
    
    # Filter for 'RIGHT' side and 'V00' visit
    right_baseline_df = filter_data_by_side_and_visit(oai_extra, 'RIGHT', 'V00')

    # Define your specific columns
    my_specific_cols = [
        'id', 'side', 'pred_kl', 'KL', 'oa_prog', 'total_or_partial', 'tkr',
        'hisp', 'race', 'gender', 'Varus', 'Tenderness', 'Injury_history',
        'Mild_symptoms', 'Heberden', 'Crepitus', 'Morning_stiffness',
        'age', 'height', 'weight', 'BMI', 'womac_pain', 'womac_adl',
        'womac_stiff', 'womac_total', 'koos_pain', 'koos_symptom', 'koos_func', 'koos_qol'
    ]

    # Call the function with your DataFrame and specific columns
    oai_particular = select_columns(right_baseline_df, my_specific_cols)

    # Handle missing data in the principal component columns
    df_no_missing_pc, df_with_missing = drop_rows_missing_pc_columns(oai_particular, 110)
    
    # Handle excessive missing data in all other data columns
    columns_to_check = oai_particular.iloc[:, -110:].columns.tolist()
    df_no_missing_dropped = drop_columns_with_missing_values(df_no_missing_pc, columns=columns_to_check, threshold=5)

    # Mulitiple Imputation 
    df = df_no_missing_dropped.copy()

    # Choose columns to impute the remaining minimal missing data
    custom_categorical_columns = df.columns[7:16]  # Your categorical columns
    custom_continuous_columns = df.columns[16:-110]    # Your continuous columns

    df_imputed_tkr = impute_by_group(df, 'tkr', categorical_columns=custom_categorical_columns, numerical_columns=custom_continuous_columns)
    df_imputed_oa_inc = impute_by_group(df, 'oa_prog', categorical_columns=custom_categorical_columns, numerical_columns=custom_continuous_columns)

    # ****************** Sensitivity Analysis ****************** #
    # Sensitivity Test to evaluate the effects of our imputation method
    # File path for the Excel output
    # excel_file_path = 'publish_dataframes/tkr_ks_test_results.xlsx'
    # excel_file_path = 'publish_dataframes/oa_inc_ks_test_results.xlsx'

    # Use the function for 'tkr' column and save results to Excel
    # columns_to_test = df_imputed_tkr.columns[7:-110].tolist()
    # ks_results_tkr = ks_test_by_group_to_excel(df, df_imputed_tkr, columns_to_test, 'tkr', excel_file_path)
    # ks_results_oa_inc = ks_test_by_group_to_excel(df, df_imputed_oa_inc, columns_to_test, 'oa_prog', excel_file_path)
    # ********************************************************* #
    
    # Scaling columns for df_imputed_tkr
    transformed_tkr = scale_columns(df_imputed_tkr, 16, -110)

    # Scaling columns for df_imputed_oa_inc
    transformed_oa_inc = scale_columns(df_imputed_oa_inc, 16, -110)

    # Filtering to only include patients who did not have OA to start with but may progress
    transformed_oa_inc_control = transformed_oa_inc[(transformed_oa_inc['pred_kl']==0) | (transformed_oa_inc['pred_kl']==1)]

    # Preparing dataframes for matching
    tkr_covariate_df = prepare_dataframe_for_matching(transformed_tkr, 'tkr', slice(7, -110), 'id')
    oa_inc_covariate_df = prepare_dataframe_for_matching(transformed_oa_inc_control, 'oa_prog', slice(7, -110), 'id')

    # Ensuring 'id' column is of string type
    tkr_covariate_df['id'] = tkr_covariate_df['id'].astype('str')
    oa_inc_covariate_df['id'] = oa_inc_covariate_df['id'].astype('str')

    # ****************** Matching with R Matchit ****************** #
    
    # Define parameters for CEM matching
    cem_params = {
        'cuts': 30,
        'M': 1.0,
        'weighting': 'TRUE',
        # Additional parameters here
    }

    # Perform matching for the first dataframe
    input_df_tkr = pd.read_csv('publish_dataframes/tkr_covariate_df.csv')
    output_csv_tkr = 'publish_dataframes/tkr_CEM_noRep_df.csv'
    output_matched_csv_tkr = 'publish_dataframes/tkr_temp_matched_data.csv'
    perform_matching(input_df_tkr, 'tkr', output_csv_tkr, output_matched_csv_tkr, method='cem', params=cem_params)

    # Perform matching for the second dataframe
    input_df_oa_inc = pd.read_csv('publish_dataframes/oa_inc_covariate_df.csv')
    output_csv_oa_inc = 'publish_dataframes/oa_inc_CEM_noRep_df.csv'
    output_matched_csv_oa_inc = 'publish_dataframes/oa_inc_temp_matched_data.csv'
    perform_matching(input_df_oa_inc, 'oa_prog', output_csv_oa_inc, output_matched_csv_oa_inc, method='cem', params=cem_params)

    #********** TSNE Matching ************#
    oa_tsne_params = {'n_components': 3, 'perplexity': 50, 'n_iter': 4000}
    tkr_tsne_params = {'n_components': 3, 'perplexity': 65, 'n_iter': 4000, 'random_state' : 42}  # perp is sqrt(N) -> sqrt(4283) ~ 65

    # Process for transformed_oa_inc_control dataframe
    transformed_oa_inc_control = pd.read_csv('publish_dataframes/control_oa_inc_standardized_df.csv')
    tsne_df_oa_inc = perform_tsne(transformed_oa_inc_control, slice(7, 27), oa_tsne_params)
    dist_df_oa_inc = calculate_distances(tsne_df_oa_inc, 'id')
    matched_oa_inc_df = find_closest_pairs(transformed_oa_inc_control, 'oa_prog', dist_df_oa_inc, 'id')
    # matched_oa_inc_df.to_csv('/oai/oa_inc_matchit_TSNE_EuclideanDist_Replacement_df.csv', index=False)

    # Process for transformed_tkr dataframe
    transformed_tkr = pd.read_csv('publish_dataframes/tkr_standardized_df.csv')
    tsne_df_tkr = perform_tsne(transformed_tkr, slice(7, 27), tkr_tsne_params)
    dist_df_tkr = calculate_distances(tsne_df_tkr, 'id')
    matched_tkr_df = find_closest_pairs(transformed_tkr, 'tkr', dist_df_tkr, 'id')
    # matched_tkr_df.to_csv('/oai/tkr_matchit_TSNE_EuclideanDist_Replacement_df.csv', index=False)


if __name__ == "__main__":
    main()
