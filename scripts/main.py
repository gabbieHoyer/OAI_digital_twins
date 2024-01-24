import pandas as pd
from data_loading import load_all_clin_data, load_clean_oai_data
from data_preprocessing import process_all_clin, derive_process_tkr, merge_dataframes, process_right_side_data, get_baseline_data, select_columns
from data_preprocessing import drop_rows_missing_last_n_columns, drop_columns_with_missing_values, impute_by_group, scale_columns
from OAI_digital_twins.scripts.statistics.statistical_analysis import ks_test_by_group_to_excel
from statistical_matching import prepare_dataframe_for_matching, perform_matching
from tsne_matching import perform_tsne, calculate_distances, find_closest_pairs  # or from statistical_matching import ...


def main():
    base_path = '/content/drive/MyDrive/Colab_Notebooks/oai/TKR_twin/'
    all_clin = load_all_clin_data(base_path)
    clean_oai = load_clean_oai_data(base_path)

    all_clin = process_all_clin(all_clin)
    tkr, tkr_ids, tkr_right, tkr_right_ids = derive_process_tkr(clean_oai)
    oai_extra = merge_dataframes(clean_oai, all_clin)
    right = process_right_side_data(oai_extra)
    baseline_clean_oai = get_baseline_data(right)
    oai_particular = select_columns(baseline_clean_oai)

    # Call the new preprocessing functions
    df_no_missing, df_with_missing = drop_rows_missing_last_n_columns(oai_particular, 110)
    df_no_missing_dropped = drop_columns_with_missing_values(df_no_missing)

    # Imputation
    df = df_no_missing_dropped.copy()
    df_imputed_tkr = impute_by_group(df, 'tkr')
    df_imputed_oa_inc = impute_by_group(df, 'oa_prog')

    
    # Assuming df is the original DataFrame and df_imputed_tkr is the DataFrame after imputation
    # lols what
    columns_to_test = df_imputed_tkr.columns[7:16].tolist() + df_imputed_tkr.columns[16:-110].tolist()
    # columns_to_test = df_imputed_oa_inc.columns[7:16].tolist() + df_imputed_oa_inc.columns[16:-110].tolist()

    # File path for the Excel output
    excel_file_path = 'publish_dataframes/tkr_ks_test_results.xlsx'
    # excel_file_path = 'publish_dataframes/oa_inc_ks_test_results.xlsx'

    # Use the function for 'tkr' column and save results to Excel
    ks_results_tkr = ks_test_by_group_to_excel(df, df_imputed_tkr, columns_to_test, 'tkr', excel_file_path)
    # ks_results_oa_inc = ks_test_by_group_to_excel(df, df_imputed_oa_inc, columns_to_test, 'oa_prog', excel_file_path)

    # Print a preview of the results
    # print(ks_results_tkr.head())

    # Scaling columns for df_imputed_tkr
    transformed_tkr = scale_columns(df_imputed_tkr, 16, -110)

    # Scaling columns for df_imputed_oa_inc
    transformed_oa_inc = scale_columns(df_imputed_oa_inc, 16, -110)

    # Filtering to only include patients who did not have OA to start with but may progress
    transformed_oa_inc_control = transformed_oa_inc[(transformed_oa_inc['pred_kl']==0) | (transformed_oa_inc['pred_kl']==1)]

    # Check unique values in 'pred_kl'
    # print(transformed_oa_inc_control['pred_kl'].unique())   # Should output: array([1., 0.])

    # Preparing dataframes for matching
    tkr_covariate_df = prepare_dataframe_for_matching(transformed_tkr, 'tkr', slice(7, -110), 'id')
    oa_inc_covariate_df = prepare_dataframe_for_matching(transformed_oa_inc_control, 'oa_prog', slice(7, -110), 'id')

    # Ensuring 'id' column is of string type
    tkr_covariate_df['id'] = tkr_covariate_df['id'].astype('str')
    oa_inc_covariate_df['id'] = oa_inc_covariate_df['id'].astype('str')

    #******** Matching with R Matchit ********#
    
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
    tsne_params = {'n_components': 3, 'perplexity': 50, 'n_iter': 4000}

    # Process for transformed_oa_inc_control dataframe
    transformed_oa_inc_control = pd.read_csv('publish_dataframes/control_oa_inc_standardized_df.csv')
    tsne_df_oa_inc = perform_tsne(transformed_oa_inc_control, slice(7, 27), tsne_params)
    dist_df_oa_inc = calculate_distances(tsne_df_oa_inc, 'id')
    matched_oa_inc_df = find_closest_pairs(transformed_oa_inc_control, 'oa_prog', dist_df_oa_inc, 'id')

    # Process for transformed_tkr dataframe
    transformed_tkr = pd.read_csv('publish_dataframes/tkr_standardized_df.csv')
    tsne_df_tkr = perform_tsne(transformed_tkr, slice(7, 27), tsne_params)
    dist_df_tkr = calculate_distances(tsne_df_tkr, 'id')
    matched_tkr_df = find_closest_pairs(transformed_tkr, 'tkr', dist_df_tkr, 'id')


if __name__ == "__main__":
    main()
