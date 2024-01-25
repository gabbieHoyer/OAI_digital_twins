import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from feature_selection.bootstrap_functions import filter_covariates, bootstrap_elastic_net_with_feature_selection

def main(args):
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Data loading
    oa_inc_matched_df = pd.read_csv(args.data_path)

    # Filtering covariates
    confounders = ['height', 'weight', 'womac_pain', 'womac_adl', 'womac_stiff', 'womac_total']
    target = args.target_variable
    df = filter_covariates(oa_inc_matched_df, target, confounders, slice_start=8)

    # Shuffling and splitting data
    df_shuffled = shuffle(df, random_state=42)
    X = df_shuffled.drop(target, axis=1)
    y = df_shuffled[target]

    # Bootstrap Elastic Net
    bootstrap_results, feature_selection_df = bootstrap_elastic_net_with_feature_selection(df_shuffled, X, y, n_bootstrap=1000, maxiter=2000, checkpoint_interval=50)

    # Saving results
    coefficients_output = f"{args.output_dir}/bootstrap_elasticNet_coef_raw.csv"
    coefficients_df = pd.DataFrame(bootstrap_results).T
    coefficients_df.to_csv(coefficients_output, index=True)

    # Feature selection frequencies
    feature_selection_output = f"{args.output_dir}/bootstrap_elasticNet_feature_selection_frequencies.csv"
    feature_selection_df.to_csv(feature_selection_output, index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical modeling for OA Incidence data.")
    parser.add_argument("data_path", help="Path to the input CSV file containing the dataset")
    parser.add_argument("output_dir", help="Path to the output directory where results will be saved")
    parser.add_argument("target_variable", choices=['oa_prog', 'tkr'], help="Target variable for the analysis (oa_prog or tkr)")

    args = parser.parse_args()
    main(args)
