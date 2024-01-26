import argparse
import pandas as pd
import numpy as np
import statsmodel as sm
from sklearn.utils import shuffle
from feature_selection.bootstrap_functions import bootstrap_elastic_net_with_feature_selection
from feature_selection.coefficient_analysis import calculate_feature_stat_summary, select_top_features, calculate_odds_ratios_and_stats, extract_significant_variables
from utils.utils import load_and_merge_data, filter_covariates, save_model_reports

def main(args):
    # Set the random seed for reproducibility
    np.random.seed(42)

    # ******** Data loading ******** #
    # combine Cohort Match IDs with original clinical factors and PC modes
    matched_cohort_df = load_and_merge_data(args.imputed_data_path, args.match_ids_path)

    # Filtering covariates
    target = args.target_variable
    filtered_df = filter_covariates(matched_cohort_df, target)

    # Shuffling and splitting data
    df_shuffled = shuffle(filtered_df, random_state=42)
    X = df_shuffled.drop(target, axis=1)
    y = df_shuffled[target]

    # Bootstrap Elastic Net - checkpoint saving
    bootstrap_results, feature_selection_df = bootstrap_elastic_net_with_feature_selection(df_shuffled, X, y, n_bootstrap=1000, maxiter=2000, checkpoint_interval=50)

    # Saving results
    coefficients_output = f"{args.output_dir}/bootstrap_elasticNet_coef_raw.csv"
    coefficients_df = pd.DataFrame(bootstrap_results).T
    coefficients_df.to_csv(coefficients_output, index=True)

    # Feature selection frequencies
    feature_selection_output = f"{args.output_dir}/bootstrap_elasticNet_feature_selection_frequencies.csv"
    feature_selection_df.to_csv(feature_selection_output, index=True)

    # ********** Evaluating Feature Importance ********** #
    # Summary statistics of bootstrap GLM coefficients
    df_summary = calculate_feature_stat_summary(coefficients_df, feature_selection_df, args.output_dir)

    # Find Top 25% of biomarkers based on weighted_importance value + add back covariates
    top_biomarkers = select_top_features(df_summary, 0.75)

    # Filter multivariate dataframe with selected features + covariates
    X_selected = X[top_biomarkers]

    # Add a constant to the model (intercept)
    X_selected_with_const = sm.add_constant(X_selected)

    # Fit the GLM model with a binomial family
    glm_logit_model = sm.GLM(y, X_selected_with_const, family=sm.families.Binomial()).fit()

    # Save GLM statsmodel summaries
    save_model_reports(glm_logit_model, target, args.output_dir)

    # Compute and save odds ratio
    odds_ratio_df = calculate_odds_ratios_and_stats(glm_logit_model, args.output_dir)

    # Identify and save significant biomarkers
    significant_vars_df = extract_significant_variables(glm_logit_model, 0.05, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical modeling for OA Incidence data.")
    parser.add_argument("match_ids_path", help="Path to the input CSV file containing the dataset")
    parser.add_argument("imputed_data_path", help="Path to the imputed data CSV file")
    
    parser.add_argument("output_dir", help="Path to the output directory where results will be saved")
    parser.add_argument("target_variable", choices=['oa_prog', 'tkr'], help="Target variable for the analysis (oa_prog or tkr)")

    args = parser.parse_args()
    main(args)
