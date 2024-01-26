import argparse
import pandas as pd
import numpy as np

from utils.utils import load_and_merge_data
from statistics.descriptive_statistics import calculate_group_stats_and_save
from statistics.effect_size_statistics import calculate_effect_size_stats_and_save
from statistics.non_parametric_tests import calculate_nonparametric_stats_and_save


def main(args):
    # Set the random seed for reproducibility
    np.random.seed(42)

    target = args.target_variable

    group_mapping = {
        'oa_prog': {0: 'Control', 1: 'OA_Inc Group'},
        'tkr': {0: 'Control', 1: 'TKR Group'}
    }

    if target in group_mapping:
        group_names = group_mapping[target]
    else:
        raise ValueError("Unknown target variable")

    # *********** Data loading *********** #
    # Before Matching
    before_match_df = pd.read_csv(args.imputed_data_path)
    before_continuous_cols = before_match_df.iloc[:,16:-110].columns.tolist()
    before_categorical_cols = before_match_df.iloc[:,7:16].columns.tolist()

    # combine Cohort Match IDs with original clinical factors and PC modes
    matched_cohort_df = load_and_merge_data(args.imputed_data_path, args.match_ids_path)
    after_continuous_cols = matched_cohort_df.iloc[:,17:-110].columns.tolist()
    after_categorical_cols = matched_cohort_df.iloc[:,8:17].columns.tolist()

    # ******** Descriptive Statistics ******** 
    calculate_group_stats_and_save(before_match_df, target, group_names, before_continuous_cols, before_categorical_cols,  args.output_dir, match_flag='before',)
    calculate_group_stats_and_save(matched_cohort_df, target, group_names, after_continuous_cols, after_categorical_cols, args.output_dir, match_flag='after')

    # ******** Covariate Effect Evalutation Stats ******** #
    calculate_effect_size_stats_and_save(before_match_df, target, before_continuous_cols, before_categorical_cols, args.output_dir, match_flag='after')
    calculate_effect_size_stats_and_save(matched_cohort_df, target, after_continuous_cols, after_categorical_cols, args.output_dir, match_flag='after')

    # ******** Non-Parametric Statistics ******** #
    calculate_nonparametric_stats_and_save(before_match_df, target, before_continuous_cols, before_categorical_cols, args.output_dir)
    calculate_nonparametric_stats_and_save(matched_cohort_df, target, after_continuous_cols, after_categorical_cols, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical modeling for OA Incidence data.")
    parser.add_argument("match_ids_path", help="Path to the input CSV file containing the dataset")
    parser.add_argument("imputed_data_path", help="Path to the imputed data CSV file")
    
    parser.add_argument("output_dir", help="Path to the output directory where results will be saved")
    parser.add_argument("target_variable", choices=['oa_prog', 'tkr'], help="Target variable for the analysis (oa_prog or tkr)")

    args = parser.parse_args()
    main(args)


