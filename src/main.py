import pandas as pd
import numpy as np
from pathlib import Path
import preprocess as pp
from summary_utils.y_variable_analysis import analyze_y_variables


def load_data(file_path):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    # Create output directories
    Path('output_data').mkdir(exist_ok=True)
    Path('output_data/plots').mkdir(exist_ok=True)
    Path('output_data/stats').mkdir(exist_ok=True)

    df = load_data('input_data/universe_with_affactor.csv')

    # Step 1: Missing Value Treatment
    # Remove columns with more than 20% missing values to ensure data quality
    # Returns cleaned dataframe and list of dropped columns
    df_clean0 = pp.missing_value_treatment(df, 20)[0]

    # Step 2: Duplicate Detection
    # Identify groups of near-duplicate columns based on naming patterns
    # Helps in understanding variable relationships and potential redundancy
    duplicate_groups = pp.extract_near_duplicate_groups(df_clean0.columns.tolist())

    # Step 3: Remove Fuzzy Variables
    # Drop predefined fuzzy variables and metadata columns
    # Creates quarter_factor for seasonality
    df_clean1 = pp.drop_fuzzy_variables(df_clean0)

    # Step 4: Categorical Encoding
    # Convert categorical variables to dummy variables
    # Handles: companyid, securityid, simpleindustryid, calendaryear
    df_clean2 = pp.encode_categorical_variables(df_clean1)

    # Step 5: Correlation Analysis
    # Remove highly correlated variables (threshold > 0.8)
    # Keeps variables with stronger correlation to target
    df_clean3 = pp.drop_high_corr(df_clean2)

    # Step 6: Outlier Treatment
    # Apply winsorization to financial variables
    # Reduces impact of extreme values while preserving data structure
    df_clean4 = pp.winsorize_financial_variables(df_clean3)

    # Step 7: Variable Standardization
    # Z-score standardization of numeric features
    # Excludes: dummy variables, quarter_factor, special variables
    df_clean5 = pp.standardize_numeric_variables(df_clean4)
    
    # Step 8: Variable Classification
    # Add prefixes to identify variable types:
    # - x_ for features
    # - y_ for targets
    # - dummy_ for categorical
    df_clean6 = pp.rename_variables_xy(df_clean5)

    # Step 9: Save Processed Data
    df_clean6.to_csv('output_data/cleaned_data.csv', index=False)

    # Step 10: Y-Variable Analysis
