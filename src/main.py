import pandas as pd
from pathlib import Path
import preprocess as pp
from summary_utils.y_variable_analysis import analyze_y_variables


def load_data(file_path):
    return pd.read_parquet(file_path)


if __name__ == "__main__":
    # Create output directories
    Path('output_data').mkdir(exist_ok=True)
    Path('output_data/plots').mkdir(exist_ok=True)
    Path('output_data/stats').mkdir(exist_ok=True)

    df = load_data('input_data/universe_with_affactor.parquet')
    print(df)

    # Step 1: Missing Value Treatment
    # Remove columns with more than 50% missing values while preserving essential EPS and revenue metrics
    # Returns cleaned dataframe and list of dropped columns
    # Key metrics like EPS_actual, EPSDiff, revenue_actual etc. are retained regardless of missing percentage
    df_clean0 = pp.missing_value_treatment(df,50)[0]
    print(df_clean0)

    # Step 2: Duplicate Detection
    # Identify groups of near-duplicate columns based on naming patterns
    # Helps in understanding variable relationships and potential redundancy
    duplicate_groups = pp.extract_near_duplicate_groups(df_clean0.columns.tolist())

    # Step 3: Remove Fuzzy Variables
    # Drop predefined fuzzy variables and metadata columns
    # Removes affactor_asofdate_* variables and other metadata
    # Creates quarter_factor for seasonality
    df_clean1 = pp.drop_fuzzy_variables(df_clean0)
    print(df_clean1)

    # Step 4: Categorical Encoding
    # Convert categorical variables to dummy variables with 'dummy_' prefix
    # Processes: companyid, securityid, simpleindustryid, calendaryear
    df_clean2 = pp.encode_categorical_variables(df_clean1)
    print(df_clean2)

    # Step 5: Correlation Analysis
    # Remove highly correlated variables (threshold > 0.8)
    # Preserves EPS and revenue-related variables regardless of correlation
    # For other variables, keeps those with stronger correlation to target surprise metrics
    df_clean3 = pp.drop_high_corr(df_clean2)
    print(df_clean3)

    # Step 6: Outlier Treatment
    # Apply winsorization to outcome variables
    df_clean4 = pp.winsorize_financial_variables(df_clean3)
    print(df_clean4)

    # Step 7: Variable Standardization
    # Z-score standardization of numeric features (mean=0, std=1)
    # Excludes:
    # - Dummy variables (dummy_*)
    # - Categorical indicators (quarter_factor)
    # - ID fields (companyid)
    # - Key financial metrics (EPS_*, revenue_* variables)
    df_clean5 = pp.standardize_numeric_variables(df_clean4)
    print(df_clean5)

    # Step 8: Variable Classification
    # Add prefixes to identify variable types:
    # - x_ for feature variables (standardized numeric variables)
    # - y_ for target variables (EPS and revenue metrics)
    # - dummy_ for categorical variables (one-hot encoded)
    # Special variables (companyid) remain unprefixed
    df_clean6 = pp.rename_variables_xy(df_clean5)

    # Step 9: Save Processed Data
    df_clean6.to_csv('output_data/cleaned_data.csv', index=False)
    print(df_clean6)

    # Step 10: Y-Variable Analysis
    # Analyze distributions and relationships of target variables
