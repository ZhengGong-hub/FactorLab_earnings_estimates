import pandas as pd
import numpy as np
from pathlib import Path
import re

from preprocess_utils.near_duplicates_var import extract_near_duplicate_groups
from preprocess_utils.missing_var_treat import missing_value_treatment
from preprocess_utils.drop_fuzzy_var import drop_fuzzy_variables
from preprocess_utils.categorize_var import encode_categorical_variables
from preprocess_utils.drop_high_corr import drop_high_corr
from preprocess_utils.outlier_check import winsorize_financial_variables
from preprocess_utils.standard_var import standardize_numeric_variables
from preprocess_utils.rename_x_y import rename_variables_xy

def preprocess_pipeline(df: pd.DataFrame, 
                       missing_threshold: float = 20,
                       correlation_threshold: float = 0.8) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline on the input DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess
        missing_threshold (float): Threshold percentage for missing values
        correlation_threshold (float): Threshold for correlation between variables
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Step 1: Missing Value Treatment
    df_clean0 = missing_value_treatment(df, missing_threshold)[0]
    
    # Step 2: Duplicate Detection
    duplicate_groups = extract_near_duplicate_groups(df_clean0.columns.tolist())
    
    # Step 3: Remove Fuzzy Variables
    df_clean1 = drop_fuzzy_variables(df_clean0)
    
    # Step 4: Categorical Encoding
    df_clean2 = encode_categorical_variables(df_clean1)
    
    # Step 5: Correlation Analysis
    df_clean3 = drop_high_corr(df_clean2, threshold=correlation_threshold)
    
    # Step 6: Outlier Treatment
    df_clean4 = winsorize_financial_variables(df_clean3)
    
    # Step 7: Variable Standardization
    df_clean5 = standardize_numeric_variables(df_clean4)
    
    # Step 8: Variable Classification
    df_clean6 = rename_variables_xy(df_clean5)
    
    return df_clean6