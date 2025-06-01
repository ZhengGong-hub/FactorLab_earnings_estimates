import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

# Configure logging with a more visible format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define variables to exclude from variance check (same as standardization)
EXCLUDE_VARIABLES = {
    'EPS_actual', 'EPSDiff', 'EPS_surprise', 'EPS_count', 
    'EPS_std', 'EPS_guidance_high', 'EPS_guidance_low', 'EPSNormalized_actual',
    'EPSNormalized_diff', 'EPSNormalized_surprise', 
    'EPSNormalized_count', 'EPSNormalized_std', 'EPSNormalized_guidance_high', 
    'EPSNormalized_guidance_low', 'revenue_actual', 
    'revenueDiff', 'revenue_surprise', 'revenue_count', 'revenue_std',
    'revenue_guidance_high', 'revenue_guidance_low'
}

# Default patterns to exclude
DEFAULT_EXCLUDE_PATTERNS = ['dummy_', 'quarter_factor', 'companyid']

def drop_low_variance(
    df: pd.DataFrame,
    var_threshold: float = 0.01,
    same_threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop features with low variance based on multiple criteria:
    1. Zero variance (constant features)
    2. Low coefficient of variation (CV = std/mean) below threshold
    3. High percentage of samples sharing the same value
    
    Excludes certain financial variables and specific patterns from the variance check.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features
    var_threshold : float, default=0.01
        Threshold for coefficient of variation (CV) below which features are dropped.
        CV is calculated as std/mean for each feature.
    same_threshold : float, default=0.95
        Threshold for proportion of samples sharing the same value
        above which features are dropped
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - DataFrame with low variance features removed
        - List of dropped column names
        
    Notes
    -----
    - CV is used instead of normalized variance as it's a better measure for 
      comparing variability across features with different scales
    - For features with mean close to zero, CV might be unstable or undefined
    - Features with undefined CV (mean = 0) are dropped if they have zero variance
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Create working copy
    df_filtered = df.copy()
    
    # Get x_ prefix columns
    x_columns = [col for col in df_filtered.columns if col.startswith('x_')]
    
    if not x_columns:
        logger.warning("No x_ prefix columns found in DataFrame")
        return df_filtered, []
    
    logger.info(f"\n{'='*80}\nDropping low variance features\n{'='*80}")
    logger.info(f"Total features to check: {len(x_columns)}")
    
    # Lists to store dropped columns for each criterion
    zero_var_cols = []
    low_cv_cols = []
    same_value_cols = []
    
    for col in x_columns:
        # Skip excluded variables and patterns
        base_col = col.replace('x_', '', 1)  # Remove x_ prefix for checking
        if base_col in EXCLUDE_VARIABLES or any(pattern in col for pattern in DEFAULT_EXCLUDE_PATTERNS):
            continue
            
        values = df_filtered[col].dropna()
        
        if values.empty:
            logger.warning(f"Column {col} has all NaN values")
            continue
        
        # Calculate basic statistics
        mean = values.mean()
        std = values.std()
        var = values.var()
        
        # 1. Check for zero variance
        if var == 0:
            zero_var_cols.append(col)
            continue
        
        # 2. Check coefficient of variation (CV)
        # Handle cases where mean is zero or close to zero
        if abs(mean) < 1e-10:  # Using small threshold to check for "practical" zero
            if std < var_threshold:  # If std is also small, consider it low variance
                low_cv_cols.append(col)
            continue
            
        cv = abs(std / mean)  # Use absolute value as CV can be negative
        
        if cv < var_threshold:
            low_cv_cols.append(col)
            continue
        
        # 3. Check for high percentage of same value
        value_counts = values.value_counts(normalize=True)
        max_freq = value_counts.iloc[0]
        if max_freq > same_threshold:
            same_value_cols.append(col)
            continue
    
    # Combine all columns to drop
    columns_to_drop = zero_var_cols + low_cv_cols + same_value_cols
    
    # Drop the columns and print summary
    if columns_to_drop:
        df_filtered = df_filtered.drop(columns=columns_to_drop)
        
        logger.info("\nDropped variables:")
        if zero_var_cols:
            logger.info("Zero variance:")
            logger.info(zero_var_cols)
        
        if low_cv_cols:
            logger.info(f"\nLow CV (< {var_threshold}):")
            logger.info(low_cv_cols)
        
        if same_value_cols:
            logger.info(f"\nHigh same-value proportion (> {same_threshold*100}%):")
            logger.info(same_value_cols)
        
        logger.info(f"\nTotal dropped: {len(columns_to_drop)} out of {len(x_columns)}")
    else:
        logger.info("\nNo features met the criteria for dropping")
    
    return df_filtered, columns_to_drop
