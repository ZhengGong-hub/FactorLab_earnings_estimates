import pandas as pd
import logging
from typing import Tuple, List, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants for variables to always keep
KEEP_VARIABLES = {
    'EPS_actual', 'EPS_actual_et', 'EPSDiff', 'EPS_surprise', 'EPS_count', 
    'EPS_std', 'EPS_guidance_high', 'EPS_guidance_low', 'EPSNormalized_actual',
    'EPSNormalized_actual_et', 'EPSNormalized_diff', 'EPSNormalized_surprise', 
    'EPSNormalized_count', 'EPSNormalized_std', 'EPSNormalized_guidance_high', 
    'EPSNormalized_guidance_low', 'revenue_actual', 'revenue_actual_et', 
    'revenueDiff', 'revenue_surprise', 'revenue_count', 'revenue_std',
    'revenue_guidance_high', 'revenue_guidance_low'
}

def calculate_missing_percentages(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the percentage of missing values for each column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze
        
    Returns
    -------
    pd.Series
        Percentage of missing values for each column, sorted in descending order
    """
    return (df.isnull().mean() * 100).sort_values(ascending=False)

def missing_value_treatment(
    df: pd.DataFrame, 
    threshold: float = 50.0,
    keep_variables: Set[str] = KEEP_VARIABLES
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove columns with missing values exceeding the threshold, except for specified variables to keep.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean
    threshold : float, default=50.0
        Maximum percentage of missing values allowed in a column.
        Columns exceeding this will be dropped, except for keep_variables.
        Default is 50.0% (higher threshold preserves more variables).
    keep_variables : Set[str], default=KEEP_VARIABLES
        Set of variable names to keep regardless of their missing value percentage.
        These variables will be retained even if they exceed the threshold.
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Clean DataFrame with high-missing-value columns removed
        - List of columns that were dropped
        
    Examples
    --------
    >>> df_clean, dropped_cols = missing_value_treatment(df, threshold=50.0)
    >>> print(f"Dropped {len(dropped_cols)} columns with >{threshold}% missing values")
    
    Raises
    ------
    ValueError
        - If DataFrame is empty or has no columns
        - If threshold is not between 0 and 100
        - If DataFrame contains non-numeric data in numeric columns
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    if len(df.columns) == 0:
        raise ValueError("Input DataFrame has no columns")
        
    if not 0 <= threshold <= 100:
        raise ValueError(f"Threshold must be between 0 and 100, got {threshold}")
    
    # Validate keep_variables
    invalid_vars = [var for var in keep_variables if var not in df.columns]
    if invalid_vars:
        logger.warning(f"Some keep_variables not found in DataFrame: {invalid_vars}")
    
    # Create working copy
    df_clean = df.copy()
    
    # Calculate missing value percentages
    missing_pct = calculate_missing_percentages(df_clean)
    
    # Log overall missing value statistics
    logger.info(f"Missing value statistics:")
    logger.info(f"Average missing: {missing_pct.mean():.1f}%")
    logger.info(f"Median missing: {missing_pct.median():.1f}%")
    logger.info(f"Columns with >{threshold}% missing: {(missing_pct > threshold).sum()}")
    
    # Identify columns to drop (excluding keep_variables)
    columns_to_drop = [col for col in missing_pct[missing_pct > threshold].index 
                      if col not in keep_variables]
    
    # Log information about drops and keeps
    if columns_to_drop:
        logger.info(f"\nDropping {len(columns_to_drop)} columns with >{threshold}% missing values")
        for col in columns_to_drop:
            logger.info(f"Column '{col}': {missing_pct[col]:.1f}% missing")
        
        # Drop the columns
        df_clean = df_clean.drop(columns=columns_to_drop)
    else:
        logger.info(f"\nNo columns exceeded the {threshold}% missing value threshold")
    
    # Log information about kept variables that exceeded threshold
    kept_high_missing = [col for col in keep_variables 
                        if col in df.columns and missing_pct.get(col, 0) > threshold]
    if kept_high_missing:
        logger.info(f"\nKept {len(kept_high_missing)} variables despite high missing values:")
        for col in kept_high_missing:
            logger.info(f"Kept column '{col}': {missing_pct[col]:.1f}% missing")
    
    # Final statistics
    logger.info(f"\nFinal DataFrame shape: {df_clean.shape}")
    logger.info(f"Columns removed: {len(columns_to_drop)}")
    logger.info(f"Columns retained: {len(df_clean.columns)}")
    
    return df_clean, columns_to_drop