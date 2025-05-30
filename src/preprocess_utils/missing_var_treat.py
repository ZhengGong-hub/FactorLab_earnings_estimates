import pandas as pd
from typing import Tuple, List, Set

# internal imports
from logger import setup_logger

# setup logger
logger = setup_logger(__name__)

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

def calculate_nan_percentages(df: pd.DataFrame) -> pd.Series:
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
    
    # check all keep_variables are in the dataframe
    invalid_vars = [var for var in keep_variables if var not in df.columns]
    if invalid_vars:
        raise ValueError(f"Some keep_variables not found in DataFrame: {invalid_vars}")
    
    # Create working copy
    df_clean = df.copy()
    
    # Calculate missing value percentages
    missing_pct = calculate_nan_percentages(df_clean)
    
    # Log overall missing value statistics
    logger.info(f"Average missing: {missing_pct.mean():.1f}%")
    logger.info(f"Median missing: {missing_pct.median():.1f}%")
    logger.info(f"Columns with >{threshold}% missing: {(missing_pct > threshold).sum()}")
    
    # Process columns based on missing value threshold
    high_missing_cols = missing_pct[missing_pct > threshold]
    
    # Separate columns into drops and keeps
    columns_to_drop = [col for col in high_missing_cols.index if col not in keep_variables]
    kept_high_missing = [col for col in keep_variables if col in high_missing_cols.index]
    
    # Log and process drops
    if columns_to_drop:
        logger.info(f"Dropping {len(columns_to_drop)} columns with >{threshold}% missing values")
        for col in columns_to_drop:
            logger.info(f"Column '{col}': {high_missing_cols[col]:.1f}% missing")
        df_clean = df_clean.drop(columns=columns_to_drop)
    else:
        logger.info(f"No columns exceeded the {threshold}% missing value threshold!")
    
    # Log kept variables
    if kept_high_missing:
        logger.info(f"Kept {len(kept_high_missing)} variables despite high missing values:")
        for col in kept_high_missing:
            logger.info(f"Kept column '{col}': {high_missing_cols[col]:.1f}% missing")
    
    # Final statistics
    logger.info(f"Final DataFrame shape: {df_clean.shape}")
    logger.info(f"Columns removed: {len(columns_to_drop)}")
    logger.info(f"Columns retained: {len(df_clean.columns)}")
    return df_clean, columns_to_drop