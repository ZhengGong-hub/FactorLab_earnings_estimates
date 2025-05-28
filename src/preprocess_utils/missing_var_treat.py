import pandas as pd
import logging
from typing import Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        Percentage of missing values for each column
    """
    return df.isnull().mean() * 100

def missing_value_treatment(
    df: pd.DataFrame, 
    threshold: float = 20.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove columns with missing values exceeding the threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean
    threshold : float, default=20.0
        Maximum percentage of missing values allowed in a column.
        Columns exceeding this will be dropped.
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Clean DataFrame with high-missing-value columns removed
        - List of columns that were dropped
        
    Examples
    --------
    >>> df_clean, dropped_cols = missing_value_treatment(df, threshold=30.0)
    >>> print(f"Dropped {len(dropped_cols)} columns")
    
    Raises
    ------
    ValueError
        If DataFrame is empty or threshold is invalid
    """
    # Input validation
    if df.empty or len(df.columns) == 0:
        raise ValueError("Input DataFrame is empty or has no columns")
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")
    
    # Create working copy
    df_clean = df.copy()
    
    # Calculate missing value percentages
    missing_pct = calculate_missing_percentages(df_clean)
    
    # Identify columns to drop
    columns_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    # Log information about drops
    if columns_to_drop:
        logger.info(f"Dropping {len(columns_to_drop)} columns with >{threshold}% missing values")
        for col in columns_to_drop:
            logger.info(f"Column '{col}': {missing_pct[col]:.1f}% missing")
        
        # Drop the columns
        df_clean = df_clean.drop(columns=columns_to_drop)
    else:
        logger.info("No columns exceeded the missing value threshold")
    
    return df_clean, columns_to_drop