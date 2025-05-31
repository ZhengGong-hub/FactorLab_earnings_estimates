import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def drop_low_variance(
    df: pd.DataFrame,
    var_threshold: float = 0.01,
    same_threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop features with low variance based on multiple criteria:
    1. Zero variance (constant features)
    2. Normalized variance below threshold
    3. High percentage of samples sharing the same value
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features
    var_threshold : float, default=0.01
        Threshold for normalized variance below which features are dropped
    same_threshold : float, default=0.95
        Threshold for proportion of samples sharing the same value
        above which features are dropped
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - DataFrame with low variance features removed
        - List of dropped column names
        
    Examples
    --------
    >>> df_filtered, dropped_cols = drop_low_variance(df)
    >>> # Or with custom thresholds:
    >>> df_filtered, dropped_cols = drop_low_variance(df, variance_threshold=0.02)
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
    
    # Lists to store dropped columns for each criterion
    zero_var_cols = []
    low_var_cols = []
    same_value_cols = []
    
    for col in x_columns:
        values = df_filtered[col].dropna()
        
        if values.empty:
            logger.warning(f"Column {col} has all NaN values")
            continue
        
        # 1. Check for zero variance
        if values.var() == 0:
            zero_var_cols.append(col)
            continue
        
        # 2. Check normalized variance
        # Normalize the values to [0,1] range
        normalized_values = (values - values.min()) / (values.max() - values.min())
        normalized_variance = normalized_values.var()
        
        if normalized_variance < var_threshold:
            low_var_cols.append(col)
            continue
        
        # 3. Check for high percentage of same value
        value_counts = values.value_counts(normalize=True)
        if value_counts.iloc[0] > same_threshold:
            same_value_cols.append(col)
            continue
    
    # Combine all columns to drop
    columns_to_drop = zero_var_cols + low_var_cols + same_value_cols
    
    # Drop the columns
    if columns_to_drop:
        df_filtered = df_filtered.drop(columns=columns_to_drop)
        
        # Log summary
        logger.info(f"Dropped {len(zero_var_cols)} columns with zero variance:")
        if zero_var_cols:
            logger.info(f"  {zero_var_cols}")
            
        logger.info(f"Dropped {len(low_var_cols)} columns with normalized variance < {var_threshold}:")
        if low_var_cols:
            logger.info(f"  {low_var_cols}")
            
        logger.info(f"Dropped {len(same_value_cols)} columns with >{same_threshold*100}% same value:")
        if same_value_cols:
            logger.info(f"  {same_value_cols}")
    else:
        logger.info("No columns met the criteria for dropping")
    
    return df_filtered, columns_to_drop
