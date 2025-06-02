import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

# imports
from constants import EXCLUDE_VARIABLES

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# exclude patterns
DEFAULT_EXCLUDE_PATTERNS = ['dummy_', 'quarter_factor']

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
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features
    var_threshold : float, default=0.01
        Threshold for coefficient of variation (CV) below which features are dropped
    same_threshold : float, default=0.95
        Threshold for proportion of samples sharing the same value
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - DataFrame with low variance features removed
        - List of dropped column names
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # working copy
    df_filtered = df.copy()
    
    # get x_ prefix columns
    x_columns = [col for col in df_filtered.columns if col.startswith('x_')]
    
    if not x_columns:
        logger.warning("No x_ prefix columns found in DataFrame")
        return df_filtered, []
    
    logger.info(f"\n{'='*80}\nDropping low variance features\n{'='*80}")
    logger.info(f"Total features to check: {len(x_columns)}")
    
    # track dropped columns
    zero_var_cols = []
    low_cv_cols = []
    same_value_cols = []
    
    for col in x_columns:
        # check exclusions
        base_col = col.replace('x_', '', 1)
        if base_col in EXCLUDE_VARIABLES.union({'EPSSurpC', 'IndRel_EPSSurpC', 'IndRel_SUEC', 'SUEC'}) or any(pattern in col for pattern in DEFAULT_EXCLUDE_PATTERNS):
            continue
            
        values = df_filtered[col].dropna()
        
        if values.empty:
            logger.warning(f"Column {col} has all NaN values")
            continue
        
        # calculate statistics
        mean = values.mean()
        std = values.std()
        var = values.var()
        
        # check zero variance
        if var == 0:
            zero_var_cols.append(col)
            continue
        
        # check coefficient of variation
        if abs(mean) < 1e-10:
            if std < var_threshold:
                low_cv_cols.append(col)
            continue
            
        cv = abs(std / mean)
        
        if cv < var_threshold:
            low_cv_cols.append(col)
            continue
        
        # check for high percentage of same value
        value_counts = values.value_counts(normalize=True)
        max_freq = value_counts.iloc[0]
        if max_freq > same_threshold:
            same_value_cols.append(col)
            continue
    
    # combine dropped columns
    columns_to_drop = zero_var_cols + low_cv_cols + same_value_cols
    
    # drop and log summary
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
