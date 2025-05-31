import pandas as pd
import numpy as np
from typing import Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants for variables to exclude from standardization
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

def standardize_numeric_variables(
    df: Union[pd.DataFrame, Tuple],
    exclude_patterns: list = None,
    exclude_variables: set = EXCLUDE_VARIABLES
) -> pd.DataFrame:
    """
    Standardize numeric variables in the DataFrame, excluding one-hot encoded columns,
    specified patterns, and specific variables.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, Tuple]
        Input DataFrame containing variables to standardize.
        If a tuple is provided, the first element should be the DataFrame.
    exclude_patterns : list, optional
        List of patterns to exclude from standardization.
        Default excludes 'dummy_' prefixed columns.
    exclude_variables : set, optional
        Set of specific variable names to exclude from standardization.
        Default includes EPS and revenue-related variables.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with numeric variables standardized
        
    Examples
    --------
    >>> df_standardized = standardize_numeric_variables(df)
    >>> # Or with custom exclude patterns:
    >>> df_standardized = standardize_numeric_variables(df, ['dummy_', 'raw_', 'id'])
    
    Notes
    -----
    Standardization uses z-score method: (x - mean) / std
    """
    # Handle input that might be a tuple
    if isinstance(df, tuple):
        if len(df) == 0 or not isinstance(df[0], pd.DataFrame):
            raise TypeError("If input is a tuple, first element must be a DataFrame")
        df = df[0]
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame or tuple containing DataFrame")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Default exclude patterns if none provided
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
    
    # Create working copy
    df_standardized = df.copy()
    
    # Get numeric columns
    numeric_columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    # Filter out columns matching exclude patterns and specific variables
    columns_to_standardize = [
        col for col in numeric_columns 
        if not any(pattern in col for pattern in exclude_patterns)
        and col not in exclude_variables
    ]
    
    logger.info(f"Standardizing {len(columns_to_standardize)} numeric variables")
    logger.info(f"Excluding {len(exclude_variables)} specific variables from standardization")
    
    # Standardize selected columns
    for column in columns_to_standardize:
        mean = df_standardized[column].mean()
        std = df_standardized[column].std()
        
        # Check for zero standard deviation
        if std == 0:
            logger.warning(f"Column {column} has zero standard deviation, skipping standardization")
            continue
            
        df_standardized[column] = (df_standardized[column] - mean) / std
        
    
    logger.info("Completed numeric variable standardization")
    return df_standardized
