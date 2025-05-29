import pandas as pd
import numpy as np
from typing import Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardize_numeric_variables(
    df: Union[pd.DataFrame, Tuple],
    exclude_patterns: list = None
) -> pd.DataFrame:
    """
    Standardize numeric variables in the DataFrame, excluding one-hot encoded columns
    and other specified patterns.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, Tuple]
        Input DataFrame containing variables to standardize.
        If a tuple is provided, the first element should be the DataFrame.
    exclude_patterns : list, optional
        List of patterns to exclude from standardization.
        Default excludes 'dummy_' prefixed columns.
        
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
        exclude_patterns = ['dummy_', 'quarter_factor', 'companyid', 'securityid']
    
    # Create working copy
    df_standardized = df.copy()
    
    # Get numeric columns
    numeric_columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    # Filter out columns matching exclude patterns
    columns_to_standardize = [
        col for col in numeric_columns 
        if not any(pattern in col for pattern in exclude_patterns)
    ]
    
    logger.info(f"Standardizing {len(columns_to_standardize)} numeric variables")
    
    # Standardize selected columns
    for column in columns_to_standardize:
        try:
            mean = df_standardized[column].mean()
            std = df_standardized[column].std()
            
            # Check for zero standard deviation
            if std == 0:
                logger.warning(f"Column {column} has zero standard deviation, skipping standardization")
                continue
                
            df_standardized[column] = (df_standardized[column] - mean) / std
            logger.debug(f"Standardized {column}: mean={mean:.4f}, std={std:.4f}")
            
        except Exception as e:
            logger.error(f"Error standardizing column {column}: {str(e)}")
            raise
    
    logger.info("Completed numeric variable standardization")
    return df_standardized
