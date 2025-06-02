import pandas as pd
import numpy as np
from typing import Union, Tuple
import logging

# imports
from constants import EXCLUDE_VARIABLES

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# exclude patterns
DEFAULT_EXCLUDE_PATTERNS = ['dummy_', 'quarter_factor']

def standardize_numeric_variables(
    df: Union[pd.DataFrame, Tuple],
    exclude_patterns: list = None,
    exclude_variables: set = EXCLUDE_VARIABLES.union({'EPS_SUE', 'EPSNorm_SUE', 'revenue_SUE', 'EPSSurpC', 'IndRel_EPSSurpC', 'IndRel_SUEC', 'SUEC'})
) -> pd.DataFrame:
    """
    Standardize numeric variables in the DataFrame, excluding specified patterns and variables.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, Tuple]
        Input DataFrame containing variables to standardize
    exclude_patterns : list, optional
        List of patterns to exclude from standardization
    exclude_variables : set, optional
        Set of specific variable names to exclude from standardization
        
    Returns
    -------
    pd.DataFrame
        DataFrame with numeric variables standardized
    """
    if isinstance(df, tuple):
        if len(df) == 0 or not isinstance(df[0], pd.DataFrame):
            raise TypeError("If input is a tuple, first element must be a DataFrame")
        df = df[0]
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame or tuple containing DataFrame")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS
    
    # working copy
    df_standardized = df.copy()
    
    # get numeric columns
    numeric_columns = df_standardized.select_dtypes(include=[np.number]).columns
    
    # filter columns to standardize
    columns_to_standardize = []
    for col in numeric_columns:
        if any(pattern in col for pattern in exclude_patterns):
            continue
            
        base_col = col
        if col.startswith(('x_', 'y_')):
            base_col = col[2:]
            
        if base_col in exclude_variables:
            continue
            
        columns_to_standardize.append(col)
    
    logger.info(f"Standardizing {len(columns_to_standardize)} numeric variables")
    logger.info(f"Excluding {len(exclude_variables)} specific variables from standardization")
    
    # standardize columns
    for column in columns_to_standardize:
        mean = df_standardized[column].mean()
        std = df_standardized[column].std()
        
        if std == 0:
            logger.warning(f"Column {column} has zero standard deviation, skipping standardization")
            continue
            
        df_standardized[column] = (df_standardized[column] - mean) / std
    
    logger.info("Completed numeric variable standardization")
    return df_standardized
