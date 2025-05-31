import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rename_variables_xy(
    df: Union[pd.DataFrame, Tuple],
    outcome_variables: Set[str] = None,
    exclude_patterns: List[str] = None
) -> pd.DataFrame:
    """
    Rename variables by adding prefixes:
    - 'y_' for outcome variables
    - 'x_' for numeric independent variables
    Also calculates SUE (Standardized Unexpected Earnings) variables.
    Excludes one-hot encoded columns and other specified patterns.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, Tuple]
        Input DataFrame containing variables to rename.
        If a tuple is provided, the first element should be the DataFrame.
    outcome_variables : Set[str], optional
        Set of outcome variable names. If None, uses default set.
    exclude_patterns : List[str], optional
        List of patterns to exclude from x_ prefixing.
        Default excludes 'dummy_' prefixed columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed variables and additional SUE variables
        
    Examples
    --------
    >>> df_renamed = rename_variables_xy(df)
    >>> # Or with custom outcome variables:
    >>> df_renamed = rename_variables_xy(df, {'custom_outcome1', 'custom_outcome2'})
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
    
    # Default outcome variables if none provided
    if outcome_variables is None:
        outcome_variables = {
       'EPS_actual', 'EPSDiff', 'EPS_surprise',
       'EPSNormalized_actual', 'EPSNormalized_diff',
       'EPSNormalized_surprise',
       'revenue_actual', 'revenueDiff', 'revenue_surprise'
        }
    
    # Default exclude patterns if none provided
    if exclude_patterns is None:
        exclude_patterns = ['dummy_', 'companyid']
    
    # Variables to exclude from x_ prefixing
    exclude_from_x_prefix = {
        'EPS_count', 'EPS_std', 'EPS_guidance_high', 'EPS_guidance_low',
        'EPSNormalized_count', 'EPSNormalized_std', 'EPSNormalized_guidance_high', 'EPSNormalized_guidance_low',
        'revenue_count', 'revenue_std', 'revenue_guidance_high', 'revenue_guidance_low'
    }
    
    # Create working copy
    df_renamed = df.copy()
    
    # Calculate SUE variables
    # EPS SUE
    df_renamed['y_EPS_SUE'] = np.where(
        df_renamed['EPS_count'] >= 3,
        df_renamed['EPS_surprise'] / df_renamed['EPS_std'],
        np.nan
    )
    
    # EPSNormalized SUE
    df_renamed['y_EPSNorm_SUE'] = np.where(
        df_renamed['EPSNormalized_count'] >= 3,
        df_renamed['EPSNormalized_surprise'] / df_renamed['EPSNormalized_std'],
        np.nan
    )
    
    # Revenue SUE
    df_renamed['y_revenue_SUE'] = np.where(
        df_renamed['revenue_count'] >= 3,
        df_renamed['revenue_surprise'] / df_renamed['revenue_std'],
        np.nan
    )
    
    # Add new SUE variables to outcome variables
    outcome_variables.update({'y_EPS_SUE', 'y_EPSNorm_SUE', 'y_revenue_SUE'})
    
    # Get numeric columns
    numeric_columns = df_renamed.select_dtypes(include=[np.number]).columns
    
    # Dictionary to store column renames
    rename_dict = {}
    
    # Process outcome variables (y_ prefix)
    for col in outcome_variables:
        if col in df_renamed.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                new_name = f'y_{col}' if not col.startswith('y_') else col
                rename_dict[col] = new_name
        else:
            logger.warning(f"Outcome variable {col} not found in DataFrame")
    
    # Process independent variables (x_ prefix)
    for col in numeric_columns:
        # Skip if already processed as outcome variable
        if col in rename_dict:
            continue
            
        # Skip if matches exclude patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue
            
        # Skip if already has x_ prefix
        if col.startswith('x_'):
            continue
            
        # Skip if in exclude_from_x_prefix set
        if col in exclude_from_x_prefix:
            continue
            
        rename_dict[col] = f'x_{col}'
    
    # Apply renaming
    df_renamed = df_renamed.rename(columns=rename_dict)
    
    # Log summary
    n_y = sum(1 for col in df_renamed.columns if col.startswith('y_'))
    n_x = sum(1 for col in df_renamed.columns if col.startswith('x_'))
    logger.info(f"Renamed {n_y} outcome variables (y_) and {n_x} independent variables (x_)")
    logger.info("Added SUE variables: y_EPS_SUE, y_EPSNorm_SUE, y_revenue_SUE")
    
    return df_renamed
