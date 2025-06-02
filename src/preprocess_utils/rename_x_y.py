import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Set
import logging

# imports
from constants import EXCLUDE_VARIABLES

# logging setup
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
    
    Parameters
    ----------
    df : Union[pd.DataFrame, Tuple]
        Input DataFrame containing variables to rename
    outcome_variables : Set[str], optional
        Set of outcome variable names
    exclude_patterns : List[str], optional
        List of patterns to exclude from x_ prefixing
        
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed variables
    """
    if isinstance(df, tuple):
        if len(df) == 0 or not isinstance(df[0], pd.DataFrame):
            raise TypeError("If input is a tuple, first element must be a DataFrame")
        df = df[0]
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame or tuple containing DataFrame")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # default outcome variables
    if outcome_variables is None:
        outcome_variables = {
            'EPS_actual', 'EPSDiff', 'EPS_surprise',
            'EPSNormalized_actual', 'EPSNormalized_diff',
            'EPSNormalized_surprise',
            'revenue_actual', 'revenueDiff', 'revenue_surprise',
            'EPS_SUE', 'EPSNorm_SUE', 'revenue_SUE'
        }
    
    # default exclude patterns
    if exclude_patterns is None:
        exclude_patterns = ['dummy_', 'companyid']
    
    # variables to exclude from x_ prefixing
    exclude_from_x_prefix = EXCLUDE_VARIABLES
    
    # working copy
    df_renamed = df.copy()
    
    # get numeric columns
    numeric_columns = df_renamed.select_dtypes(include=[np.number]).columns
    
    # track renames
    rename_dict = {}
    
    # process outcome variables (y_ prefix)
    for col in outcome_variables:
        if col in df_renamed.columns:
            if not any(pattern in col for pattern in exclude_patterns):
                new_name = f'y_{col}' if not col.startswith('y_') else col
                rename_dict[col] = new_name
        else:
            logger.warning(f"Outcome variable {col} not found in DataFrame")
    
    # process independent variables (x_ prefix)
    for col in numeric_columns:
        if col in rename_dict or \
           any(pattern in col for pattern in exclude_patterns) or \
           col.startswith('x_') or \
           col in exclude_from_x_prefix:
            continue
            
        rename_dict[col] = f'x_{col}'
    
    # apply renaming
    df_renamed = df_renamed.rename(columns=rename_dict)
    
    # log summary
    n_y = sum(1 for col in df_renamed.columns if col.startswith('y_'))
    n_x = sum(1 for col in df_renamed.columns if col.startswith('x_'))
    logger.info(f"Renamed {n_y} outcome variables (y_) and {n_x} independent variables (x_)")
    
    return df_renamed
