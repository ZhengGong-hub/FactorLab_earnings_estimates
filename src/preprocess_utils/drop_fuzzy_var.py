import pandas as pd
import logging
from typing import List, Tuple, Set

# Configure logging
from logger import setup_logger

# setup logger
logger = setup_logger(__name__)

# Define constants for columns to drop
FUZZY_VARIABLES = {
    'BVEV_2', 'AssetTurn', 'ChgATO', 'EbitToAst', 
    'OCFAst', 'REToAst', 'ROA'
}

METADATA_COLUMNS = {
    'transcriptid', 'keydevid', 'fiscalyear', 'fiscalquarter',
    'tradingitemid', 'earningsdate', 'marketindicatortypename',
    'EPSNormalized_actual_et', 'revenue_actual_et', 'EPS_actual_et', 'Unnamed: 0'
}.union({f'affactor_asofdate_{i}' for i in range(13)})  # Add all affactor_asofdate_0 through affactor_asofdate_12

def drop_fuzzy_variables(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop fuzzy variables and metadata columns, then create quarter factors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with financial variables
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Clean DataFrame with dropped columns and quarter factor added
        - List of columns that were dropped
        
    Examples
    --------
    >>> df_clean, dropped_cols = drop_fuzzy_variables(df)
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    
    df_clean = df.copy()
    cols_to_drop = list(FUZZY_VARIABLES.union(METADATA_COLUMNS).intersection(df.columns))
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} columns")
        logger.info(f"Dropped columns: {cols_to_drop}")
    else:
        logger.info("No fuzzy variables or metadata columns to drop")
    
    return df_clean, cols_to_drop
