import pandas as pd
import logging
from typing import List, Tuple, Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants for columns to drop
FUZZY_VARIABLES = {
    'BVEV_2', 'AssetTurn', 'ChgATO', 'EbitToAst', 
    'OCFAst', 'REToAst', 'ROA'
}

METADATA_COLUMNS = {
    'transcriptid', 'keydevid', 'fiscalyear', 'fiscalquarter',
    'tradingitemid', 'ec_et', 'earningsdate', 'marketindicatortypename',
    'EPSnormalized_et', 'revenue_et', 'EPS_et', 'affactor_asofdate_0'
}

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
    
    # Create working copy
    df_clean = df.copy()
    dropped_cols = []
    
    # Drop fuzzy variables and metadata columns
    all_cols_to_drop = FUZZY_VARIABLES.union(METADATA_COLUMNS)
    existing_cols = all_cols_to_drop.intersection(df_clean.columns)
    
    if existing_cols:
        df_clean = df_clean.drop(columns=list(existing_cols))
        dropped_cols = list(existing_cols)
        logger.info(f"Dropped {len(dropped_cols)} columns")
    
    # Create quarter factor if EPS_et exists
    if 'EPS_et' in df.columns:
        try:
            # Convert date to quarter number (1-4)
            df_clean['quarter_factor'] = pd.to_datetime(df['EPS_et']).dt.quarter
            logger.info("Created quarter_factor column (1-4)")
        except Exception as e:
            logger.error(f"Failed to create quarter factor: {str(e)}")
            raise ValueError(f"Error creating quarter factor: {str(e)}")
    
    return df_clean, dropped_cols
