import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_SUE_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Standardized Unexpected Earnings (SUE) variables for EPS, EPSNormalized, and revenue.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the necessary columns for SUE calculation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added SUE variables
        
    Notes
    -----
    Calculates three SUE variables:
    - EPS_SUE: SUE based on EPS surprise
    - EPSNorm_SUE: SUE based on normalized EPS surprise
    - revenue_SUE: SUE based on revenue surprise
    
    SUE is calculated as surprise/std when count >= 3, otherwise NaN
    """
    # Create working copy
    df_with_sue = df.copy()
    
    # Calculate SUE variables
    # EPS SUE
    df_with_sue['EPS_SUE'] = np.where(
        df_with_sue['EPS_count'] >= 3,
        df_with_sue['EPS_surprise'] / df_with_sue['EPS_std'],
        np.nan
    )
    
    # EPSNormalized SUE
    df_with_sue['EPSNorm_SUE'] = np.where(
        df_with_sue['EPSNormalized_count'] >= 3,
        df_with_sue['EPSNormalized_surprise'] / df_with_sue['EPSNormalized_std'],
        np.nan
    )
    
    # Revenue SUE
    df_with_sue['revenue_SUE'] = np.where(
        df_with_sue['revenue_count'] >= 3,
        df_with_sue['revenue_surprise'] / df_with_sue['revenue_std'],
        np.nan
    )
    
    logger.info("Added SUE variables: EPS_SUE, EPSNorm_SUE, revenue_SUE")
    
    return df_with_sue
