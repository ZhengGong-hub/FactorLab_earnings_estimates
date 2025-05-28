import pandas as pd
import numpy as np
from typing import Set, List
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define financial variables to winsorize
FINANCIAL_VARS = {
    'EPS_normalizedDiff',
    'EPS',
    'EPS_normalized',
    'EPSDiff',
    'revenue',
    'revenueDiff'
}

def winsorize_financial_variables(
    df: pd.DataFrame,
    variables: Set[str] = None,
    lower_percentile: float = 0.05,
    upper_percentile: float = 0.95
) -> pd.DataFrame:
    """
    Winsorize financial variables to handle outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing financial variables
    variables : Set[str], optional
        Set of variables to winsorize. If None, uses default financial variables
    lower_percentile : float, default=0.05
        Lower bound percentile (5%)
    upper_percentile : float, default=0.95
        Upper bound percentile (95%)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with winsorized variables
        
    Examples
    --------
    >>> df_clean = winsorize_financial_variables(df)
    >>> # Or with custom variables:
    >>> df_clean = winsorize_financial_variables(df, {'revenue', 'EPS'})
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Use default variables if none provided
    if variables is None:
        variables = FINANCIAL_VARS
    
    # Create working copy
    df_clean = df.copy()
    
    # Check which variables exist in the DataFrame
    existing_vars = variables.intersection(df.columns)
    missing_vars = variables - existing_vars
    
    if missing_vars:
        logger.warning(f"Variables not found in DataFrame: {missing_vars}")
    
    if not existing_vars:
        logger.warning("No variables to winsorize")
        return df_clean
    
    # Winsorize each variable
    for var in existing_vars:
        try:
            # Get original statistics
            orig_mean = df_clean[var].mean()
            orig_std = df_clean[var].std()
            
            # Calculate percentiles
            lower_bound = np.percentile(df_clean[var].dropna(), lower_percentile * 100)
            upper_bound = np.percentile(df_clean[var].dropna(), upper_percentile * 100)
            
            # Winsorize the variable
            df_clean[var] = df_clean[var].clip(lower=lower_bound, upper=upper_bound)
            
            # Get new statistics
            new_mean = df_clean[var].mean()
            new_std = df_clean[var].std()
            
            # Log the changes
            logger.info(f"Winsorized {var}:")
            logger.info(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            logger.info(f"  Mean: {orig_mean:.2f} → {new_mean:.2f}")
            logger.info(f"  Std: {orig_std:.2f} → {new_std:.2f}")
            
        except Exception as e:
            logger.error(f"Error winsorizing {var}: {str(e)}")
            raise
    
    logger.info(f"Successfully winsorized {len(existing_vars)} variables")
    return df_clean

def plot_winsorization_effect(
    df_original: pd.DataFrame,
    df_winsorized: pd.DataFrame,
    variable: str
) -> None:
    """
    Plot histograms comparing original and winsorized data.
    
    Parameters
    ----------
    df_original : pd.DataFrame
        Original DataFrame
    df_winsorized : pd.DataFrame
        Winsorized DataFrame
    variable : str
        Variable name to plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 6))
        
        # Plot original data
        plt.subplot(1, 2, 1)
        sns.histplot(df_original[variable].dropna(), bins=50)
        plt.title(f'Original {variable}')
        plt.xlabel('Value')
        
        # Plot winsorized data
        plt.subplot(1, 2, 2)
        sns.histplot(df_winsorized[variable].dropna(), bins=50)
        plt.title(f'Winsorized {variable}')
        plt.xlabel('Value')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Plotting requires matplotlib and seaborn")
