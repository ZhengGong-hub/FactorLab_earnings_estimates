import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function for backward compatibility.
    Checks correlations and returns the correlation matrix.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric variables
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    _, corr_matrix, _ = check_correlations(df)
    return corr_matrix

def check_correlations(
    df: pd.DataFrame,
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.8
) -> Tuple[Dict[str, float], pd.DataFrame, List[List[str]]]:
    """
    Check correlations among numeric variables and identify multicollinearity using VIF.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric variables
    vif_threshold : float, default=10.0
        Threshold for VIF values to identify multicollinearity
    corr_threshold : float, default=0.8
        Threshold for correlation coefficient to identify highly correlated pairs
        
    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame, List[List[str]]]
        - Dictionary of VIF values for each variable
        - Correlation matrix
        - List of correlated variable clusters
        
    Examples
    --------
    >>> # Full analysis
    >>> vif_values, corr_matrix, corr_clusters = check_correlations(df)
    >>> print("High VIF variables:", [v for v, vif in vif_values.items() if vif > 10])
    >>>
    >>> # Simple correlation matrix only
    >>> corr_matrix = check_correlation(df)
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty or len(df.columns) == 0:
        raise ValueError("Input DataFrame is empty or has no columns")
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame")
    
    # Remove columns with zero variance
    zero_var_cols = numeric_df.columns[numeric_df.std() == 0]
    if len(zero_var_cols) > 0:
        logger.warning(f"Removing {len(zero_var_cols)} columns with zero variance")
        numeric_df = numeric_df.drop(columns=zero_var_cols)
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find clusters of highly correlated variables
    corr_clusters = []
    processed_vars = set()
    
    for var1 in corr_matrix.columns:
        if var1 in processed_vars:
            continue
            
        # Find all variables highly correlated with var1
        cluster = [var1]
        for var2 in corr_matrix.columns:
            if var2 != var1 and abs(corr_matrix.loc[var1, var2]) > corr_threshold:
                cluster.append(var2)
                
        if len(cluster) > 1:  # Only add clusters with 2+ variables
            corr_clusters.append(sorted(cluster))
            processed_vars.update(cluster)
    
    # Calculate VIF for each variable
    vif_values = {}
    for i, column in enumerate(numeric_df.columns):
        try:
            # Create X matrix for VIF calculation (all other variables)
            X = numeric_df.drop(columns=[column])
            y = numeric_df[column]
            
            # Calculate VIF
            vif = variance_inflation_factor(
                exog=np.column_stack([np.ones(len(X)), X]),
                exog_idx=1
            )
            vif_values[column] = vif
            
            if vif > vif_threshold:
                logger.warning(f"High VIF ({vif:.1f}) detected for: {column}")
        except Exception as e:
            logger.warning(f"Could not calculate VIF for {column}: {str(e)}")
    
    # Log findings
    logger.info(f"Found {len(corr_clusters)} clusters of highly correlated variables")
    for i, cluster in enumerate(corr_clusters, 1):
        logger.info(f"Cluster {i}: {', '.join(cluster)}")
    
    high_vif_vars = {k: v for k, v in vif_values.items() if v > vif_threshold}
    if high_vif_vars:
        logger.info("Variables with high VIF:")
        for var, vif in high_vif_vars.items():
            logger.info(f"  {var}: {vif:.1f}")
    
    return vif_values, corr_matrix, corr_clusters

def plot_correlation_heatmap(
    corr_matrix: Union[pd.DataFrame, Tuple],
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Plot a correlation matrix heatmap.
    
    Parameters
    ----------
    corr_matrix : Union[pd.DataFrame, Tuple]
        Correlation matrix to plot. Can be either a DataFrame or a tuple from check_correlations
    figsize : Tuple[int, int], default=(12, 10)
        Figure size in inches
    """
    # Handle input that might be a tuple from check_correlations
    if isinstance(corr_matrix, tuple):
        if len(corr_matrix) >= 2 and isinstance(corr_matrix[1], pd.DataFrame):
            corr_matrix = corr_matrix[1]
        else:
            raise ValueError("If providing a tuple, it must be from check_correlations")
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        square=True
    )
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
