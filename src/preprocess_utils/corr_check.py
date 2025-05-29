import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Dict
import logging
import os
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_correlation(df: pd.DataFrame, threshold: float = 0.8, target_var: str = 'EPS_normalizedDiff', output_dir: str = "output_data") -> pd.DataFrame:
    """
    Check correlations and remove highly correlated variables based on their correlation with target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with variables to check
    threshold : float, default=0.8
        Correlation threshold to report
    target_var : str, default='EPS_normalizedDiff'
        Target variable to compare correlations against
    output_dir : str, default="output_data"
        Directory to save the output files
        
    Returns
    -------
    pd.DataFrame
        DataFrame with highly correlated variables removed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in DataFrame")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame")
        
    logger.info(f"Computing correlations for {len(numeric_df.columns)} numeric variables")
    
    # Calculate correlation matrix for numeric columns
    corr_matrix = numeric_df.corr()
    
    # Calculate correlations with target variable
    target_corrs = corr_matrix[target_var].abs()
    
    # Find high correlations
    high_corr = []
    high_corr_vars = set()  # Set to store names of highly correlated variables
    var_count = Counter()  # Counter for variable occurrences
    dropped_var_list = set()  # Set to store variables to be dropped
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):  # upper triangle only
            correlation = corr_matrix.iloc[i, j]
            if abs(correlation) > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                
                # Count variable occurrences
                var_count[var1] += 1
                var_count[var2] += 1
                
                # Get non-missing value counts
                non_missing_var1 = numeric_df[var1].count()
                non_missing_var2 = numeric_df[var2].count()
                non_missing_both = numeric_df[[var1, var2]].dropna().shape[0]
                
                # Determine which variable to drop based on correlation with target
                var1_target_corr = target_corrs[var1]
                var2_target_corr = target_corrs[var2]
                
                # Add the variable with weaker correlation to dropped list
                if var1_target_corr >= var2_target_corr:
                    dropped_var_list.add(var2)
                else:
                    dropped_var_list.add(var1)
                
                high_corr.append({
                    'variable1': var1,
                    'variable2': var2,
                    'correlation': correlation,
                    'non_missing_var1': non_missing_var1,
                    'non_missing_var2': non_missing_var2,
                    'non_missing_both': non_missing_both,
                    'var1_target_corr': var1_target_corr,
                    'var2_target_corr': var2_target_corr,
                    'dropped_variable': var2 if var1_target_corr >= var2_target_corr else var1
                })
                high_corr_vars.add(var1)
                high_corr_vars.add(var2)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save correlation matrix to file
    output_path = os.path.join(output_dir, "correlation_check.csv")
    corr_matrix.to_csv(output_path)
    logger.info(f"Correlation matrix saved to: {output_path}")
    
    # Save high correlation information
    if high_corr:
        logger.info(f"\nHigh correlations (>{threshold}):")
        
        # Sort high correlations by absolute correlation value
        high_corr_sorted = sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True)
        
        # Log and save detailed correlations
        for corr_info in high_corr_sorted:
            logger.info(
                f"{corr_info['variable1']:<20} -- {corr_info['variable2']:<20}: "
                f"{corr_info['correlation']:>6.3f} "
                f"(Target corr: {corr_info['var1_target_corr']:.3f} vs {corr_info['var2_target_corr']:.3f}, "
                f"Dropped: {corr_info['dropped_variable']})"
            )
        
        # Save detailed correlations to CSV
        high_corr_path = os.path.join(output_dir, "high_correlations.csv")
        pd.DataFrame(high_corr_sorted).to_csv(high_corr_path, index=False)
        logger.info(f"High correlation details saved to: {high_corr_path}")
        
        # Save list of variables to drop
        drop_path = os.path.join(output_dir, "dropped_var_list.txt")
        with open(drop_path, 'w') as f:
            for var in sorted(dropped_var_list):
                f.write(f"{var}\n")
        logger.info(f"List of variables to drop saved to: {drop_path}")
        
        # Save variable count summary
        count_data = []
        for var, count in var_count.most_common():
            count_data.append({
                'variable': var,
                'high_correlation_count': count,
                'non_missing_values': numeric_df[var].count(),
                'missing_values': numeric_df[var].isna().sum(),
                'total_rows': len(numeric_df),
                'target_correlation': target_corrs[var],
                'is_dropped': var in dropped_var_list
            })
        
        count_path = os.path.join(output_dir, "high_corr_count.csv")
        pd.DataFrame(count_data).to_csv(count_path, index=False)
        logger.info(f"Variable count summary saved to: {count_path}")
        
    else:
        logger.info(f"No correlations above {threshold} found")
    
    # Drop the identified variables and return the filtered dataset
    logger.info(f"Dropping {len(dropped_var_list)} variables due to high correlation")
    return df.drop(columns=list(dropped_var_list))
