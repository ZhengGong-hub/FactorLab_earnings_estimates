import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Dict
import logging
import os
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants for variables to always keep
KEEP_VARIABLES = {
    'EPS_actual', 'EPSDiff', 'EPS_surprise', 'EPS_count', 
    'EPS_std', 'EPS_guidance_high', 'EPS_guidance_low', 'EPSNormalized_actual',
    'EPSNormalized_diff', 'EPSNormalized_surprise', 
    'EPSNormalized_count', 'EPSNormalized_std', 'EPSNormalized_guidance_high', 
    'EPSNormalized_guidance_low', 'revenue_actual', 
    'revenueDiff', 'revenue_surprise', 'revenue_count', 'revenue_std',
    'revenue_guidance_high', 'revenue_guidance_low'
}

def drop_high_corr(df: pd.DataFrame, threshold: float = 0.8, target_var: str = 'EPSNormalized_surprise', output_dir: str = "output_data", keep_variables: Set[str] = KEEP_VARIABLES) -> pd.DataFrame:
    """
    Check correlations and remove highly correlated variables based on their correlation with target variable.
    Variables in keep_variables set will not be dropped regardless of correlation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with variables to check
    threshold : float, default=0.8
        Correlation threshold to report
    target_var : str, default='EPSNormalized_surprise'
        Target variable to compare correlations against
    output_dir : str, default="output_data"
        Directory to save the output files
    keep_variables : Set[str], default=KEEP_VARIABLES
        Set of variables to keep regardless of correlation
        
    Returns
    -------
    pd.DataFrame
        DataFrame with highly correlated variables removed (except keep_variables)
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
                
                # Skip if both variables are in keep_variables
                if var1 in keep_variables and var2 in keep_variables:
                    continue
                
                # Count variable occurrences
                var_count[var1] += 1
                var_count[var2] += 1
                
                # Get non-missing value counts
                non_missing_var1 = numeric_df[var1].count()
                non_missing_var2 = numeric_df[var2].count()
                non_missing_both = numeric_df[[var1, var2]].dropna().shape[0]
                
                # Determine which variable to drop based on correlation with target and keep_variables
                var1_target_corr = target_corrs[var1]
                var2_target_corr = target_corrs[var2]
                
                # If one variable is in keep_variables, drop the other
                if var1 in keep_variables:
                    dropped_var = var2
                elif var2 in keep_variables:
                    dropped_var = var1
                # Otherwise, drop the one with weaker correlation to target
                else:
                    dropped_var = var2 if var1_target_corr >= var2_target_corr else var1
                
                dropped_var_list.add(dropped_var)
                
                high_corr.append({
                    'variable1': var1,
                    'variable2': var2,
                    'correlation': correlation,
                    'non_missing_var1': non_missing_var1,
                    'non_missing_var2': non_missing_var2,
                    'non_missing_both': non_missing_both,
                    'var1_target_corr': var1_target_corr,
                    'var2_target_corr': var2_target_corr,
                    'dropped_variable': dropped_var,
                    'keep_variable_involved': var1 in keep_variables or var2 in keep_variables
                })
                high_corr_vars.add(var1)
                high_corr_vars.add(var2)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save correlation matrix to file
    output_path = os.path.join(output_dir, "correlation_matrix.csv")
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
                f"Dropped: {corr_info['dropped_variable']}, Keep involved: {corr_info['keep_variable_involved']})"
            )
        
        # Save detailed correlations to CSV
        high_corr_path = os.path.join(output_dir, "high_corr_var_list.csv")
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
