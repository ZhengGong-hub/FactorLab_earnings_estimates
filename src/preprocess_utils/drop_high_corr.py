import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Dict
import logging
import os
from collections import Counter

# imports
from logger import setup_logger
from constants import EXCLUDE_VARIABLES

# logger setup
logger = setup_logger(__name__)

# keep variables
KEEP_VARIABLES = EXCLUDE_VARIABLES.union({'EPSSurpC', 'IndRel_EPSSurpC', 'IndRel_SUEC', 'SUEC'})

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
    
    # select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame")
        
    logger.info(f"Computing correlations for {len(numeric_df.columns)} numeric variables")
    
    # correlation matrix
    corr_matrix = numeric_df.corr()
    target_corrs = corr_matrix[target_var].abs()
    
    # track correlations and variables
    high_corr = []
    high_corr_vars = set()
    var_count = Counter()
    dropped_var_list = set()
    
    # check correlations
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            correlation = corr_matrix.iloc[i, j]
            if abs(correlation) > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                
                if var1 in keep_variables and var2 in keep_variables:
                    continue
                
                var_count[var1] += 1
                var_count[var2] += 1
                
                # count non-missing values
                non_missing_var1 = numeric_df[var1].count()
                non_missing_var2 = numeric_df[var2].count()
                non_missing_both = numeric_df[[var1, var2]].dropna().shape[0]
                
                # determine which variable to drop
                var1_target_corr = target_corrs[var1]
                var2_target_corr = target_corrs[var2]
                
                if var1 in keep_variables:
                    dropped_var = var2
                elif var2 in keep_variables:
                    dropped_var = var1
                else:
                    dropped_var = var2 if var1_target_corr >= var2_target_corr else var1
                
                dropped_var_list.add(dropped_var)
                
                # record correlation info
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
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # save correlation matrix
    output_path = os.path.join(output_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(output_path)
    logger.info(f"Correlation matrix saved to: {output_path}")
    
    # process and save high correlations
    if high_corr:
        logger.info(f"\nHigh correlations (>{threshold}):")
        high_corr_sorted = sorted(high_corr, key=lambda x: abs(x['correlation']), reverse=True)
        
        # log correlations
        for corr_info in high_corr_sorted:
            logger.info(
                f"{corr_info['variable1']:<20} -- {corr_info['variable2']:<20}: "
                f"{corr_info['correlation']:>6.3f} "
                f"(Target corr: {corr_info['var1_target_corr']:.3f} vs {corr_info['var2_target_corr']:.3f}, "
                f"Dropped: {corr_info['dropped_variable']}, Keep involved: {corr_info['keep_variable_involved']})"
            )
        
        # save detailed correlations
        high_corr_path = os.path.join(output_dir, "high_corr_var_list.csv")
        pd.DataFrame(high_corr_sorted).to_csv(high_corr_path, index=False)
        logger.info(f"High correlation details saved to: {high_corr_path}")
        
        # save dropped variables
        drop_path = os.path.join(output_dir, "dropped_var_list.txt")
        with open(drop_path, 'w') as f:
            for var in sorted(dropped_var_list):
                f.write(f"{var}\n")
        logger.info(f"List of variables to drop saved to: {drop_path}")
        
        # save variable count summary
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
    
    logger.info(f"Dropping {len(dropped_var_list)} variables due to high correlation")
    return df.drop(columns=list(dropped_var_list))
