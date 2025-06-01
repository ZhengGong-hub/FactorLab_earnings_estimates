import pandas as pd
import numpy as np
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FINANCIAL_VARS = {
    'EPS_surprise', 'EPSNormalized_surprise', 'revenue_surprise',
    'EPS_SUE', 'EPSNorm_SUE', 'revenue_SUE'
}

def winsorize_financial_variables(df, variables=None, lower_percentile=0.05, upper_percentile=0.95):
    """Winsorize financial variables to handle outliers."""

    df_clean = df.copy()

    variables = variables or FINANCIAL_VARS
    existing_vars = variables.intersection(df.columns)
    
    if not existing_vars:
        logger.warning("No variables to winsorize")
        return df_clean
    
    for var in existing_vars:
        orig_mean, orig_std = df_clean[var].mean(), df_clean[var].std()
        lower_bound = np.percentile(df_clean[var].dropna(), lower_percentile * 100)
        upper_bound = np.percentile(df_clean[var].dropna(), upper_percentile * 100)
        
        df_clean[var] = df_clean[var].clip(lower=lower_bound, upper=upper_bound)
        
        new_mean, new_std = df_clean[var].mean(), df_clean[var].std()
        logger.info(f"Winsorized {var}: Bounds [{lower_bound:.2f}, {upper_bound:.2f}], "
                    f"Mean {orig_mean:.2f} → {new_mean:.2f}, Std {orig_std:.2f} → {new_std:.2f}")

    return df_clean

def plot_winsorization_effect(df_original, df_winsorized, variable):
    """Plot histograms comparing original and winsorized data."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(df_original[variable].dropna(), bins=50, ax=ax1)
        sns.histplot(df_winsorized[variable].dropna(), bins=50, ax=ax2)
        
        ax1.set_title(f'Original {variable}')
        ax2.set_title(f'Winsorized {variable}')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Plotting requires matplotlib and seaborn")
