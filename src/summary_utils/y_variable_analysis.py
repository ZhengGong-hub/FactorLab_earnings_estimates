import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import os
from scipy import stats
from .plot_style import (
    set_default_style,
    get_figure_size,
    style_distribution_plot,
    style_correlation_plot,
    style_time_series_plot,
    set_color_scheme
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_y_variables(df: pd.DataFrame, output_dir: str = "output_data", sample_size: int = 10000) -> Dict:
    """
    Analyze y-variables with efficient memory handling for large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data containing y-variables
    output_dir : str
        Directory to save analysis outputs
    sample_size : int
        Size of random sample for distribution analysis
        
    Returns
    -------
    Dict
        Analysis results
    """
    # Set default plotting style
    set_default_style()
    
    # Create output directories
    plots_dir = os.path.join(output_dir, "plots")
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Get y-variables
    y_vars = [col for col in df.columns if col.startswith('y_')]
    if not y_vars:
        raise ValueError("No y-variables found in the dataset")
    
    logger.info(f"Analyzing {len(y_vars)} y-variables")
    
    # Create sample for distribution analysis
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        logger.info(f"Using random sample of {sample_size} rows for distribution analysis")
    else:
        df_sample = df
    
    results = {}
    
    # 1. Basic Statistics
    results['basic_stats'] = _calculate_basic_stats(df[y_vars], stats_dir)
    
    # 2. Distribution Analysis
    results['distribution'] = _analyze_distributions(df_sample[y_vars], plots_dir, stats_dir)
    
    # 3. Correlation Analysis
    results['correlation'] = _analyze_correlations(df_sample[y_vars], plots_dir, stats_dir)
    
    # 4. Time Series Analysis
    required_cols = ['calendaryear', 'quarter_factor']
    if all(col in df.columns for col in required_cols):
        results['time_series'] = _analyze_time_series(df, y_vars, plots_dir, stats_dir)
    else:
        logger.warning(f"Missing required columns for time series analysis: {[col for col in required_cols if col not in df.columns]}")
    
    # Save results to CSV
    _save_results(results, stats_dir)
    
    return results

def _calculate_basic_stats(df: pd.DataFrame, stats_dir: str) -> pd.DataFrame:
    """Calculate basic statistics for y-variables"""
    stats = df.agg(['count', 'mean', 'std', 'min', 'max', 'median', 
                    lambda x: x.skew(), lambda x: x.kurtosis()])
    stats.index = ['count', 'mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']
    
    
    return stats

def _analyze_distributions(df: pd.DataFrame, plots_dir: str, stats_dir: str) -> Dict:
    """Analyze distributions with memory-efficient binning"""
    dist_stats = {}
    
    for col in df.columns:
        data = df[col].dropna()
        
        # Calculate optimal number of bins
        n_bins = min(int(np.sqrt(len(data))), 50)
        
        # Distribution plot
        fig, ax = plt.subplots(figsize=get_figure_size('distribution'))
        sns.histplot(data=data, bins=n_bins, kde=False, ax=ax)
        style_distribution_plot(ax, f'Distribution of {col}', col)
        plt.savefig(os.path.join(plots_dir, f'{col}_distribution.png'))
        plt.close()
        
        # Boxplot
        fig, ax = plt.subplots(figsize=get_figure_size('boxplot'))
        sns.boxplot(y=data, ax=ax)
        style_distribution_plot(ax, f'Boxplot of {col}', col)
        plt.savefig(os.path.join(plots_dir, f'{col}_boxplot.png'))
        plt.close()
        
        # Calculate distribution statistics
        dist_stats[col] = {
            'normality_test': stats.normaltest(data),
            'quantiles': data.quantile([0.25, 0.5, 0.75]).to_dict()
        }
    
    logger.info("Distribution analysis completed")
    return dist_stats

def _analyze_correlations(df: pd.DataFrame, plots_dir: str, stats_dir: str) -> pd.DataFrame:
    """Analyze correlations with memory efficiency"""
    corr = df.corr()
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=get_figure_size('correlation'))
    mask = np.triu(np.ones_like(corr), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap=set_color_scheme('correlation'), center=0, ax=ax)
    style_correlation_plot(fig, ax, 'Correlation Matrix of Y Variables')
    plt.savefig(os.path.join(plots_dir, 'y_variables_correlation.png'))
    plt.close()
    
    # Save correlation matrix
    corr.to_csv(os.path.join(stats_dir, 'y_variables_correlation.csv'))
    logger.info("Correlation analysis completed")
    
    return corr

def _analyze_time_series(df: pd.DataFrame, y_vars: List[str], plots_dir: str, stats_dir: str) -> Dict:
    """Analyze time series with efficient aggregation"""
    ts_stats = {}
    
    # Yearly aggregation
    yearly_stats = df.groupby('calendaryear')[y_vars].agg(['mean', 'std', 'count'])
    year_quarter_stats = df.groupby(['calendaryear', 'quarter_factor'])[y_vars].agg(['mean', 'std', 'count'])
    
    for var in y_vars:
        # 1. Yearly trend
        fig, ax = plt.subplots(figsize=get_figure_size('time_series'))
        yearly_means = df.groupby('calendaryear')[var].mean()
        yearly_std = df.groupby('calendaryear')[var].std()
        
        ax.errorbar(yearly_means.index, yearly_means.values,
                   yerr=yearly_std.values, fmt='o-', capsize=5)
        style_time_series_plot(ax, f'Yearly Trend: {var}', 'Year', 'Value')
        plt.savefig(os.path.join(plots_dir, f'{var}_yearly_trend.png'))
        plt.close()
        
        # 2. Quarterly patterns within years
        fig, ax = plt.subplots(figsize=get_figure_size('multi_series'))
        for year in sorted(df['calendaryear'].unique()):
            year_data = df[df['calendaryear'] == year]
            quarterly_means = year_data.groupby('quarter_factor')[var].mean()
            quarterly_std = year_data.groupby('quarter_factor')[var].std()
            
            ax.errorbar(quarterly_means.index, quarterly_means.values,
                       yerr=quarterly_std.values, fmt='o-', label=f'Year {year}',
                       capsize=5)
        
        style_time_series_plot(ax, f'Quarterly Patterns by Year: {var}', 'Quarter', 'Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{var}_quarterly_by_year.png'))
        plt.close()
        
        # 3. Heatmap of quarterly patterns across years
        fig, ax = plt.subplots(figsize=get_figure_size('heatmap'))
        pivot_data = df.pivot_table(
            values=var,
            index='calendaryear',
            columns='quarter_factor',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f',
                   cmap=set_color_scheme('correlation'), center=0, ax=ax)
        style_correlation_plot(fig, ax, f'Year-Quarter Heatmap: {var}')
        plt.savefig(os.path.join(plots_dir, f'{var}_year_quarter_heatmap.png'))
        plt.close()
    
    # Combined year-over-year comparison
    fig, ax = plt.subplots(figsize=get_figure_size('combined'))
    for var in y_vars:
        yearly_means = df.groupby('calendaryear')[var].mean()
        ax.plot(yearly_means.index, yearly_means.values, 'o-', label=var)
    
    style_time_series_plot(ax, 'Year-over-Year Comparison of All Variables', 'Year', 'Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_variables_yearly_comparison.png'))
    plt.close()
    
    # Save statistics
    yearly_stats.to_csv(os.path.join(stats_dir, 'y_variables_yearly_stats.csv'))
    year_quarter_stats.to_csv(os.path.join(stats_dir, 'y_variables_year_quarter_stats.csv'))
    
    ts_stats = {
        'yearly_stats': yearly_stats,
        'year_quarter_stats': year_quarter_stats
    }
    
    logger.info("Time series analysis completed")
    return ts_stats

def _save_results(results: Dict, stats_dir: str):
    """Save analysis results to files"""
    # Save basic stats
    if 'basic_stats' in results:
        results['basic_stats'].to_csv(os.path.join(stats_dir, 'y_variables_summary.csv'))
    
    # Save distribution stats
    if 'distribution' in results:
        pd.DataFrame(results['distribution']).to_csv(os.path.join(stats_dir, 'y_variables_distribution_stats.csv'))
    
    logger.info("Analysis results saved to files") 