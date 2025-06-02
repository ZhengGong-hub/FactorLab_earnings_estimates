"""Plot ML results visualization module"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import os
import logging
from .plot_style import (
    set_default_style,
    get_figure_size,
    style_ml_metrics_plot,
    set_color_scheme
)

# logger setup
logger = logging.getLogger(__name__)

def plot_metrics_over_time(
    df_results: pd.DataFrame,
    output_dir: str = 'output_data/plots',
    metrics: Optional[List[str]] = None,
    figsize: Optional[tuple] = None
) -> None:
    """
    Generate plots showing how different metrics change over years for each variable.
    Creates one plot per metric, with lines for each variable.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame containing ML results with columns: year, variable_name, metric, value
    output_dir : str, default='output_data/plots'
        Directory to save the plots
    metrics : Optional[List[str]], default=None
        List of metrics to plot. If None, plots all metrics (MSE, RMSE, MAE, R2)
    figsize : Optional[tuple], default=None
        Custom figure size. If None, uses predefined size from plot_style
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default metrics if none provided
    if metrics is None:
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    
    # Set default style
    set_default_style()
    
    # Get color palette for ML metrics
    colors = set_color_scheme('ml_metrics')
    
    # Create one plot per metric
    for metric in metrics:
        # Filter data for current metric
        metric_data = df_results[df_results['metric'] == metric]
        
        if metric_data.empty:
            logger.warning(f"No data found for metric: {metric}")
            continue
            
        # Create figure with standard size if not specified
        if figsize is None:
            figsize = get_figure_size('ml_metrics')
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create line plot
        sns.lineplot(
            data=metric_data,
            x='year',
            y='value',
            hue='variable_name',
            marker='o',
            markersize=8,
            palette=colors,
            ax=ax
        )
        
        # Apply consistent styling
        style_ml_metrics_plot(ax, f'{metric} Over Time by Variable', metric)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(output_dir, f'{metric.lower()}_over_time.png')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated plot for {metric}: {output_file}")
    
    logger.info(f"All plots have been saved to {output_dir}")
