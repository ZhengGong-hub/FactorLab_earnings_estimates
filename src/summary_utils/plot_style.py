"""
Plot styling utilities for consistent visualization across analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import os
import sys
from pathlib import Path

# Add the src directory to sys.path to allow importing logger
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from logger import setup_logger

# setup logger
logger = setup_logger(__name__)

def set_default_style():
    """Set default style for all plots"""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Memory-efficient parameters
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['figure.max_open_warning'] = 50
        
        # Visual parameters
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['grid.color'] = '#E5E5E5'
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.5
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        # Memory optimization
        plt.rcParams['agg.path.chunksize'] = 10000
        
        logger.info("Successfully set plot style and parameters")
    except Exception as e:
        logger.error(f"Failed to set plot style: {str(e)}")
        raise
    
def get_figure_size(plot_type: str) -> Tuple[int, int]:
    """
    Get standardized figure sizes for different plot types
    
    Parameters
    ----------
    plot_type : str
        Type of plot ('distribution', 'correlation', 'time_series', 'combined')
        
    Returns
    -------
    Tuple[int, int]
        Width and height of the figure
    """
    sizes = {
        'distribution': (8, 5),
        'boxplot': (6, 4),
        'correlation': (8, 6),
        'time_series': (10, 5),
        'combined': (12, 6),
        'heatmap': (8, 6),
        'multi_series': (12, 6),
        'ml_metrics': (12, 8)  # Added size for ML metrics plots
    }
    return sizes.get(plot_type, (8, 5))

def style_axis(ax, title: str, xlabel: str, ylabel: str, rotate_xticks: bool = False):
    """
    Apply consistent styling to plot axis
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    rotate_xticks : bool, default=False
        Whether to rotate x-axis tick labels
    """
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

def style_distribution_plot(ax, title: str, var_name: str):
    """Style for distribution plots"""
    style_axis(ax, title, var_name, 'Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def style_correlation_plot(fig, ax, title: str):
    """Style for correlation heatmaps"""
    ax.set_title(title, pad=15)
    fig.tight_layout()

def style_time_series_plot(ax, title: str, xlabel: str = 'Time', ylabel: str = 'Value'):
    """Style for time series plots"""
    style_axis(ax, title, xlabel, ylabel, rotate_xticks=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def style_ml_metrics_plot(ax, title: str, metric: str):
    """
    Style specifically for ML metrics plots
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style
    title : str
        Plot title
    metric : str
        Metric name for y-axis label
    """
    style_axis(ax, title, 'Year', metric, rotate_xticks=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Customize legend
    ax.legend(
        title='Variables',
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

def set_color_scheme(plot_type: str):
    """
    Set color scheme for different plot types
    
    Parameters
    ----------
    plot_type : str
        Type of plot
    """
    if plot_type == 'correlation':
        return sns.color_palette("coolwarm", n_colors=11, as_cmap=True)
    elif plot_type == 'distribution':
        return sns.color_palette("deep", n_colors=6)
    elif plot_type == 'time_series':
        return sns.color_palette("husl", n_colors=8)
    elif plot_type == 'ml_metrics':
        return sns.color_palette("husl", n_colors=10)  # Added color scheme for ML metrics
    return sns.color_palette("deep", n_colors=6) 