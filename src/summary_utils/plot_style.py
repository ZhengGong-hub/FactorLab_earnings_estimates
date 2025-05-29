"""
Plot styling utilities for consistent visualization across analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def set_default_style():
    """Set default style for all plots"""
    plt.style.use('seaborn')
    sns.set_palette("deep")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10

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
        'distribution': (10, 6),
        'boxplot': (8, 6),
        'correlation': (10, 8),
        'time_series': (12, 6),
        'combined': (15, 8),
        'heatmap': (10, 8),
        'multi_series': (15, 8)
    }
    return sizes.get(plot_type, (10, 6))

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
    ax.set_title(title, pad=20)
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
    ax.set_title(title, pad=20)
    fig.tight_layout()

def style_time_series_plot(ax, title: str, xlabel: str = 'Time', ylabel: str = 'Value'):
    """Style for time series plots"""
    style_axis(ax, title, xlabel, ylabel, rotate_xticks=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def set_color_scheme(plot_type: str):
    """
    Set color scheme for different plot types
    
    Parameters
    ----------
    plot_type : str
        Type of plot
    """
    if plot_type == 'correlation':
        return sns.color_palette("coolwarm", as_cmap=True)
    elif plot_type == 'distribution':
        return sns.color_palette("deep")
    elif plot_type == 'time_series':
        return sns.color_palette("husl", 8)
    return sns.color_palette("deep") 