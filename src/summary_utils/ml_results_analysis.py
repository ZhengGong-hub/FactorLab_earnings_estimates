"""ML results analysis main module"""

import os
import re
import pandas as pd
import logging
from .plot_ml_results import plot_metrics_over_time

# logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_ml_results(ml_results_dir: str) -> pd.DataFrame:
    """
    Extract machine learning results from all years and variables.
    
    Parameters
    ----------
    ml_results_dir : str
        Directory containing ML results folders
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, variable_name, metrics, value
    """
    results_data = []
    
    # get all year directories
    year_pattern = re.compile(r'ml_run_year_(\d{4})')
    var_pattern = re.compile(r'ml_run_y_(.+)')
    
    try:
        # walk through all subdirectories
        for root, dirs, files in os.walk(ml_results_dir):
            # check if this is a year directory
            year_match = year_pattern.search(root)
            if year_match:
                year = int(year_match.group(1))
                
                # check for variable directories
                for dir_name in dirs:
                    var_match = var_pattern.search(dir_name)
                    if var_match:
                        variable_name = var_match.group(1)
                        
                        # read ml_run.txt file
                        ml_run_file = os.path.join(root, dir_name, 'ml_run.txt')
                        if os.path.exists(ml_run_file):
                            with open(ml_run_file, 'r') as f:
                                content = f.read()
                                
                                # extract metrics
                                metrics_pattern = r'- (MSE|RMSE|MAE|R2): ([\d.]+)'
                                metrics_matches = re.finditer(metrics_pattern, content)
                                
                                for match in metrics_matches:
                                    metric, value = match.groups()
                                    results_data.append({
                                        'year': year,
                                        'variable_name': variable_name,
                                        'metric': metric,
                                        'value': float(value)
                                    })
        
        # create DataFrame
        if not results_data:
            logger.warning("No ML results found in the specified directory")
            return pd.DataFrame(columns=['year', 'variable_name', 'metric', 'value'])
        
        df_results = pd.DataFrame(results_data)
        
        # sort for better organization
        df_results = df_results.sort_values(['year', 'variable_name', 'metric'])
        
        logger.info(f"Successfully extracted ML results for {df_results['year'].nunique()} years "
                   f"and {df_results['variable_name'].nunique()} variables")
        
        return df_results
    
    except Exception as e:
        logger.error(f"Error extracting ML results: {str(e)}")
        raise

def ml_summary(input_dir: str = 'input_data/ml_results', output_dir: str = 'output_data'):
    """
    Extract ML results, save to CSV, and generate visualization plots.
    
    Parameters
    ----------
    input_dir : str, default='input_data/ml_results'
        Directory containing ML results folders
    output_dir : str, default='output_data'
        Directory to save analysis outputs and plots
        
    Returns
    -------
    pd.DataFrame
        Raw ML results DataFrame
    """
    logger.info("Extracting ML results...")
    ml_results = extract_ml_results(input_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results to CSV
    logger.info("Saving raw ML results...")
    output_file = os.path.join(output_dir, 'ml_results_all.csv')
    ml_results.to_csv(output_file, index=False)
    logger.info(f"ML results saved to {output_file}")
    
    # Generate and save plots
    logger.info("Generating ML metrics plots...")
    plots_dir = os.path.join(output_dir, 'plots')
    plot_metrics_over_time(
        df_results=ml_results,
        output_dir=plots_dir
    )
    
    return ml_results
