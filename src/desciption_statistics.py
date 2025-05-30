import pandas as pd
import logging
from pathlib import Path
from summary_utils.y_variable_analysis import analyze_y_variables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def describe_data(df: pd.DataFrame):
    """
    Run descriptive analysis on the cleaned dataset,
    focusing on y-variables (target variables).
    
    The analysis includes:
    1. Basic statistics (mean, std, etc.)
    2. Distribution analysis with plots
    3. Correlation analysis between y-variables
    4. Time series patterns by quarter
    
    All results are saved in output_data/analysis/
    """
    # Load cleaned data
    df = pd.read_parquet('output_data/cleaned_data.parquet')
    logger.info(f"Loaded cleaned data with {len(df)} rows and {len(df.columns)} columns")
    
    # Analyze y-variables
    results = analyze_y_variables(df)
    logger.info("Y-variable analysis completed successfully")