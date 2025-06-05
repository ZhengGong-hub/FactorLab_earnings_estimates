import pandas as pd
import logging
from pathlib import Path
from summary_utils.y_variable_analysis import analyze_y_variables, summarize_y_variables_by_attention

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def describe_data():
    """
    Run descriptive analysis on the cleaned dataset by attention groups.
    
    The analysis includes:
    1. Basic summary statistics by attention group
       - Saved to output_data/stats/y_variables_summary_{high|low}.csv
    
    2. Detailed analysis for each attention group:
       - Basic statistics (mean, std, etc.)
       - Distribution analysis with plots
       - Correlation analysis between y-variables
       - Time series patterns by quarter
    
    All detailed results are saved in:
       output_data/analysis/high_attention/
       output_data/analysis/low_attention/
    """
    # First, generate summary statistics for both attention groups
    logger.info("Generating summary statistics by attention group...")
    summarize_y_variables_by_attention(
        high_path=Path("input_data/data_sample_high_attention.parquet"),
        low_path=Path("input_data/data_sample_low_attention.parquet"),
        save_dir=Path("output_data/stats")
    )
    
    # Then do detailed analysis for each attention group
    attention_groups = {
        "high": "input_data/data_sample_high_attention.parquet",
        "low": "input_data/data_sample_low_attention.parquet"
    }
    
    for attention, data_file in attention_groups.items():
        logger.info(f"\nAnalyzing {attention}-attention group...")
        
        # Load data for this attention group
        df = pd.read_parquet(data_file)
        logger.info(f"Loaded {attention}-attention data: {len(df)} rows, {len(df.columns)} columns")
        
        # Run detailed analysis
        output_dir = f"output_data/analysis/{attention}_attention"
        results = analyze_y_variables(df, output_dir=output_dir)
        logger.info(f"Completed detailed analysis for {attention}-attention group")

if __name__ == "__main__":
    describe_data()