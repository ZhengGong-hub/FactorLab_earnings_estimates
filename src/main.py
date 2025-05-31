import pandas as pd
from pathlib import Path

# internal imports
from logger import setup_logger
import preprocess as pp
from desciption_statistics import describe_data

# setup logger
logger = setup_logger(__name__)

if __name__ == "__main__":
    # Create output directories
    Path('output_data').mkdir(exist_ok=True)
    Path('output_data/plots').mkdir(exist_ok=True)
    Path('output_data/stats').mkdir(exist_ok=True)

    # Load data
    df = pd.read_parquet('input_data/universe_with_affactor.parquet')
    logger.info(f"Original data shape: {df.shape}")

    if 'calendaryear' not in df.columns or 'calendarquarter' not in df.columns:
        # Convert date to quarter number (1-4)
        df['calendarquarter'] = pd.to_datetime(df['EPS_actual_et']).dt.quarter
        df['calendaryear'] = pd.to_datetime(df['EPS_actual_et']).dt.year
        logger.info("Created calendaryear and calendarquarter columns")

    # manually banned columns
    manually_banned_cols = ['AE-style', 'EQ-style', 'CE-style', 'HG-style', 'PM-style', 'Sz-style', 'Val-style', 'Vol-style',]
    df = df.drop(columns=manually_banned_cols)
    logger.info(f"Dropped {len(manually_banned_cols)} columns: {manually_banned_cols}")

    # Run preprocessing pipeline
    df_processed = pp.preprocess_pipeline(
        df,
        missing_threshold=20,  # Remove columns with more than 20% missing values
        correlation_threshold=0.8,  # Remove highly correlated variables
        variance_threshold=0.01,  # Remove variables with variance below 0.01
        same_value_threshold=0.95  # Remove variables with more than 95% same value
    )
    logger.info(f"Processed data shape: {df_processed.shape}")
    
    # Save processed data
    df_processed.to_parquet('output_data/cleaned_data.parquet', index=False)
    logger.info("Processed data saved to output_data/cleaned_data.parquet")

    # description statistics
    describe_data(df_processed)


    
