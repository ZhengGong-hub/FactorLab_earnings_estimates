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

    # Run preprocessing pipeline
    df_processed = pp.preprocess_pipeline(
        df,
        missing_threshold=20,  # Remove columns with more than 20% missing values
        correlation_threshold=0.8  # Remove highly correlated variables
    )
    logger.info(f"Processed data shape: {df_processed.shape}")
    
    # Save processed data
    df_processed.to_parquet('output_data/cleaned_data.parquet', index=False)
    logger.info("Processed data saved to output_data/cleaned_data.parquet")

    # description statistics
    describe_data(df_processed)


    
