import pandas as pd
from pathlib import Path

# internal imports
import preprocess as pp
from desciption_statistics import describe_data

if __name__ == "__main__":
    # Create output directories
    Path('output_data').mkdir(exist_ok=True)
    Path('output_data/plots').mkdir(exist_ok=True)
    Path('output_data/stats').mkdir(exist_ok=True)

    # Load data
    df = pd.read_parquet('input_data/universe_with_affactor.parquet')
    print("Original data shape:", df.shape)

    # Run preprocessing pipeline
    df_processed = pp.preprocess_pipeline(
        df,
        missing_threshold=20,  # Remove columns with more than 20% missing values
        correlation_threshold=0.8  # Remove highly correlated variables
    )
    print("Processed data shape:", df_processed.shape)
    
    # Save processed data
    df_processed.to_parquet('output_data/cleaned_data.parquet', index=False)
    print("Processed data saved to output_data/cleaned_data.parquet")

    # description statistics
    describe_data(df_processed)
    

    
