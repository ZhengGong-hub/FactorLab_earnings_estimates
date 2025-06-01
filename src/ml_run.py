import pandas as pd
import multiprocessing as mp
import numpy as np
from functools import partial

# internal imports
from ml_utils.ml_framework import MLFramework

def run_ml_for_target(target_col: str, df: pd.DataFrame) -> str:
    """Run ML framework for a specific target column."""
    print(f"Processing target: {target_col}")
    
    # Prepare the data
    feature_cols = [col for col in df.columns if col.startswith('x_')]
        
    # Initialize the framework
    ml = MLFramework(df, output_dir=f'output_data/ml_run_{target_col}')

    ml.prepare_data(feature_cols, target_col)
    
    # Train and evaluate models
    scores = ml.train_models(cv=2)
    
    # Evaluate the best model
    metrics = ml.evaluate_model()
    
    print(f"Completed processing target: {target_col}")
    return target_col

if __name__ == "__main__":
    # Load data once
    df = pd.read_parquet('output_data/cleaned_data.parquet')
    print("Available columns:", df.columns.tolist())

    # df.to_csv('output_data/cleaned_data.csv')
    # assert False

    # Define target columns
    target_cols = [
        'y_EPS_surprise', 
        'y_EPSNormalized_surprise', 
        'y_revenue_surprise', 
        'y_EPS_SUE', 
        'y_EPSNorm_SUE', 
        'y_revenue_SUE'
    ]

    # Create a partial function with the DataFrame
    run_ml = partial(run_ml_for_target, df=df)

    with mp.Pool(processes=len(target_cols)) as pool:
        # Use imap_unordered to get results as they complete
        for completed_target in pool.imap_unordered(run_ml, target_cols):
            print(f"Finished processing: {completed_target}")
