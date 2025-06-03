import pandas as pd
import multiprocessing as mp
import numpy as np
from functools import partial

# internal imports
from ml_utils.ml_framework import MLFramework

def run_ml_for_target(target_col: str, df: pd.DataFrame, output_dir: str = '') -> str:
    """Run ML framework for a specific target column."""
    print(f"Processing target: {target_col}")
    
    # Prepare the data
    feature_cols = [col for col in df.columns if col.startswith('x_')]
        
    # Initialize the framework
    ml = MLFramework(df, output_dir=f'{output_dir}/ml_run_{target_col}')

    ml.prepare_data(feature_cols, target_col)
    
    # Train and evaluate models
    scores = ml.train_models(cv=5)
    
    # Evaluate the best model
    metrics = ml.evaluate_model()
    
    print(f"Completed processing target: {target_col}")
    return target_col

if __name__ == "__main__":
    # Load data once
    df = pd.read_parquet('output_data/cleaned_data.parquet')
    print("Available columns:", df.columns.tolist())

    # Define target columns
    target_cols = [
        'y_EPS_surprise', 
        'y_EPSNormalized_surprise', 
        'y_revenue_surprise', 
        'y_EPS_SUE', 
        'y_EPSNorm_SUE', 
        'y_revenue_SUE'
    ]

    #############################################
    # ALL DATA
    #############################################
    if True:
        # Create a partial function with the DataFrame
        run_ml = partial(run_ml_for_target, df=df, output_dir='output_data/ml_run_all')

        with mp.Pool(processes=len(target_cols)) as pool:
            # Use imap_unordered to get results as they complete
            for completed_target in pool.imap_unordered(run_ml, target_cols):
                print(f"Finished processing: {completed_target}")


    #############################################
    # HETEROGENEITY ANALYSIS: ATTENTION (EPS_count)
    #############################################
    if True:
        # choose top 1/3 of EPS_count
        high_attention = df.sort_values(by='EPS_count', ascending=False).head(len(df) // 3)
        # Create a partial function with the DataFrame
        run_ml = partial(run_ml_for_target, df=high_attention, output_dir='output_data/heterogeneity/ml_run_high_attention')

        with mp.Pool(processes=len(target_cols)) as pool:
            # Use imap_unordered to get results as they complete
            for completed_target in pool.imap_unordered(run_ml, target_cols):
                print(f"Finished processing: {completed_target}")

        # # choose bottom 1/3 of EPS_count
        low_attention = df.sort_values(by='EPS_count', ascending=True).head(len(df) // 3)
        # Create a partial function with the DataFrame
        run_ml = partial(run_ml_for_target, df=low_attention, output_dir='output_data/heterogeneity/ml_run_low_attention')

        with mp.Pool(processes=len(target_cols)) as pool:
            # Use imap_unordered to get results as they complete
            for completed_target in pool.imap_unordered(run_ml, target_cols):
                print(f"Finished processing: {completed_target}")

    
    #############################################
    # HETEROGENEITY ANALYSIS: CALENDAR YEAR
    #############################################
    if True:
        for year in range(2009, 2023):
            year_df = df[df[f'dummy_calendaryear_{year}'] == 1]
            # Create a partial function with the DataFrame
            run_ml = partial(run_ml_for_target, df=year_df, output_dir=f'output_data/time_variance/ml_run_year_{year}')

            with mp.Pool(processes=len(target_cols)) as pool:
                # Use imap_unordered to get results as they complete
                for completed_target in pool.imap_unordered(run_ml, target_cols):
                    print(f"Finished processing: {completed_target}")
