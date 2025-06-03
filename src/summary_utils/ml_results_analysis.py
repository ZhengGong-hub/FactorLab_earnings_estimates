"""ML results analysis module for aggregating feature importance and model performance data."""

import pandas as pd
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_importance_data(df: pd.DataFrame, importance_data: List[Dict[str, Any]], 
                          metadata: Dict[str, str]) -> None:
    """
    Process importance data from a DataFrame, taking absolute values and filtering zeros.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature importance data
    importance_data : List[Dict[str, Any]]
        List to append processed data to
    metadata : Dict[str, str]
        Dictionary containing metadata to include with each row
    """
    for _, row in df.iterrows():
        abs_imp = abs(row['importance'])
        if abs_imp > 0:  # Only include non-zero importance
            record = {
                'feature_category': row['feature_category'],
                'feature': row['feature'],
                'feature_abs_importance': abs_imp,  # Renamed from abs_importance
                **metadata
            }
            importance_data.append(record)

def calculate_importance_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Calculate relative importance metrics at feature and category levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature importance data
    group_cols : List[str]
        Columns to group by for calculating relative importance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added importance metrics
    """
    # Calculate feature-level relative importance
    tot_feat_imp = df.groupby(group_cols)["feature_abs_importance"].transform("sum")
    df["feature_relative_importance"] = df["feature_abs_importance"] / tot_feat_imp
    
    # Calculate category-level importance
    cat_tot = df.groupby(group_cols + ["feature_category"])["feature_abs_importance"].transform("sum")
    df["category_abs_importance"] = cat_tot
    
    # Calculate category relative importance
    tot_cat_imp = df.groupby(group_cols)["category_abs_importance"].transform("max")
    df["category_relative_importance"] = df["category_abs_importance"] / tot_cat_imp
    
    return df

def aggregate_feature_importance(base_dir: str) -> pd.DataFrame:
    """
    Extract machine learning feature importance data from all ML runs.
    Only includes features with non-zero absolute importance.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing ml_results folders
        
    Returns
    -------
    pd.DataFrame
        Combined feature importance data with additional importance metrics
    """
    importance_data: List[Dict[str, Any]] = []
    base_path = Path(base_dir)
    
    # Find all feature importance files
    for importance_file in base_path.rglob('*_feature_importance.csv'):
        try:
            # Extract outcome from parent directory name
            outcome = importance_file.parent.parent.name.replace('ml_run_', '')
            
            # Extract model name from filename
            model = importance_file.stem.replace('_feature_importance', '')
            
            # Read the feature importance CSV
            df = pd.read_csv(importance_file)
            
            # Process importance data
            metadata = {'outcome': outcome, 'model': model}
            process_importance_data(df, importance_data, metadata)
                
        except Exception as e:
            logger.error(f"Error processing {importance_file}: {str(e)}")
            continue
    
    if not importance_data:
        logger.warning("No non-zero feature importance data found")
        return pd.DataFrame(columns=['outcome', 'model', 'feature_category', 'feature', 
                                   'feature_abs_importance', 'feature_relative_importance',
                                   'category_abs_importance', 'category_relative_importance'])
    
    # Create DataFrame and calculate importance metrics
    df_results = pd.DataFrame(importance_data)
    df_results = calculate_importance_metrics(df_results, ["model", "outcome"])
    logger.info(f"Extracted {len(df_results)} non-zero feature importance records")
    
    return df_results

def aggregate_model_performance(base_dir: str) -> pd.DataFrame:
    """
    Aggregate model performance data from all ML runs.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing ml_results folders
        
    Returns
    -------
    pd.DataFrame
        Combined model performance data
    """
    performance_data: List[Dict[str, Any]] = []
    base_path = Path(base_dir)
    
    # Process CV and OOS test scores
    for score_file in base_path.rglob('*scores.csv'):
        try:
            # Extract outcome from parent directory name
            outcome = score_file.parent.parent.name.replace('ml_run_', '')
            
            # Determine dataset type from filename
            dataset = 'cv' if 'cv_scores' in score_file.name else 'oos'
            
            # Read the scores CSV
            df = pd.read_csv(score_file)
            
            # Process each row
            for _, row in df.iterrows():
                performance_data.append({
                    'outcome': outcome,
                    'model': row['model'],
                    'dataset': dataset,
                    'r2': row['r2'],
                    'mse': row['mse'],
                    'rmse': row['rmse'],
                    'mae': row['mae']
                })
                
        except Exception as e:
            logger.error(f"Error processing {score_file}: {str(e)}")
            continue
    
    if not performance_data:
        logger.warning("No model performance data found")
        return pd.DataFrame(columns=['outcome', 'model', 'dataset', 'r2', 'mse', 'rmse', 'mae'])
    
    return pd.DataFrame(performance_data)

def build_heterogeneity_importance(in_dir: Path, out_file: Path) -> None:
    """
    Parse *_feature_importance.csv across high/low-attention runs,
    keep rows with abs(importance) > 0, and save combined CSV.
    
    Parameters
    ----------
    in_dir : Path
        Input directory containing heterogeneity analysis folders
    out_file : Path
        Output file path for the combined CSV
    """
    importance_data: List[Dict[str, Any]] = []
    
    # Find all feature importance files
    for importance_file in in_dir.rglob('*_feature_importance.csv'):
        try:
            # Extract attention group from parent folder name
            attention_group = "high_attention" if "high_attention" in str(importance_file) else "low_attention"
            
            # Extract outcome from parent directory name
            outcome = importance_file.parent.parent.name.replace('ml_run_', '')
            
            # Extract model name from filename
            model = importance_file.stem.replace('_feature_importance', '')
            
            # Read the feature importance CSV
            df = pd.read_csv(importance_file)
            
            # Process importance data
            metadata = {
                'attention_group': attention_group,
                'outcome': outcome,
                'model': model
            }
            process_importance_data(df, importance_data, metadata)
                
        except Exception as e:
            logger.error(f"Error processing heterogeneity importance file {importance_file}: {str(e)}")
            continue
    
    if not importance_data:
        logger.warning("No non-zero heterogeneity feature importance data found")
        return
    
    # Create output directory if it doesn't exist
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame and calculate importance metrics
    df_results = pd.DataFrame(importance_data)
    df_results = calculate_importance_metrics(df_results, ["model", "outcome", "attention_group"])
    
    # Save results
    df_results.to_csv(out_file, index=False)
    logger.info(f"Saved {len(df_results)} non-zero heterogeneity importance records to {out_file}")

def build_heterogeneity_performance(in_dir: Path, out_file: Path) -> None:
    """
    Parse cv_scores.csv and oos_test_scores.csv across high/low-attention runs,
    tag rows by dataset type, and save a combined CSV.
    
    Parameters
    ----------
    in_dir : Path
        Input directory containing heterogeneity analysis folders
    out_file : Path
        Output file path for the combined CSV
    """
    performance_data: List[Dict[str, Any]] = []
    
    # Process CV and OOS test scores
    for score_file in in_dir.rglob('*scores.csv'):
        try:
            # Extract attention group from parent folder name
            attention_group = "high_attention" if "high_attention" in str(score_file) else "low_attention"
            
            # Extract outcome from parent directory name
            outcome = score_file.parent.parent.name.replace('ml_run_', '')
            
            # Determine dataset type from filename
            dataset = 'cv' if 'cv_scores' in score_file.name else 'oos'
            
            # Read the scores CSV
            df = pd.read_csv(score_file)
            
            # Process each row
            for _, row in df.iterrows():
                performance_data.append({
                    'attention_group': attention_group,
                    'outcome': outcome,
                    'model': row['model'],
                    'dataset': dataset,
                    'r2': row['r2'],
                    'mse': row['mse'],
                    'rmse': row['rmse'],
                    'mae': row['mae']
                })
                
        except Exception as e:
            logger.error(f"Error processing heterogeneity performance file {score_file}: {str(e)}")
            continue
    
    if not performance_data:
        logger.warning("No heterogeneity performance data found")
        return
    
    # Create output directory if it doesn't exist
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create and save DataFrame
    df_results = pd.DataFrame(performance_data)
    df_results.to_csv(out_file, index=False)
    logger.info(f"Saved {len(df_results)} heterogeneity performance records to {out_file}")

def build_time_variance_importance(in_dir: Path, out_file: Path) -> None:
    """
    Gather *_feature_importance.csv across all years/outcomes,
    drop zero-importance rows, and save combined CSV.
    
    Parameters
    ----------
    in_dir : Path
        Input directory containing time variance analysis folders
    out_file : Path
        Output file path for the combined CSV
    """
    importance_data: List[Dict[str, Any]] = []
    year_pattern = re.compile(r'ml_run_year_(\d{4})')
    
    # Find all feature importance files
    for importance_file in in_dir.rglob('*_feature_importance.csv'):
        try:
            # Extract year from parent directory name
            year_match = year_pattern.search(str(importance_file))
            if not year_match:
                continue
            year = int(year_match.group(1))
            
            # Extract outcome from parent directory name
            outcome = importance_file.parent.parent.name.replace('ml_run_', '')
            
            # Extract model name from filename
            model = importance_file.stem.replace('_feature_importance', '')
            
            # Read the feature importance CSV
            df = pd.read_csv(importance_file)
            
            # Process importance data
            metadata = {
                'year': year,
                'outcome': outcome,
                'model': model
            }
            process_importance_data(df, importance_data, metadata)
                
        except Exception as e:
            logger.error(f"Error processing time variance importance file {importance_file}: {str(e)}")
            continue
    
    if not importance_data:
        logger.warning("No non-zero time variance feature importance data found")
        return
    
    # Create output directory if it doesn't exist
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame and calculate importance metrics
    df_results = pd.DataFrame(importance_data)
    df_results = calculate_importance_metrics(df_results, ["model", "outcome", "year"])
    
    # Save results
    df_results.to_csv(out_file, index=False)
    logger.info(f"Saved {len(df_results)} non-zero time variance importance records to {out_file}")

def build_time_variance_performance(in_dir: Path, out_file: Path) -> None:
    """
    Gather cv_scores.csv and oos_test_scores.csv across all years/outcomes,
    tag rows with dataset type, and save combined CSV.
    
    Parameters
    ----------
    in_dir : Path
        Input directory containing time variance analysis folders
    out_file : Path
        Output file path for the combined CSV
    """
    performance_data: List[Dict[str, Any]] = []
    year_pattern = re.compile(r'ml_run_year_(\d{4})')
    
    # Process CV and OOS test scores
    for score_file in in_dir.rglob('*scores.csv'):
        try:
            # Extract year from parent directory name
            year_match = year_pattern.search(str(score_file))
            if not year_match:
                continue
            year = int(year_match.group(1))
            
            # Extract outcome from parent directory name
            outcome = score_file.parent.parent.name.replace('ml_run_', '')
            
            # Determine dataset type from filename
            dataset = 'cv' if 'cv_scores' in score_file.name else 'oos'
            
            # Read the scores CSV
            df = pd.read_csv(score_file)
            
            # Process each row
            for _, row in df.iterrows():
                performance_data.append({
                    'year': year,
                    'outcome': outcome,
                    'model': row['model'],
                    'dataset': dataset,
                    'r2': row['r2'],
                    'mse': row['mse'],
                    'rmse': row['rmse'],
                    'mae': row['mae']
                })
                
        except Exception as e:
            logger.error(f"Error processing time variance performance file {score_file}: {str(e)}")
            continue
    
    if not performance_data:
        logger.warning("No time variance performance data found")
        return
    
    # Create output directory if it doesn't exist
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create and save DataFrame
    df_results = pd.DataFrame(performance_data)
    df_results.to_csv(out_file, index=False)
    logger.info(f"Saved {len(df_results)} time variance performance records to {out_file}")

def ml_summary(input_dir: str = 'input_data/ml_results/ml_run_all', 
              output_dir: str = 'output_data/ml_run_all',
              heterogeneity_input: str = 'input_data/ml_results/heterogeneity',
              heterogeneity_output: str = 'output_data/heterogeneity',
              time_variance_input: str = 'input_data/ml_results/time_variance',
              time_variance_output: str = 'output_data/time_variance'):
    """
    Aggregate ML results and save to CSV files.
    
    Parameters
    ----------
    input_dir : str
        Directory containing ML results folders
    output_dir : str
        Directory to save aggregated results
    heterogeneity_input : str
        Directory containing heterogeneity analysis folders
    heterogeneity_output : str
        Directory to save heterogeneity analysis results
    time_variance_input : str
        Directory containing time variance analysis folders
    time_variance_output : str
        Directory to save time variance analysis results
    """
    logger.info("Starting ML results aggregation...")
    
    # Regular ML results analysis
    logger.info("Processing main ML results...")
    importance_df = aggregate_feature_importance(input_dir)
    importance_output = Path(output_dir) / 'ml_all_importance.csv'
    importance_output.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_output, index=False)
    logger.info(f"Feature importance data saved to {importance_output}")
    
    performance_df = aggregate_model_performance(input_dir)
    performance_output = Path(output_dir) / 'ml_all_performance.csv'
    performance_df.to_csv(performance_output, index=False)
    logger.info(f"Model performance data saved to {performance_output}")
    
    # Heterogeneity analysis
    logger.info("Processing heterogeneity analysis...")
    heterogeneity_importance_output = Path(heterogeneity_output) / 'heterogeneity_importance.csv'
    build_heterogeneity_importance(Path(heterogeneity_input), heterogeneity_importance_output)
    
    heterogeneity_performance_output = Path(heterogeneity_output) / 'heterogeneity_performance.csv'
    build_heterogeneity_performance(Path(heterogeneity_input), heterogeneity_performance_output)
    
    # Time variance analysis
    logger.info("Processing time variance analysis...")
    time_variance_importance_output = Path(time_variance_output) / 'time_variance_importance.csv'
    build_time_variance_importance(Path(time_variance_input), time_variance_importance_output)
    
    time_variance_performance_output = Path(time_variance_output) / 'time_variance_performance.csv'
    build_time_variance_performance(Path(time_variance_input), time_variance_performance_output)
    
    logger.info("ML results aggregation completed successfully")

if __name__ == '__main__':
    ml_summary()

