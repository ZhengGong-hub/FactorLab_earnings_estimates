import pandas as pd
from typing import List, Set, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_categorical_variables(
    df: Union[pd.DataFrame, Tuple],
    categorical_columns: Set[str] = None,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    One-hot encode categorical variables (companyid, securityid, simpleindustryid).
    
    Parameters
    ----------
    df : Union[pd.DataFrame, Tuple]
        Input DataFrame containing categorical columns to encode.
        If a tuple is provided, the first element should be the DataFrame.
    categorical_columns : Set[str], optional
        Set of columns to encode. If None, uses default columns
    drop_original : bool, default=True
        Whether to drop original categorical columns after encoding
        
    Returns
    -------
    pd.DataFrame
        DataFrame with categorical variables one-hot encoded
        
    Examples
    --------
    >>> df_encoded = encode_categorical_variables(df)
    >>> # Or with custom columns:
    >>> df_encoded = encode_categorical_variables(df, {'companyid', 'custom_category'})
    
    Raises
    ------
    TypeError
        If input is not a DataFrame or tuple containing DataFrame
    ValueError
        If required columns are missing
    """
    # Handle input that might be a tuple
    if isinstance(df, tuple):
        if len(df) == 0 or not isinstance(df[0], pd.DataFrame):
            raise TypeError("If input is a tuple, first element must be a DataFrame")
        df = df[0]
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame or tuple containing DataFrame")
    
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")
        
    # Default categorical columns if none provided
    if categorical_columns is None:
        categorical_columns = {'simpleindustryid','calendaryear'}
    
    # Validate columns exist
    missing_cols = categorical_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required categorical columns: {missing_cols}")
    
    # Create working copy
    df_encoded = df.copy()
    
    # Process each categorical column
    for column in categorical_columns:
        try:
            # Perform one-hot encoding
            encoded = pd.get_dummies(
                df_encoded[column],
                prefix=f'dummy_{column}',
                prefix_sep='_'
            )
            
            # Add encoded columns
            df_encoded = pd.concat([df_encoded, encoded], axis=1)
            
            # Drop original column if requested
            if drop_original:
                df_encoded = df_encoded.drop(columns=[column])
                
            logger.info(f"Encoded {column} into {encoded.shape[1]} categories")
            
        except Exception as e:
            logger.error(f"Error encoding column {column}: {str(e)}")
            raise
    
    logger.info("Completed categorical encoding")
    return df_encoded
