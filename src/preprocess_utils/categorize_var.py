import pandas as pd
from typing import List, Set, Union, Tuple

# internal imports
from logger import setup_logger

# setup logger
logger = setup_logger(__name__)

def encode_categorical_variables(
    df: pd.DataFrame,
    categorical_columns: Set[str] = None,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    One-hot encode categorical variables (calendaryear).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns to encode.
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
    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Set default and validate categorical columns
    categorical_columns = categorical_columns or {'calendaryear'}
    if missing := categorical_columns - set(df.columns):
        raise ValueError(f"Missing required categorical columns: {missing}")
    
    # Create working copy
    df_encoded = df.copy()
    
    # Process each categorical column
    for column in categorical_columns:
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

    
    logger.info("Completed categorical encoding")
    return df_encoded
