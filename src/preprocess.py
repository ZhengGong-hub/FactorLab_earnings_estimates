import pandas as pd
import numpy as np
from pathlib import Path
import re

from preprocess_utils.near_duplicates_var import extract_near_duplicate_groups


def missing_value_treatment(df: pd.DataFrame, threshold: float = 20.0):
    """
    Drop columns with more than `threshold` percent missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to clean.
    threshold : float, default=20.0
        Drop any column whose percentage of missing values exceeds this.
    
    Returns
    -------
    df_clean : pd.DataFrame
        A copy of `df` with high-missing‐value columns removed.
    dropped_cols : List[str]
        The names of the columns that were dropped.
    """
    def report_missing(df_inner: pd.DataFrame) -> pd.Series:
        """Return the % missing per column."""
        return df_inner.isnull().mean() * 100

    def drop_cols(df_inner: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Drop the given list of columns."""
        return df_inner.drop(columns=cols)

    # 1) report
    miss_pct = report_missing(df)
    dropped = miss_pct[miss_pct > threshold].index.tolist()

    # 2) drop
    df_clean = drop_cols(df, dropped)

    return df_clean, dropped




def run():

    # near-duplicate variables
    extract_near_duplicate_groups(xxxxxx)

    # categorical variables

    # correlation 

    # outliers

    # normalized standardized 

    # rename the x and y variables



if __name__ == "__main__":
    run()