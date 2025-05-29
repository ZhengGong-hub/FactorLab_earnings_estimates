import pandas as pd
import numpy as np
from pathlib import Path
import preprocess as pp


# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    df = load_data('input_data/universe_with_affactor.csv')

    # drop the columns with more than 20% missing values
    df_clean0 = pp.missing_value_treatment(df, 20)[0]

    # check the duplicate columns
    duplicate_groups = pp.extract_near_duplicate_groups(df_clean0.columns.tolist())

    # drop the fuzzy variables
    df_clean1 = pp.drop_fuzzy_variables(df_clean0)


    # define and change into categorical variables
    df_clean2 = pp.encode_categorical_variables(df_clean1)

    # check the correlation
    df_clean3 = pp.check_correlation(df_clean2)

    # winsorize the outliers in outcome variables   
    df_clean4 = pp.winsorize_financial_variables(df_clean3)
    print(df_clean3)
    print(df_clean4)
    # check the normalized and standardized variables


    # check the missing values
