import pandas as pd
import numpy as np
from pathlib import Path
import re

from preprocess_utils.near_duplicates_var import extract_near_duplicate_groups
from preprocess_utils.missing_var_treat import missing_value_treatment
from preprocess_utils.drop_fuzzy_var import drop_fuzzy_variables
from preprocess_utils.categorize_var import encode_categorical_variables
from preprocess_utils.drop_high_corr import drop_high_corr
from preprocess_utils.outlier_check import winsorize_financial_variables
from preprocess_utils.standard_var import standardize_numeric_variables
from preprocess_utils.rename_x_y import rename_variables_xy

def run():

    # missing value treatment
    missing_value_treatment()

    # near-duplicate variables and drop fuzzy variables
    extract_near_duplicate_groups()
    drop_fuzzy_variables()

    # categorical variables
    encode_categorical_variables()

    # drop high correlated variables
    drop_high_corr()

    # winsorize the outliers in outcome variables   
    winsorize_financial_variables()

    # normalized standardized 
    standardize_numeric_variables()

    # rename the x and y variables
    rename_variables_xy()

if __name__ == "__main__":
    run()