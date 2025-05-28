import pandas as pd
import numpy as np
from pathlib import Path
import re

from preprocess_utils.near_duplicates_var import extract_near_duplicate_groups
from preprocess_utils.missing_var_treat import missing_value_treatment
from preprocess_utils.drop_fuzzy_var import drop_fuzzy_variables
from preprocess_utils.categorize_var import encode_categorical_variables
from preprocess_utils.corr_check import check_correlation
from preprocess_utils.outlier_check import winsorize_financial_variables

def run():

    # missing value treatment
    missing_value_treatment()

    # near-duplicate variables and drop fuzzy variables
    extract_near_duplicate_groups()
    drop_fuzzy_variables()

    # categorical variables
    encode_categorical_variables()

    # correlation
    check_correlation()
    # winsorize the outliers in outcome variables   
    winsorize_financial_variables()
    # normalized standardized 

    # rename the x and y variables


if __name__ == "__main__":
    run()