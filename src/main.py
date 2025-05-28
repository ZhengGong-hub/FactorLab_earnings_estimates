import pandas as pd
import numpy as np
from pathlib import Path
import preprocess as pp
# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)
df = load_data('/home/azureuser/FactorLab_earnings_estimates/input_data/universe_with_affactor.csv')


# drop the columns with more than 20% missing values
df_clean0 = pp.missing_value_treatment(df, 20)[0]

print(df_clean0)

# check the duplicate columns
duplicate_groups = pp.extract_near_duplicate_groups(df_clean0.columns.tolist())
print("Duplicate groups found:")
for base_name, cols in duplicate_groups.items():
    print(f"Base name: {base_name}, Columns: {cols}")
# keep EPS, EPS_et, EPS_normalized, and EPS_normalizedDiff
# keep revenue, revenue_et
# keep BVEV (cz it's adjusted)
# keep AssetTurn_2 (avoid negative values)
# keep ChgATO_2
# keep EbitToAst_2
# keep OCFAst_2
# keep REToAst_2
# keep ROA_2
# keep both IO_Level and IO_Level_AM
# keep both STO and STO_6M



