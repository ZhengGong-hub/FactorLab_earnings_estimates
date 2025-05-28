import pandas as pd

df = pd.read_csv("input_data/universe_with_affactor.csv", index_col=0)
print(df.head())