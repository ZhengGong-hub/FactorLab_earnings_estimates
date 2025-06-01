import pandas as pd

df = pd.read_parquet('output_data/cleaned_data.parquet')

# print the first 5 rows
print(df.sort_values(by='y_EPS_SUE', ascending=False).head())

# print the last 5 rows
print(df.sort_values(by='y_EPS_SUE', ascending=False).tail())

# print the shape of the dataframe