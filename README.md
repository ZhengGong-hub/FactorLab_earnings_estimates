# FactorLab Earnings Estimates

This project processes and analyzes earnings estimates data using various preprocessing utilities and machine learning techniques.

## Project Structure

```
.
├── input_data/                   # Input data directory
│   └── universe_with_affactor.parquet  # Raw input data
├── output_data/                  # Output directory for processed data
│   ├── cleaned_data.csv         # Final preprocessed dataset
│   ├── plots/                   # Generated plots and visualizations
│   └── stats/                   # Statistical analysis outputs
├── src/                         # Source code directory
│   ├── main.py                 # Main entry point of the application
│   ├── preprocess.py           # Main preprocessing pipeline
│   ├── ml_run.py              # Machine learning execution script
│   └── preprocess_utils/       # Preprocessing utility functions
│       ├── near_duplicates_var.py    # Handle near-duplicate variables
│       ├── drop_fuzzy_var.py         # Drop fuzzy matching variables
│       ├── missing_var_treat.py      # Handle missing values
│       ├── standard_var.py           # Standardize variables
│       └── drop_high_corr.py         # Handle correlated variables
```

## Data Processing Pipeline

### Input Data Requirements
- File: `input_data/universe_with_affactor.parquet`
- Format: Parquet file containing financial data with the following key variables:
  - EPS-related variables (actual, diff, surprise, etc.)
  - Revenue-related variables (actual, diff, guidance, etc.)
  - Company identifiers (companyid)
  - Calendar information (calendaryear)

### Preprocessing Steps
1. Missing Value Treatment (50% threshold)
2. Duplicate Detection
3. Fuzzy Variable Removal
4. Categorical Encoding
5. Correlation Analysis (0.8 threshold)
6. Outlier Treatment
7. Variable Standardization
8. Variable Classification (x_, y_ prefixing)

### Output Data
- File: `output_data/cleaned_data.csv`
- Format: CSV file with processed variables:
  - Features (x_ prefix): Standardized numeric variables
  - Targets (y_ prefix): EPS and revenue metrics
  - Categorical (dummy_ prefix): Encoded categorical variables
  - Special variables: quarter_factor, companyid

## Usage

### Prerequisites
```bash
# Install dependencies
pip install pandas numpy
```

### Running the Pipeline
```bash
# Create necessary directories
mkdir -p input_data output_data/plots output_data/stats

# Place your input data
cp path/to/your/data.parquet input_data/universe_with_affactor.parquet

# Run the preprocessing pipeline
python src/main.py

# The cleaned data will be available at:
# output_data/cleaned_data.csv
```

### Output Variables

Key variables in the cleaned dataset:

1. Target Variables (y_ prefix):
   - y_EPS_actual, y_EPSDiff, y_EPS_surprise
   - y_EPSNormalized_actual, y_EPSNormalized_diff, y_EPSNormalized_surprise
   - y_revenue_actual, y_revenueDiff, y_revenue_surprise

2. Feature Variables (x_ prefix):
   - x_EPS_count, x_EPS_std, x_EPS_guidance_high, x_EPS_guidance_low
   - x_EPSNormalized_count, x_EPSNormalized_std
   - x_revenue_count, x_revenue_std, x_revenue_guidance_high, x_revenue_guidance_low
   - x_quarter_factor

3. Identifiers:
   - companyid


