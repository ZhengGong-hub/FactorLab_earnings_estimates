# FactorLab Earnings Estimates

This project processes and analyzes earnings estimates data using various preprocessing utilities and machine learning techniques.

## Project Structure

```
.
├── src/                          # Source code directory
│   ├── main.py                  # Main entry point of the application
│   ├── preprocess.py            # Main preprocessing pipeline
│   ├── ml_run.py               # Machine learning execution script
│   ├── ml_utils/               # Machine learning utilities
│   │   ├── ml_framework.py     # Core ML framework implementation
│   │   └── nn.yaml            # Neural network configurations
│   └── preprocess_utils/        # Preprocessing utility functions
│       ├── near_duplicates_var.py    # Handle near-duplicate variables
│       ├── drop_fuzzy_var.py         # Drop fuzzy matching variables
│       └── missing_var_treat.py      # Handle missing values
```

## Code Structure Details

### Main Components

1. **main.py**
   - Entry point of the application
   - Orchestrates the overall data processing pipeline

3. **ml_run.py**
   - Handles machine learning model execution
   - (Currently empty, prepared for future implementation)

### Machine Learning Framework (`ml_utils/`)

1. **ml_framework.py**
   - Core machine learning implementation
   - Features:
     - Multiple model support (Linear, Tree-based, Neural Networks)
     - Cross-validation with multiple metrics
     - Feature importance analysis
     - Model evaluation and comparison
     - Comprehensive logging
   - Supported Models:
     - Linear: LinearRegression, Ridge, Lasso, ElasticNet
     - Tree-based: XGBoost, LightGBM, CatBoost
     - Neural Networks: MLPRegressor (3 configurations)
   - Metrics:
     - R² Score
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
   - Output Files:
     - Cross-validation scores (cv_scores.csv)
     - Cross-validation standard deviations (cv_scores_std.csv)
     - Out-of-sample test scores (oos_test_scores.csv)
     - Feature importance per model (*_feature_importance.csv)

2. **nn.yaml**
   - Neural network configuration file
   - Contains three model configurations:
     - nn1_stable: Stable configuration
     - nn2_deep: Deep network configuration
     - nn3_fast: Fast training configuration

### Preprocessing Utilities (`preprocess_utils/`)

1. **near_duplicates_var.py**
   - Purpose: Identifies and groups "near-duplicate" columns in datasets
   - Key Features:
     - Detects columns with similar base names but different suffixes
     - Supports customizable suffix patterns
     - Handles both numeric and word-based suffixes
   - Example patterns:
     - Numeric suffixes (e.g., column_1, column_2)
     - Word suffixes (e.g., column_normalized, column_adjusted)
   - Size: 2.9KB (86 lines)

2. **drop_fuzzy_var.py**
   - Purpose: Handles fuzzy matching for variable removal
   - Size: 562B (21 lines)

3. **missing_var_treat.py**
   - Purpose: Implements missing value treatment strategies
   - Size: 1.1KB (39 lines)

## Usage

To run the main application:

```bash
uv run src/main.py
```
and
```bash
uv run src/ml_run.py
```