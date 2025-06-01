# Machine Learning Framework

This document provides an overview of the machine learning framework implemented in `src/ml_utils/ml_framework.py`. The framework is designed to handle various machine learning tasks with a focus on regression problems, particularly for financial data analysis.

## Features

### Model Support
The framework supports multiple types of machine learning models:

1. **Linear Models**
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Elastic Net

2. **Tree-Based Models**
   - HistGradientBoostingRegressor
   - XGBoost
   - LightGBM
   - CatBoost

3. **Neural Networks** (configurable via `nn.yaml`)
   - Multiple network architectures available
   - Configurable hyperparameters

### Key Capabilities

- **Data Preprocessing**
  - Automatic handling of missing values
  - Feature scaling using StandardScaler
  - Train-test splitting
  - Cross-validation support

- **Model Training**
  - Automated model training pipeline
  - Cross-validation evaluation
  - Model performance tracking
  - Best model selection

- **Feature Importance Analysis**
  - Cross-validated feature importance calculation
  - Importance ranking and visualization
  - Standard deviation of importance scores

- **Evaluation Metrics**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
  - Explained Variance Score

- **Comprehensive Logging**
  - Detailed training logs
  - Model performance metrics
  - Feature importance analysis
  - Prediction statistics

## Usage

### Basic Usage

```python
from ml_framework import MLFramework
import pandas as pd

# Initialize the framework
ml = MLFramework(df=your_dataframe)

# Prepare the data
feature_cols = ['feature1', 'feature2', ...]
target_col = 'target'
ml.prepare_data(feature_cols=feature_cols, target_col=target_col)

# Train models
scores = ml.train_models(cv=5)

# Evaluate best model
metrics = ml.evaluate_model()

# Make predictions
predictions = ml.predict(new_data)
```

### Configuration

The framework can be configured through:

1. **Model Parameters**: Adjust model hyperparameters in the `_initialize_models` method
2. **Neural Network Config**: Modify `nn.yaml` for neural network architectures
3. **Output Directory**: Specify custom output directory during initialization
4. **Cross-validation**: Adjust the number of CV folds in `train_models`

## Output Structure

The framework generates the following outputs:

```
output_data/ml_run/
├── ml_run.txt           # Main log file
├── models/             # Model-specific outputs
│   ├── model_name_feature_importance.csv
│   └── ...
└── ...
```
