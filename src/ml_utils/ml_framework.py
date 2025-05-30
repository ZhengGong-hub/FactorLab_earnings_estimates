#    transcriptid   keydevid  companyid  fiscalyear  fiscalquarter  ...     SP  SalesAcc  SolvencyRatio  StdErr180D WCTurn
# 0        261760  143123704      27685      2011.0            4.0  ...  0.884     1.051          0.598      31.881  7.433
# 1        309189  170941554      27685      2012.0            1.0  ...  0.813    -0.416          0.619      12.547  7.723
# 2        347564  170941641      27685      2012.0            2.0  ...  0.736    -0.192          0.653      17.110  7.616
# 3        373214  222711696     251704      2012.0            1.0  ...  4.305     7.450          0.119     -32.358  6.892
# 4        387960  170941703      27685      2012.0            3.0  ...  0.632    -1.052          0.669      52.331  7.487

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
from typing import List, Dict, Union, Optional
import logging
import os
from datetime import datetime

# internal imports
from logger import setup_logger

# import yaml
import yaml

# load nn.yaml
with open('src/ml_utils/nn.yaml', 'r') as file:
    nn_config = yaml.safe_load(file)

# setup logger
logger = setup_logger(__name__)


class MLFramework:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ML framework with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing features and target
        """
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {
            # linear models
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'enet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000),

            # tree based models
            'hgb': HistGradientBoostingRegressor(),
            'xgb': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
            'lgbm': LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, num_leaves=31, n_jobs=-1),
            'catboost': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0),
            'et': ExtraTreesRegressor(n_estimators=500, n_jobs=-1),

            # neural network models
            'nn1': MLPRegressor(**nn_config['nn1_stable']),
            'nn2': MLPRegressor(**nn_config['nn2_deep']),
            'nn3': MLPRegressor(**nn_config['nn3_fast']),
        }
        self.best_model = None
        self.best_score = float('-inf')
        
        # Create output directories
        self.output_dir = 'output_data/ml_run'
        self.models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize main log file with timestamp
        self.main_log_file = os.path.join(self.output_dir, 'ml_run.txt')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write_to_file(self.main_log_file, f"ML Framework Run - {timestamp}\n", mode='w')
        
        # Initialize model log files with timestamps
        for model_name in self.models.keys():
            model_log_file = os.path.join(self.models_dir, f"{model_name}.txt")
            self._write_to_file(model_log_file, f"Model: {model_name} - {timestamp}\n", mode='w')
        
    def _write_to_file(self, filepath: str, content: str, mode: str = 'a') -> None:
        """
        Write content to a file.
        
        Args:
            filepath (str): Path to the file
            content (str): Content to write
            mode (str): File open mode ('a' for append, 'w' for write)
        """
        with open(filepath, mode) as f:
            f.write(content + '\n')
            
    def _log_to_both(self, message: str, model_name: Optional[str] = None) -> None:
        """
        Log message to both console and file.
        
        Args:
            message (str): Message to log
            model_name (Optional[str]): Name of the model for individual model logs
        """
        logger.info(message)
        self._write_to_file(self.main_log_file, message)
        
        if model_name:
            model_log_file = os.path.join(self.models_dir, f"{model_name}.txt")
            self._write_to_file(model_log_file, message)

    def prepare_data(self, 
                    feature_cols: List[str], 
                    target_col: str,
                    test_size: float = 0.2,
                    random_state: int = 42) -> None:
        """
        Prepare the data for training by selecting features and target.
        
        Args:
            feature_cols (List[str]): List of column names to use as features
            target_col (str): Column name to use as target
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """

        # y should have no missing values
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[target_col])
        dropped_rows = initial_rows - len(self.df)
        
        self._log_to_both("Data preparation started:")
        self._log_to_both(f"- Initial dataset size: {initial_rows} rows")
        self._log_to_both(f"- Dropped {dropped_rows} rows with missing target values")
        self._log_to_both(f"- Final dataset size: {len(self.df)} rows")

        self.X = self.df[feature_cols]
        self.y = self.df[target_col]

        # For mean imputation
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)
        self._log_to_both("- Applied mean imputation to features")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self._log_to_both("Data split and scaling completed:")
        self._log_to_both(f"- Training set size: {len(self.X_train)} samples")
        self._log_to_both(f"- Test set size: {len(self.X_test)} samples")
        self._log_to_both(f"- Number of features: {len(feature_cols)}")
        self._log_to_both(f"- Target variable: {target_col}")
        
    def train_models(self, cv: int = 5) -> Dict[str, float]:
        """
        Train multiple models and evaluate their performance.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Dictionary of model names and their scores
        """
        scores = {}
        
        self._log_to_both(f"Starting model training with {cv}-fold cross-validation")
        
        for name, model in self.models.items():
            self._log_to_both(f"\nTraining {name} model...", name)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            scores[name] = mean_score
            
            self._log_to_both(f"{name} model results:", name)
            self._log_to_both(f"- Mean R2 score: {mean_score:.4f} ± {std_score:.4f}", name)
            self._log_to_both(f"- Individual fold scores: {cv_scores}", name)
            
            # Update best model if current model performs better
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model
                
        self._log_to_both(f"\nTraining completed. Best model: {self.best_model.__class__.__name__} with R2 score: {self.best_score:.4f}")
        return scores
    
    def train_best_model(self) -> None:
        """Train the best performing model on the full training set."""
        if self.best_model is None:
            raise ValueError("No best model selected. Run train_models() first.")
            
        self._log_to_both(f"\nTraining best model ({self.best_model.__class__.__name__}) on full training set...")
        self.best_model.fit(self.X_train, self.y_train)
        
        # Log model parameters if available
        if hasattr(self.best_model, 'get_params'):
            params = self.best_model.get_params()
            self._log_to_both("Model parameters:")
            for param, value in params.items():
                self._log_to_both(f"- {param}: {value}")
                
        self._log_to_both("Best model training completed successfully")
        
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the best model on the test set.
        
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run train_best_model() first.")
            
        self._log_to_both("\nEvaluating model on test set...")
        y_pred = self.best_model.predict(self.X_test)
        
        metrics = {
            'mse': mean_squared_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'r2': r2_score(self.y_test, y_pred),
            'explained_variance': explained_variance_score(self.y_test, y_pred)
        }
        
        self._log_to_both("Model evaluation metrics:")
        self._log_to_both(f"- Mean Squared Error (MSE): {metrics['mse']:.4f}")
        self._log_to_both(f"- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        self._log_to_both(f"- Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        self._log_to_both(f"- R-squared (R2): {metrics['r2']:.4f}")
        self._log_to_both(f"- Explained Variance: {metrics['explained_variance']:.4f}")
        
        # Log prediction statistics
        self._log_to_both("\nPrediction statistics:")
        self._log_to_both(f"- Mean prediction: {np.mean(y_pred):.4f}")
        self._log_to_both(f"- Std prediction: {np.std(y_pred):.4f}")
        self._log_to_both(f"- Min prediction: {np.min(y_pred):.4f}")
        self._log_to_both(f"- Max prediction: {np.max(y_pred):.4f}")
        
        # Log feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            top_features = np.argsort(importances)[-10:]  # Top 10 features
            self._log_to_both("\nTop 10 most important features:")
            for idx in top_features:
                self._log_to_both(f"- Feature {idx}: {importances[idx]:.4f}")
                
        return metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (Union[pd.DataFrame, np.ndarray]): Features to predict on
            
        Returns:
            np.ndarray: Model predictions
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run train_best_model() first.")
            
        self._log_to_both("\nMaking predictions...")
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        predictions = self.best_model.predict(X_scaled)
        
        self._log_to_both(f"Prediction statistics:")
        self._log_to_both(f"- Number of predictions: {len(predictions)}")
        self._log_to_both(f"- Mean prediction: {np.mean(predictions):.4f}")
        self._log_to_both(f"- Std prediction: {np.std(predictions):.4f}")
        self._log_to_both(f"- Min prediction: {np.min(predictions):.4f}")
        self._log_to_both(f"- Max prediction: {np.max(predictions):.4f}")
        
        return predictions
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            path (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run train_best_model() first.")
            
        self._log_to_both(f"\nSaving model to {path}...")
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
        self._log_to_both("Model saved successfully")
        
    @classmethod
    def load_model(cls, path: str) -> 'MLFramework':
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            MLFramework: Instance with loaded model
        """
        logger.info(f"\nLoading model from {path}...")
        model_data = joblib.load(path)
        instance = cls(pd.DataFrame())  # Create empty instance
        instance.best_model = model_data['model']
        instance.scaler = model_data['scaler']
        logger.info(f"Model loaded successfully: {instance.best_model.__class__.__name__}")
        return instance
