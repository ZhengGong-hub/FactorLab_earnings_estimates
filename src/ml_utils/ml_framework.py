#    transcriptid   keydevid  companyid  fiscalyear  fiscalquarter  ...     SP  SalesAcc  SolvencyRatio  StdErr180D WCTurn
# 0        261760  143123704      27685      2011.0            4.0  ...  0.884     1.051          0.598      31.881  7.433
# 1        309189  170941554      27685      2012.0            1.0  ...  0.813    -0.416          0.619      12.547  7.723
# 2        347564  170941641      27685      2012.0            2.0  ...  0.736    -0.192          0.653      17.110  7.616
# 3        373214  222711696     251704      2012.0            1.0  ...  4.305     7.450          0.119     -32.358  6.892
# 4        387960  170941703      27685      2012.0            3.0  ...  0.632    -1.052          0.669      52.331  7.487

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
from typing import List, Dict, Union, Optional, Any
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
    def __init__(self, df: pd.DataFrame, output_dir: str = 'output_data/ml_run'):
        """Initialize the ML framework with a pandas DataFrame."""
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Initialize models with default configurations
        self.models = self._initialize_models()
        self.best_model = None
        self.best_score = float('-inf')
        
        # Setup output directories
        self.output_dir = output_dir
        self.models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize main log file
        self.main_log_file = os.path.join(self.output_dir, 'ml_run.txt')
        self._write_to_file(self.main_log_file, f"ML Framework Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", mode='w')

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize and return dictionary of models with their configurations."""
        return {
            # linear models
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'enet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000),
            # tree based models
            'hgb': HistGradientBoostingRegressor(),
            'xgb': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, 
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
            'lgbm': LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, 
                                num_leaves=31, n_jobs=-1),
            'catboost': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0),
            
            # neural network models
            # 'nn1': MLPRegressor(**nn_config['nn1_stable']),
            # 'nn2': MLPRegressor(**nn_config['nn2_deep']),
            # 'nn3': MLPRegressor(**nn_config['nn3_fast']),
        }
        
    def _write_to_file(self, filepath: str, content: str, mode: str = 'a') -> None:
        """Write content to a file."""
        with open(filepath, mode) as f:
            f.write(content + '\n')
            
    def _log_to_both(self, message: str, model_name: Optional[str] = None) -> None:
        """Log message to both console and file."""
        logger.info(message)
        if model_name:
            message = f"[{model_name}] {message}"
        self._write_to_file(self.main_log_file, message)

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics in a consistent format."""
        for metric, value in metrics.items():
            self._log_to_both(f"{prefix}- {metric.upper()}: {value:.4f}")

    def prepare_data(self, 
                    feature_cols: List[str], 
                    target_col: str,
                    test_size: float = 0.2,
                    random_state: int = 42) -> None:
        """Prepare the data for training by selecting features and target."""
        # Store feature names
        self.feature_names = feature_cols
        
        # Prepare data
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[target_col])
        # drop rows where target_col is infinity
        self.df = self.df[~np.isinf(self.df[target_col])]
        dropped_rows = initial_rows - len(self.df)
        
        self._log_to_both("Data preparation started:")
        self._log_to_both(f"- Initial dataset size: {initial_rows} rows")
        self._log_to_both(f"- Dropped {dropped_rows} rows with missing target values")
        self._log_to_both(f"- Final dataset size: {len(self.df)} rows")

        # Prepare features and target
        self.X = self.df[feature_cols]
        self.y = self.df[target_col]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)
        self._log_to_both("- Applied mean imputation to features")

        # Split and scale data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test) # never fit on test set
        
        self._log_to_both("Data split and scaling completed:")
        self._log_to_both(f"- Training set size: {len(self.X_train)} samples")
        self._log_to_both(f"- Test set size: {len(self.X_test)} samples")
        self._log_to_both(f"- Number of features: {len(feature_cols)}")
        self._log_to_both(f"- Target variable: {target_col}")

    def _get_feature_importance(self, model: Any) -> Optional[np.ndarray]:
        """Get feature importance or coefficients from a model."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            return coef[0] if coef.ndim > 1 else coef
        return None

    def _calculate_fold_importance(self, model: Any, X: np.ndarray, y: pd.Series) -> Optional[np.ndarray]:
        """Calculate feature importance for a single fold."""
        fold_model = model.__class__(**model.get_params())
        fold_model.fit(X, y)
        return self._get_feature_importance(fold_model)

    def calculate_feature_importance(self, model: Any, model_name: str, cv: int = 5) -> None:
        """Calculate average feature importance across CV splits."""
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        feature_importances = []
        
        # Calculate importance for each fold
        for fold, (train_idx, _) in enumerate(kf.split(self.X_train)):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            
            importance = self._calculate_fold_importance(model, X_fold_train, y_fold_train)
            if importance is not None:
                feature_importances.append(importance)
        
        if feature_importances:
            # Calculate statistics
            avg_importances = np.mean(feature_importances, axis=0)
            std_importances = np.std(feature_importances, axis=0)
            
            # Create and save importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': avg_importances,
                'importance_std': std_importances
            })
            
            # Sort by absolute importance
            importance_df['abs_importance'] = np.abs(importance_df['importance_mean'])
            importance_df = importance_df.sort_values('abs_importance', ascending=False).drop('abs_importance', axis=1)
            
            # Save to CSV
            importance_file = os.path.join(self.models_dir, f"{model_name}_feature_importance.csv")
            importance_df.to_csv(importance_file, index=False)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all regression metrics for given predictions."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def train_models(self, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Train multiple models and evaluate their performance using multiple metrics."""
        scores = {}
        self._log_to_both(f"Starting model training with {cv}-fold cross-validation")
        
        # Define scoring metrics
        scoring_metrics = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_rmse': 'neg_root_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        
        for name, model in self.models.items():
            self._log_to_both(f"\nTraining {name} model...", name)
            
            # Initialize metrics dictionary for this model
            model_metrics = {}
            
            # Perform cross-validation for each metric
            for metric_name, metric in scoring_metrics.items():
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=metric)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                # Convert negative scores back to positive for MSE, RMSE, and MAE
                if metric_name.startswith('neg_'):
                    mean_score = -mean_score
                    std_score = std_score
                    metric_name = metric_name[4:]  # Remove 'neg_' prefix
                
                model_metrics[metric_name] = {
                    'mean': mean_score,
                    'std': std_score,
                    'scores': cv_scores
                }
            
            scores[name] = model_metrics
            
            # Log results for each metric
            self._log_to_both(f"Model results:", name)
            for metric_name, metric_data in model_metrics.items():
                self._log_to_both(
                    f"- {metric_name.upper()}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}",
                    name
                )
                self._log_to_both(f"- Individual fold scores: {metric_data['scores']}", name)
            
            # Calculate feature importance
            self.calculate_feature_importance(model, name, cv)
            
            # Update best model based on R² score
            if model_metrics['r2']['mean'] > self.best_score:
                final_model = model.__class__(**model.get_params())
                final_model.fit(self.X_train, self.y_train)
                self.best_score = model_metrics['r2']['mean']
                self.best_model = final_model
                
        return scores
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the best model on the test set."""
        if self.best_model is None:
            raise ValueError("No model trained. Run train_models() first.")
        
        # log who is the best model
        self._log_to_both(f"Best model: {self.best_model}")
            
        self._log_to_both("\nEvaluating model on test set...")
        y_pred = self.best_model.predict(self.X_test)
        
        # Calculate metrics using shared function
        metrics = self._calculate_metrics(self.y_test, y_pred)
        
        self._log_to_both("Model evaluation metrics:")
        self._log_metrics(metrics)
        
        return metrics