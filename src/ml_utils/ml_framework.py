#    transcriptid   keydevid  companyid  fiscalyear  fiscalquarter  ...     SP  SalesAcc  SolvencyRatio  StdErr180D WCTurn
# 0        261760  143123704      27685      2011.0            4.0  ...  0.884     1.051          0.598      31.881  7.433
# 1        309189  170941554      27685      2012.0            1.0  ...  0.813    -0.416          0.619      12.547  7.723
# 2        347564  170941641      27685      2012.0            2.0  ...  0.736    -0.192          0.653      17.110  7.616
# 3        373214  222711696     251704      2012.0            1.0  ...  4.305     7.450          0.119     -32.358  6.892
# 4        387960  170941703      27685      2012.0            3.0  ...  0.632    -1.052          0.669      52.331  7.487

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
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
            'xgb': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, 
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
            'lgbm': LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, 
                                num_leaves=31, n_jobs=-1),
            'catboost': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0),
            
            # neural network models
            'nn1': MLPRegressor(**nn_config['nn1_stable']),
            'nn2': MLPRegressor(**nn_config['nn2_deep']),
            'nn3': MLPRegressor(**nn_config['nn3_fast']),
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
            self._log_to_both("Feature importances: feature_importances_")
            return model.feature_importances_
        elif hasattr(model, 'get_score'):
            self._log_to_both("Feature importances: get_score")
            return model.get_score()
        elif hasattr(model, 'coef_'):
            self._log_to_both("Feature importances: coef_")
            coef = model.coef_
            return coef[0] if coef.ndim > 1 else coef
        return None

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all regression metrics for given predictions."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def _process_cv_results(self, cv_results: Dict[str, np.ndarray], scoring: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Process cross-validation results for all metrics."""
        model_metrics = {}
        for metric in scoring:
            scores_array = cv_results[f'test_{metric}']
            # Convert negative scores back to positive for MSE, RMSE, and MAE
            if metric in ['mse', 'rmse', 'mae']:
                scores_array = -scores_array
            
            model_metrics[metric] = {
                'mean': scores_array.mean(),
                'std': scores_array.std(),
                'scores': scores_array
            }
        return model_metrics

    def _log_model_metrics(self, model_metrics: Dict[str, Dict[str, float]], model_name: str) -> None:
        """Log model metrics in a consistent format."""
        self._log_to_both(f"Model results:", model_name)
        for metric_name, metric_data in model_metrics.items():
            self._log_to_both(
                f"- {metric_name.upper()}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}",
                model_name
            )
            self._log_to_both(f"- Individual fold scores: {metric_data['scores']}", model_name)

    def _save_feature_importance(self, model: Any, model_name: str) -> None:
        """Calculate and save feature importance for a model."""
        importance = self._get_feature_importance(model)
        if importance is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
            
            # Sort by absolute importance
            importance_df['abs_importance'] = np.abs(importance_df['importance'])
            importance_df = importance_df.sort_values('abs_importance', ascending=False).drop('abs_importance', axis=1)

            # merge on feature categories 
            affactor_ref = pd.read_csv('input_data/affactor.csv')[['factorabbreviation', 'styleid']]
            # add x_ to the beginning of the feature name   
            affactor_ref['feature'] = 'x_' + affactor_ref['factorabbreviation']
            importance_df = importance_df.merge(affactor_ref[['feature', 'styleid']], on='feature', how='left')
            importance_df = importance_df.rename(columns={'styleid': 'feature_category'})

            # Save to CSV
            importance_file = os.path.join(self.models_dir, f"{model_name}_feature_importance.csv")
            importance_df.to_csv(importance_file, index=False)

    def train_models(self, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Train multiple models and evaluate their performance using multiple metrics."""
        scores = {}
        self._log_to_both(f"Starting model training with {cv}-fold cross-validation")
        
        # Define scoring metrics
        scoring = {
            'r2': 'r2',
            'mse': 'neg_mean_squared_error',
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error'
        }
        
        for name, model in self.models.items():
            self._log_to_both(f"\nTraining {name} model...", name)
            
            # Perform cross-validation with multiple metrics at once
            cv_results = cross_validate(
                model, 
                self.X_train, 
                self.y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=False
            )
            
            # Process results
            model_metrics = self._process_cv_results(cv_results, scoring)
            scores[name] = model_metrics
            
            # Log results
            self._log_model_metrics(model_metrics, name)
            
            # Train final model for feature importance
            final_model = model.__class__(**model.get_params())
            final_model.fit(self.X_train, self.y_train)
            
            # Save feature importance
            self._save_feature_importance(final_model, name)
            
        # Create consolidated score files
        all_scores = []
        all_stds = []
        
        for model_name, model_metrics in scores.items():
            # Mean scores
            mean_scores = {
                'model': model_name,
                'r2': model_metrics['r2']['mean'],
                'mse': model_metrics['mse']['mean'],
                'rmse': model_metrics['rmse']['mean'],
                'mae': model_metrics['mae']['mean']
            }
            all_scores.append(mean_scores)
            
            # Standard deviations
            std_scores = {
                'model': model_name,
                'r2': model_metrics['r2']['std'],
                'mse': model_metrics['mse']['std'],
                'rmse': model_metrics['rmse']['std'],
                'mae': model_metrics['mae']['std']
            }
            all_stds.append(std_scores)
        
        # Save consolidated files
        pd.DataFrame(all_scores).to_csv(os.path.join(self.models_dir, 'cv_scores.csv'), index=False)
        pd.DataFrame(all_stds).to_csv(os.path.join(self.models_dir, 'cv_scores_std.csv'), index=False)
                
        return scores
    
    def evaluate_model(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on the test set."""
        if not self.models:
            raise ValueError("No models trained. Run train_models() first.")
        
        self._log_to_both("\nEvaluating all models on test set...")
        test_metrics = {}
        
        for name, model in self.models.items():
            self._log_to_both(f"\nEvaluating {name} model...", name)
            
            # Train model on full training set
            final_model = model.__class__(**model.get_params())
            final_model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = final_model.predict(self.X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred)
            test_metrics[name] = metrics
            
            # Log metrics
            self._log_to_both("Test set metrics:", name)
            self._log_metrics(metrics)
        
        # Save test metrics to CSV
        test_scores = []
        for model_name, metrics in test_metrics.items():
            test_scores.append({
                'model': model_name,
                'r2': metrics['r2'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae']
            })
        
        # Save to CSV
        pd.DataFrame(test_scores).to_csv(
            os.path.join(self.models_dir, 'oos_test_scores.csv'), 
            index=False
        )
        
        return test_metrics