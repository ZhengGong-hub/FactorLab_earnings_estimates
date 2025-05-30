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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import joblib
from typing import List, Dict, Union, Optional

# internal imports
from logger import setup_logger

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
            # 'linear': LinearRegression(),
            # 'ridge': Ridge(),
            # 'lasso': Lasso(),
            'rf': RandomForestRegressor(n_jobs=-1),
            'hgb': HistGradientBoostingRegressor(),
            # 'mlp': MLPRegressor(max_iter=500)
        }
        self.best_model = None
        self.best_score = float('-inf')
        
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
        self.df = self.df.dropna(subset=[target_col])

        self.X = self.df[feature_cols]
        self.y = self.df[target_col]

        # For mean imputation
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Data prepared with {len(feature_cols)} features and target {target_col}")
        
    def train_models(self, cv: int = 5) -> Dict[str, float]:
        """
        Train multiple models and evaluate their performance.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Dictionary of model names and their scores
        """
        scores = {}
        
        for name, model in self.models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2')
            mean_score = cv_scores.mean()
            scores[name] = mean_score
            
            # Update best model if current model performs better
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model
                
            logger.info(f"{name} model - Mean R2 score: {mean_score:.4f}")
            
        return scores
    
    def train_best_model(self) -> None:
        """Train the best performing model on the full training set."""
        if self.best_model is None:
            raise ValueError("No best model selected. Run train_models() first.")
            
        self.best_model.fit(self.X_train, self.y_train)
        logger.info("Best model trained successfully")
        
    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the best model on the test set.
        
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run train_best_model() first.")
            
        y_pred = self.best_model.predict(self.X_test)
        
        metrics = {
            'mse': mean_squared_error(self.y_test, y_pred),
            'r2': r2_score(self.y_test, y_pred)
        }
        
        logger.info(f"Model evaluation - MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}")
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
            
        # Scale the input features
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            path (str): Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run train_best_model() first.")
            
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
    @classmethod
    def load_model(cls, path: str) -> 'MLFramework':
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            MLFramework: Instance with loaded model
        """
        model_data = joblib.load(path)
        instance = cls(pd.DataFrame())  # Create empty instance
        instance.best_model = model_data['model']
        instance.scaler = model_data['scaler']
        return instance
