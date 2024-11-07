import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
import logging
import shutil
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import sys
from pathlib import Path

class PriceSpikePredictorModel:
    def __init__(self):
        self.logger = self._setup_logging()
        self.model = None
        self.scaler = RobustScaler()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('PriceSpikePredictor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features with enhanced transformations and scaling"""
        self.logger.info("Starting data cleaning and preprocessing...")
        
        # Create a copy to avoid modifying original data
        X_clean = X.copy()
        
        # Log transform volume features
        volume_cols = [col for col in X_clean.columns if 'volume' in col.lower()]
        for col in volume_cols:
            X_clean[col] = np.log1p(X_clean[col])
            self.logger.info(f"Log transformed {col}")
        
        # Create bins for buy_sell_ratio
        if 'buy_sell_ratio_15m' in X_clean.columns:
            X_clean['buy_sell_ratio_bins'] = pd.qcut(X_clean['buy_sell_ratio_15m'], 
                                                    q=10, 
                                                    labels=False, 
                                                    duplicates='drop')
            X_clean.drop('buy_sell_ratio_15m', axis=1, inplace=True)
            self.logger.info("Created bins for buy_sell_ratio_15m")
        
        # Handle missing values
        for col in X_clean.columns:
            missing_count = X_clean[col].isna().sum()
            if missing_count > 0:
                self.logger.info(f"Filling {missing_count} missing values in {col}")
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        # Handle infinite values
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Calculate robust statistics for outlier handling
        for col in X_clean.columns:
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Cap outliers
            X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Log transform target and handle outliers
        y_clean = y.copy()
        y_q3 = y_clean.quantile(0.75)
        y_iqr = y_q3 - y_clean.quantile(0.25)
        y_cap = y_q3 + 3 * y_iqr
        y_clean = y_clean.clip(upper=y_cap)
        y_log = np.log1p(y_clean)
        
        return X_scaled, y_log

    def _calculate_sample_weights(self, y: pd.Series) -> np.ndarray:
        """Calculate sample weights based on spike magnitude"""
        weights = np.log1p(y)
        weights = 1 + 4 * (weights - weights.min()) / (weights.max() - weights.min())
        return weights

    def _create_stratified_folds(self, y: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified folds based on spike magnitude bins"""
        bins = [0, 7, 15, 30, 100, float('inf')]
        y_binned = pd.cut(y, bins, labels=False)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return [(train_idx, val_idx) for train_idx, val_idx in kf.split(y_binned)]

    def _get_xgboost_params(self) -> Dict:
        """Get optimized XGBoost parameters with proper early stopping configuration"""
        return {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'early_stopping_rounds': 100,
            'verbosity': 1
        }

    def train_model(self, X: pd.DataFrame, y: pd.Series, run_dir: str) -> List[Dict]:
        """Train model with fixed feature importance tracking and XGBoost configuration"""
        self.logger.info(f"Starting data preprocessing. Initial shape: {X.shape}")
        
        # Preprocess data
        X_scaled, y_log = self._preprocess_data(X, y)
        
        # Create stratified folds based on target magnitude
        n_splits = 5
        y_bins = pd.qcut(y, q=5, labels=False)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = []
        feature_importances = []
        
        # Set up XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'verbosity': 1
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_bins), 1):
            self.logger.info(f"\nFold {fold} - Training size: {len(train_idx)}, Validation size: {len(val_idx)}")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_log[train_idx], y_log[val_idx]
            
            # Create sample weights based on target magnitude
            weights = np.log1p(y.iloc[train_idx])
            weights = 1 + 4 * (weights - weights.min()) / (weights.max() - weights.min())
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Set up early stopping
            callbacks = [
                xgb.callback.EarlyStopping(
                    rounds=100,
                    metric_name='rmse',
                    save_best=True
                )
            ]
            
            # Train model
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                callbacks=callbacks
            )
            
            # Make predictions
            y_pred = np.expm1(model.predict(dval))
            y_true = y.iloc[val_idx]
            
            # Calculate metrics
            fold_results = self._calculate_metrics(y_true, y_pred)
            fold_results['fold'] = fold
            cv_results.append(fold_results)
            
            # Save feature importance for this fold
            importance_dict = model.get_score(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values()),
                'fold': fold
            })
            feature_importances.append(importance_df)
            
            # Save fold predictions and plots
            self._save_fold_predictions(fold, y_true, y_pred, run_dir)
            self._plot_prediction_performance(y_true, y_pred, run_dir, fold)
            
            if fold == n_splits:
                self.model = model
        
        # Calculate and save average feature importance
        all_importances = pd.concat(feature_importances)
        avg_importance = all_importances.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        self._save_feature_importance(avg_importance, run_dir)
        self._save_model_results(cv_results, run_dir)
        
        return cv_results

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics with focus on spike detection"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Calculate metrics for different spike thresholds
        for threshold in [5, 10, 20, 50, 100]:
            true_spikes = y_true >= threshold
            pred_spikes = y_pred >= threshold
            
            metrics[f'accuracy_{threshold}x'] = accuracy_score(true_spikes, pred_spikes)
            metrics[f'precision_{threshold}x'] = precision_score(true_spikes, pred_spikes, zero_division=0)
            metrics[f'recall_{threshold}x'] = recall_score(true_spikes, pred_spikes, zero_division=0)
            metrics[f'f1_{threshold}x'] = f1_score(true_spikes, pred_spikes, zero_division=0)
        
        # Calculate rank correlation
        metrics['spearman_corr'] = spearmanr(y_true, y_pred)[0]
        
        return metrics
    
    def _save_fold_predictions(self, fold: int, y_true: np.ndarray, y_pred: np.ndarray, run_dir: str):
        """Save predictions for each fold"""
        predictions_dir = os.path.join(run_dir, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        pd.DataFrame({
            'true_value': y_true,
            'predicted_value': y_pred
        }).to_csv(os.path.join(predictions_dir, f'fold_{fold}_predictions.csv'), index=False)

    def _save_feature_importance(self, importance_df: pd.DataFrame, run_dir: str):
        """Save feature importance analysis"""
        importance_dir = os.path.join(run_dir, 'feature_importance')
        os.makedirs(importance_dir, exist_ok=True)
        
        # Save to CSV
        importance_df.to_csv(os.path.join(importance_dir, 'feature_importance.csv'), index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(importance_dir, 'feature_importance.png'))
        plt.close()

    def _save_model_results(self, cv_results: List[Dict], run_dir: str):
        """Save model results and visualizations"""
        model_dir = os.path.join(run_dir, 'model_results')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save cross-validation results
        with open(os.path.join(model_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        # Calculate and save average metrics
        avg_metrics = pd.DataFrame(cv_results).mean().to_dict()
        with open(os.path.join(model_dir, 'average_metrics.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        
        # Save model
        if self.model is not None:
            self.model.save_model(os.path.join(model_dir, 'final_model.json'))
            # Save scaler
            import joblib
            joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X_scaled = self.scaler.transform(X)
        return np.expm1(self.model.predict(X_scaled))

    def _plot_prediction_performance(self, y_true: np.ndarray, y_pred: np.ndarray, save_dir: str, fold: Optional[int] = None):
        """Create visualization of prediction performance"""
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.5)
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([0, max_val], [0, max_val], 'r--')
        ax1.set_xlabel('Actual Spike Magnitude')
        ax1.set_ylabel('Predicted Spike Magnitude')
        ax1.set_title('Actual vs Predicted Spike Magnitudes')
        
        # Plot 2: Error Distribution
        errors = y_pred - y_true
        ax2.hist(errors, bins=50)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Prediction Errors')
        
        plt.tight_layout()
        filename = f'prediction_performance_fold_{fold}.png' if fold is not None else 'prediction_performance.png'
        plt.savefig(os.path.join(plots_dir, filename))
        plt.close()

def find_latest_run_dir(base_dir: str) -> str:
    """Find the most recent run directory with complete data"""
    runs_dir = os.path.join(base_dir, "data", "runs")
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Runs directory not found at {runs_dir}")
    
    # Get all run directories sorted by name (which includes timestamp)
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {runs_dir}")
    
    run_dirs.sort(reverse=True)
    
    for run_dir in run_dirs:
        full_path = os.path.join(runs_dir, run_dir)
        features_path = os.path.join(full_path, 'features', 'selected_features.csv')
        if os.path.exists(features_path):
            return full_path
            
    raise FileNotFoundError("No valid run directory with features found")


def load_data(latest_run_dir: str, new_run_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess data with enhanced validation"""
    # Source paths
    src_features_dir = os.path.join(latest_run_dir, 'features')
    src_features_path = os.path.join(src_features_dir, 'selected_features.csv')
    
    print(f"Looking for features file at: {src_features_path}")
    
    if not os.path.exists(src_features_path):
        raise FileNotFoundError(f"Features file not found at {src_features_path}")
    
    # Load the data
    data = pd.read_csv(src_features_path)
    
    # Create destination directory
    dest_features_dir = os.path.join(new_run_dir, 'features')
    os.makedirs(dest_features_dir, exist_ok=True)
    
    # Preprocess the data
    target_col = 'future_price_increase'
    
    # Remove any string or date columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)
    
    # Select numeric columns
    data_processed = data[numeric_cols].copy()
    
    # Log data info before cleaning
    logger = logging.getLogger(__name__)
    logger.info("\nData statistics before cleaning:")
    logger.info(data_processed.describe())
    
    # Check for infinite values
    inf_counts = data_processed.isin([np.inf, -np.inf]).sum()
    if inf_counts.any():
        logger.warning(f"Found infinite values in columns: {inf_counts[inf_counts > 0]}")
    
    # Check for missing values
    na_counts = data_processed.isna().sum()
    if na_counts.any():
        logger.warning(f"Found missing values in columns: {na_counts[na_counts > 0]}")
    
    # Save preprocessing info
    feature_info = {
        'original_columns': data.columns.tolist(),
        'selected_numeric_columns': numeric_cols,
        'removed_columns': [col for col in data.columns if col not in numeric_cols],
        'infinite_value_counts': inf_counts[inf_counts > 0].to_dict(),
        'missing_value_counts': na_counts[na_counts > 0].to_dict()
    }
    
    with open(os.path.join(dest_features_dir, 'feature_preprocessing_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=4)
    
    # Save processed features
    data_processed.to_csv(os.path.join(dest_features_dir, 'processed_features.csv'), index=False)
    
    # Copy other feature files
    for filename in ['raw_features.csv', 'feature_metadata.json', 'feature_patterns.json']:
        src_file = os.path.join(src_features_dir, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dest_features_dir, filename))
    
    # Separate features and target
    y = data_processed[target_col]
    X = data_processed.drop([target_col], axis=1)
    
    # Log preprocessing summary
    logger.info(f"Original feature count: {len(data.columns)}")
    logger.info(f"Numeric feature count: {len(X.columns)}")
    logger.info(f"Removed columns: {feature_info['removed_columns']}")
    logger.info(f"Feature columns: {X.columns.tolist()}")
    
    return X, y

def setup_run_directory() -> Tuple[str, str]:
    """Create new run directory and find latest run directory"""
    # Get the absolute path to the crypto_model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory containing this script
    base_dir = script_dir  # The script is already in the crypto_model directory
    
    # Find latest run directory
    latest_run_dir = find_latest_run_dir(base_dir)
    
    # Create new run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_dir = os.path.join(base_dir, "data", "runs", f"run_{timestamp}_full")
    
    # Create necessary subdirectories
    subdirs = ['model', 'predictions', 'plots', 'feature_importance', 'features']
    for subdir in subdirs:
        os.makedirs(os.path.join(new_run_dir, subdir), exist_ok=True)
    
    return latest_run_dir, new_run_dir

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Print current working directory and script location
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Script location: {os.path.abspath(__file__)}")
        
        # Setup directories
        latest_run_dir, new_run_dir = setup_run_directory()
        logger.info(f"Found latest run directory: {latest_run_dir}")
        logger.info(f"Created new run directory: {new_run_dir}")
        
        # Load data from latest run and copy to new run
        logger.info("Loading and preprocessing data...")
        X, y = load_data(latest_run_dir, new_run_dir)
        
        logger.info(f"Feature columns: {X.columns.tolist()}")
        logger.info("Starting price spike prediction model training...")
        logger.info(f"Total samples: {len(y)}")
        logger.info("\nPrice increase statistics:")
        logger.info(y.describe())
        
        # Initialize and train model
        model = PriceSpikePredictorModel()
        cv_results = model.train_model(X, y, new_run_dir)
        
        logger.info("\nTraining completed. Average metrics across folds:")
        avg_metrics = pd.DataFrame(cv_results).mean()
        logger.info(avg_metrics)
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()