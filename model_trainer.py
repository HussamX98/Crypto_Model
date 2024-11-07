import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import traceback
import glob  # Add this line
from typing import Dict, List, Tuple
import json
from datetime import datetime
from pathlib import Path

class SpikePatternPredictor:
    def __init__(self, base_spike_threshold: float = 5.0):
        self.logger = self._setup_logging()
        self.scaler = RobustScaler()
        self.base_spike_threshold = base_spike_threshold
        self.pattern_profiles = {}
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('SpikePredictor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _validate_data(self, X: pd.DataFrame) -> None:
        """Validate input data before processing"""
        # Check for required columns
        required_cols = ['volume_15m', 'price_change_15m', 'rsi_14', 'momentum_15m']
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for all numeric data
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Non-numeric columns found: {non_numeric}")
        
        # Log data shape and columns
        self.logger.info(f"Input data shape: {X.shape}")
        self.logger.info(f"Columns: {X.columns.tolist()}")
        
        # Log basic statistics
        self.logger.info("\nInput data statistics:")
        self.logger.info(X.describe())

    def _log_feature_correlations(self, X: pd.DataFrame, y: pd.Series):
        """Log feature correlations with target"""
        correlations = pd.DataFrame()
        for col in X.columns:
            correlation = np.corrcoef(X[col], y)[0, 1]
            correlations.loc[col, 'correlation'] = correlation
        
        correlations = correlations.sort_values('correlation', ascending=False)
        self.logger.info("\nFeature correlations with target:")
        self.logger.info(correlations)

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features"""
        X_new = X.copy()
        
        # Volume-based features
        if 'volume_15m' in X_new.columns:
            X_new['volume_acceleration'] = X_new['volume_15m'].diff()
            X_new['volume_ma_ratio'] = X_new['volume_15m'] / X_new['volume_15m'].rolling(3).mean()
        
        # Price movement features
        price_cols = [col for col in X_new.columns if 'price_change' in col]
        for col in price_cols:
            X_new[f'{col}_abs'] = np.abs(X_new[col])
            
        # Momentum features
        momentum_cols = [col for col in X_new.columns if 'momentum' in col]
        for col in momentum_cols:
            X_new[f'{col}_abs'] = np.abs(X_new[col])
        
        # Interaction features
        if all(col in X_new.columns for col in ['volume_15m', 'price_change_15m']):
            X_new['vol_price_impact'] = X_new['volume_15m'] * X_new['price_change_15m']
        
        if all(col in X_new.columns for col in ['rsi_14', 'volume_15m']):
            X_new['rsi_vol_impact'] = X_new['rsi_14'] * np.log1p(X_new['volume_15m'])
        
        # Drop any new NaN values from feature engineering
        X_new = X_new.fillna(X_new.mean())
        
        return X_new
                
    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features and create proper binary target"""
        self.logger.info("Starting data cleaning and preprocessing...")
        X_clean = X.copy()
        
        # Analyze price increases
        self.logger.info("\nPrice increase distribution:")
        self.logger.info(y.describe())
        
        # Define spikes more strictly (e.g., above 75th percentile)
        spike_threshold = y.quantile(0.75)  # Adjustable threshold
        y_binary = (y >= spike_threshold).astype(int)
        
        class_counts = np.bincount(y_binary)
        self.logger.info(f"\nClass distribution with {spike_threshold:.2f}x threshold:")
        self.logger.info(f"No Spike (0): {class_counts[0]}")
        self.logger.info(f"Spike (1): {class_counts[1]}")
        self.logger.info(f"Spike Percentage: {(class_counts[1] / len(y_binary)) * 100:.2f}%")
        
        # Handle infinite values
        self.logger.info("\nCleaning features...")
        inf_counts = np.isinf(X_clean).sum()
        if inf_counts.any():
            self.logger.info(f"Found infinite values in columns: \n{inf_counts[inf_counts > 0]}")
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # Log transform volume features
        volume_cols = [col for col in X_clean.columns if 'volume' in col.lower()]
        for col in volume_cols:
            non_zero_min = X_clean[col][X_clean[col] > 0].min()
            X_clean[col] = X_clean[col].clip(lower=non_zero_min)
            X_clean[col] = np.log(X_clean[col])
            self.logger.info(f"Log transformed {col}")
        
        # Handle missing values and outliers for each column
        for col in X_clean.columns:
            # Calculate robust statistics
            median_val = X_clean[col].median()
            q1 = X_clean[col].quantile(0.25)
            q3 = X_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # Replace missing values
            is_missing = X_clean[col].isna().sum()
            if is_missing > 0:
                self.logger.info(f"Replacing {is_missing} missing values in {col}")
                X_clean[col] = X_clean[col].fillna(median_val)
            
            # Cap outliers
            X_clean[col] = X_clean[col].clip(lower_bound, upper_bound)
        
        # Add engineered features
        X_clean = self._engineer_features(X_clean)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Log feature correlations with target
        self._log_feature_correlations(X_clean, y_binary)
        
        return X_scaled, y_binary
    
    
    def _get_xgboost_params(self) -> Dict:
        """Get optimized XGBoost parameters for binary classification"""
        return {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 4,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.2,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': 1.0,  # Will be updated based on class distribution
            'random_state': 42,
            'tree_method': 'hist'
        }
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, run_dir: str) -> Dict:
        """Train binary classification model with class balancing"""
        self.logger.info(f"Starting spike pattern detection training. Shape: {X.shape}")
        
        # Validate input data
        self._validate_data(X)
        
        # Preprocess data
        X_scaled, y_binary = self._preprocess_data(X, y)
        
        # Calculate class weight
        class_counts = np.bincount(y_binary)
        weight_ratio = class_counts[0] / class_counts[1]
        
        # Update XGBoost parameters with class weight
        params = self._get_xgboost_params()
        params['scale_pos_weight'] = weight_ratio
        
        # Create stratified folds
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_binary), 1):
            self.logger.info(f"\nFold {fold} - Training size: {len(train_idx)}")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_binary[train_idx], y_binary[val_idx]
            
            # Create DMatrix with weights
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Train model
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=100
            )
            
            # Get predictions
            y_pred_proba = model.predict(dval)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            fold_results = self._calculate_metrics(y_val, y_pred, y_pred_proba)
            fold_results['fold'] = fold
            cv_results.append(fold_results)
            
            if fold == n_splits:
                self.model = model
        
        # Save results
        self._save_model_results(cv_results, run_dir)
        self._save_pattern_profiles(run_dir)
        self._summarize_patterns(run_dir)
        
        return cv_results
    
    def _analyze_patterns(self, X: pd.DataFrame, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                        fold: int, run_dir: str):
        """Analyze successful predictions to identify common patterns"""
        pattern_dir = os.path.join(run_dir, 'pattern_analysis')
        os.makedirs(pattern_dir, exist_ok=True)
        
        # High confidence correct predictions
        confidence_threshold = 0.8
        high_conf_mask = y_pred_proba >= confidence_threshold
        correct_mask = (y_pred_proba >= 0.5) == y_true
        
        successful_patterns = X[high_conf_mask & correct_mask]
        
        if len(successful_patterns) > 0:
            # Calculate pattern statistics
            pattern_stats = {
                'conditions': {
                    'buy_sell_stats': {
                        'ratio_mean': float(successful_patterns['buy_sell_ratio_15m'].mean()),
                        'ratio_min': float(successful_patterns['buy_sell_ratio_15m'].min()),
                        'imbalance_mean': float(successful_patterns['buy_sell_imbalance_15m'].mean())
                    },
                    'trend_stats': {
                        'strength_mean': float(successful_patterns['trend_strength_15m'].mean()),
                        'strength_min': float(successful_patterns['trend_strength_15m'].min())
                    },
                    'price_movement': {
                        'volatility_mean': float(successful_patterns['volatility_15m'].mean()),
                        'price_change_5m_mean': float(successful_patterns['price_change_5m'].mean()),
                        'price_change_15m_mean': float(successful_patterns['price_change_15m'].mean())
                    }
                },
                'performance': {
                    'total_patterns': len(successful_patterns),
                    'accuracy': float((y_pred_proba[high_conf_mask] >= 0.5) == y_true[high_conf_mask]).mean(),
                    'avg_confidence': float(y_pred_proba[high_conf_mask].mean())
                }
            }
            
            # Save pattern information
            with open(os.path.join(pattern_dir, f'fold_{fold}_patterns.json'), 'w') as f:
                json.dump(pattern_stats, f, indent=4)
            
            # Create visualizations
            self._plot_pattern_distributions(successful_patterns, y_true[high_conf_mask], 
                                        y_pred_proba[high_conf_mask], fold, pattern_dir)
        
    def _plot_pattern_distributions(self, patterns: pd.DataFrame, y_true: np.ndarray, 
                                y_pred: np.ndarray, fold: int, save_dir: str):
        """Create visualizations of successful patterns"""
        # Feature distributions
        plt.figure(figsize=(15, 10))
        important_features = [
            'buy_sell_ratio_15m', 'buy_sell_imbalance_15m', 'trend_strength_15m',
            'volatility_15m', 'price_change_5m', 'price_change_15m'
        ]
        
        for i, feature in enumerate(important_features, 1):
            plt.subplot(2, 3, i)
            sns.histplot(data=patterns, x=feature, bins=30)
            plt.title(f'{feature} Distribution')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'fold_{fold}_feature_distributions.png'))
        plt.close()
        
        # Confidence vs Performance
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, y_true, alpha=0.5)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Outcome')
        plt.title('Prediction Confidence vs Actual Outcomes')
        plt.savefig(os.path.join(save_dir, f'fold_{fold}_confidence_analysis.png'))
        plt.close()

    def _summarize_patterns(self, run_dir: str):
        """Summarize findings across all folds"""
        pattern_dir = os.path.join(run_dir, 'pattern_analysis')
        summary_file = os.path.join(pattern_dir, 'pattern_summary.json')
        
        all_patterns = []
        for file in glob.glob(os.path.join(pattern_dir, 'fold_*_patterns.json')):
            with open(file, 'r') as f:
                all_patterns.append(json.load(f))
        
        if all_patterns:
            summary = {
                'optimal_conditions': {
                    'buy_sell_ratio': {
                        'min': np.mean([p['conditions']['buy_sell_stats']['ratio_min'] for p in all_patterns]),
                        'mean': np.mean([p['conditions']['buy_sell_stats']['ratio_mean'] for p in all_patterns])
                    },
                    'trend_strength': {
                        'min': np.mean([p['conditions']['trend_stats']['strength_min'] for p in all_patterns]),
                        'mean': np.mean([p['conditions']['trend_stats']['strength_mean'] for p in all_patterns])
                    },
                    'price_movement': {
                        'volatility': np.mean([p['conditions']['price_movement']['volatility_mean'] for p in all_patterns]),
                        'short_term_change': np.mean([p['conditions']['price_movement']['price_change_5m_mean'] for p in all_patterns])
                    }
                },
                'performance_metrics': {
                    'average_accuracy': np.mean([p['performance']['accuracy'] for p in all_patterns]),
                    'average_confidence': np.mean([p['performance']['avg_confidence'] for p in all_patterns]),
                    'total_patterns': sum(p['performance']['total_patterns'] for p in all_patterns)
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict:
        """Calculate binary classification metrics"""
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'high_confidence_precision': precision_score(
                y_true[y_pred_proba >= 0.8],
                y_pred[y_pred_proba >= 0.8]
            ) if any(y_pred_proba >= 0.8) else 0
        }
    
    def _save_model_results(self, cv_results: List[Dict], run_dir: str):
        """Save model results and visualizations"""
        results_dir = os.path.join(run_dir, 'model_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save cross-validation results
        with open(os.path.join(results_dir, 'cv_results.json'), 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        # Calculate and save average metrics
        avg_metrics = pd.DataFrame(cv_results).mean().to_dict()
        with open(os.path.join(results_dir, 'average_metrics.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)
        
        # Save model
        if self.model is not None:
            self.model.save_model(os.path.join(results_dir, 'spike_predictor.json'))
            import joblib
            joblib.dump(self.scaler, os.path.join(results_dir, 'scaler.joblib'))


    def _save_pattern_profiles(self, run_dir: str):
        """Save aggregated pattern profiles and visualizations"""
        pattern_dir = os.path.join(run_dir, 'pattern_analysis')
        os.makedirs(pattern_dir, exist_ok=True)
        
        try:
            # Aggregate patterns across folds
            pattern_files = glob.glob(os.path.join(pattern_dir, 'fold_*_patterns.json'))
            all_patterns = []
            
            for file in pattern_files:
                with open(file, 'r') as f:
                    patterns = json.load(f)
                    all_patterns.append(patterns)
            
            if not all_patterns:
                self.logger.warning("No pattern files found to analyze")
                return
            
            # Calculate aggregate statistics
            aggregate_stats = {}
            for pattern in all_patterns:
                for stat_type in ['mean', 'std', 'min', 'max', 'median']:
                    if stat_type not in aggregate_stats:
                        aggregate_stats[stat_type] = {}
                        
                    for feature, value in pattern[stat_type].items():
                        if feature not in aggregate_stats[stat_type]:
                            aggregate_stats[stat_type][feature] = []
                        aggregate_stats[stat_type][feature].append(value)
            
            # Calculate final statistics
            final_profile = {
                'feature_stats': {},
                'frequent_patterns': []
            }
            
            for stat_type, features in aggregate_stats.items():
                final_profile['feature_stats'][stat_type] = {
                    feature: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                    for feature, values in features.items()
                }
            
            # Save aggregate profiles
            with open(os.path.join(pattern_dir, 'aggregate_pattern_profile.json'), 'w') as f:
                json.dump(final_profile, f, indent=4)
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            feature_stats = pd.DataFrame({
                feature: stats['mean']['mean']
                for feature, stats in final_profile['feature_stats'].items()
            }).T
            
            feature_stats = feature_stats.sort_values(ascending=False)
            
            plt.bar(range(len(feature_stats)), feature_stats.values)
            plt.xticks(range(len(feature_stats)), 
                    feature_stats.index, 
                    rotation=45, 
                    ha='right')
            
            plt.title('Average Feature Values in Spike Patterns')
            plt.tight_layout()
            plt.savefig(os.path.join(pattern_dir, 'feature_pattern_profile.png'))
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error in saving pattern profiles: {str(e)}")
            self.logger.error(traceback.format_exc())

    def predict_pattern(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict probability of spike and identify matching patterns"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess features
        X_clean = X.copy()
        self._add_interaction_features(X_clean)
        X_scaled = self.scaler.transform(X_clean)
        
        # Get predictions
        dmatrix = xgb.DMatrix(X_scaled)
        spike_probabilities = self.model.predict(dmatrix)
        
        return spike_probabilities
    




def find_latest_run_dir(base_dir: str) -> str:
    """Find the most recent run directory with complete data"""
    runs_dir = os.path.join(base_dir, "data", "runs")
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Runs directory not found at {runs_dir}")
    
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
    run_dirs.sort(reverse=True)
    
    for run_dir in run_dirs:
        full_path = os.path.join(runs_dir, run_dir)
        features_path = os.path.join(full_path, 'features', 'selected_features.csv')
        if os.path.exists(features_path):
            return full_path
            
    raise FileNotFoundError("No valid run directory with features found")

def setup_run_directory() -> Tuple[str, str]:
    """Create new run directory and find latest run directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    
    latest_run_dir = find_latest_run_dir(base_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_dir = os.path.join(base_dir, "data", "runs", f"run_{timestamp}_full")
    
    subdirs = ['model_results', 'predictions', 'pattern_analysis', 'features']
    for subdir in subdirs:
        os.makedirs(os.path.join(new_run_dir, subdir), exist_ok=True)
    
    return latest_run_dir, new_run_dir

def load_data(latest_run_dir: str, new_run_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from the latest run and copy to new run directory"""
    src_features_dir = os.path.join(latest_run_dir, 'features')
    src_features_path = os.path.join(src_features_dir, 'selected_features.csv')
    
    print(f"Loading data from: {src_features_path}")
    
    if not os.path.exists(src_features_path):
        raise FileNotFoundError(f"Features file not found at {src_features_path}")
    
    data = pd.read_csv(src_features_path)
    
    # Create destination directory and copy files
    dest_features_dir = os.path.join(new_run_dir, 'features')
    os.makedirs(dest_features_dir, exist_ok=True)
    
    target_col = 'future_price_increase'
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)
    
    data_processed = data[numeric_cols].copy()
    
    # Save processed features
    data_processed.to_csv(os.path.join(dest_features_dir, 'processed_features.csv'), index=False)
    
    y = data_processed[target_col]
    X = data_processed.drop([target_col], axis=1)
    
    return X, y

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Setup directories
        latest_run_dir, new_run_dir = setup_run_directory()
        logger.info(f"Found latest run directory: {latest_run_dir}")
        logger.info(f"Created new run directory: {new_run_dir}")
        
        # Load data
        logger.info("Loading data...")
        X, y = load_data(latest_run_dir, new_run_dir)
        
        # Initialize and train model
        model = SpikePatternPredictor(base_spike_threshold=5.0)
        logger.info("Starting model training...")
        cv_results = model.train_model(X, y, new_run_dir)
        
        # logging
        avg_metrics = pd.DataFrame(cv_results).mean()
        logger.info("\nTraining completed. Average metrics:")
        logger.info("-" * 50)
        logger.info(f"Precision: {avg_metrics['precision']:.4f}")
        logger.info(f"Recall: {avg_metrics['recall']:.4f}")
        logger.info(f"F1 Score: {avg_metrics['f1']:.4f}")
        logger.info(f"High Confidence Precision: {avg_metrics['high_confidence_precision']:.4f}")
        logger.info(avg_metrics)
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Results saved in: {new_run_dir}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()