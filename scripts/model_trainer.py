import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Custom JSON encoder for numpy and pandas types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class TradeResult:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    max_price: float
    return_pct: float
    max_return_pct: float
    holding_period: int
    feature_values: Dict[str, float]

class SpikeModelTrainer:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = self._setup_logging()
        
        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        
        # Enhanced feature thresholds
        self.feature_thresholds = {
            # Original features
            'price_change_15m': 0.5,
            'momentum_14m': 0.4,
            'price_change_10m': 0.3,
            'momentum_5m': 0.2,
            'trend_acceleration': 0.15,
            'momentum_7m': 0.25,
            
            # New features
            'volume_acceleration': 0.3,
            'volume_trend': 0.2,
            'volume_spike': 1.5,
            'momentum_7m_strength': 0.4,
            'momentum_7m_acceleration': 0.2,
            'price_volatility': 0.3,
            'trend_consistency': 0.7,
            'price_acceleration_5m': 0.2,
            'momentum_volume_interaction': 0.3,
            'bullish_pattern': 1.0,
            'acceleration_pattern': 1.0,
            'volatility_ratio_15m': 1.2,
            'trend_strength_15m': 0.5
        }
        
        # Testing different holding periods
        self.holding_periods = [3, 5, 10, 15, 30, 45, 60]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('SpikeModelTrainer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
            """Prepare balanced training data with enhanced features"""
            self.logger.info("Preparing training data...")
            
            # Identify spike periods (original spikes)
            spike_data = df.copy()
            
            # Convert spike_time to datetime at the beginning
            spike_data['spike_time'] = pd.to_datetime(spike_data['spike_time'])
            
            # Create non-spike periods first
            non_spike_data = []
            tokens = df['token_address'].unique()
            
            self.logger.info(f"Creating non-spike samples for {len(tokens)} tokens...")
            
            # Get original feature columns before adding new ones
            original_feature_cols = [col for col in df.columns 
                                if col not in ['token_address', 'spike_time']]
            
            for token in tokens:
                token_spikes = df[df['token_address'] == token]
                
                for _, spike in token_spikes.iterrows():
                    # Pre-spike period (30 minutes before)
                    pre_spike = spike.copy()
                    pre_spike['spike_time'] = pd.to_datetime(spike['spike_time']) - pd.Timedelta(minutes=30)
                    pre_spike['is_spike'] = 0
                    # Reduce original feature values for non-spike periods
                    for col in original_feature_cols:
                        pre_spike[col] = pre_spike[col] * 0.4
                    
                    # Post-spike period
                    post_spike = spike.copy()
                    post_spike['spike_time'] = pd.to_datetime(spike['spike_time']) + pd.Timedelta(minutes=30)
                    post_spike['is_spike'] = 0
                    # Different feature values for post-spike period
                    for col in original_feature_cols:
                        post_spike[col] = post_spike[col] * 0.3
                    
                    non_spike_data.extend([pre_spike, post_spike])
            
            # Convert to DataFrame
            non_spike_df = pd.DataFrame(non_spike_data)
            spike_data['is_spike'] = 1
            
            # Combine and sort by time
            full_df = pd.concat([spike_data, non_spike_df])
            full_df = full_df.sort_values('spike_time').reset_index(drop=True)
            
            # Now add new features to the combined dataset
            def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
                enhanced_data = data.copy()
                
                # Volume-based features
                if 'volume_15m' in enhanced_data.columns:
                    # Volume acceleration
                    enhanced_data['volume_acceleration'] = enhanced_data['volume_15m'].pct_change()
                    # Volume trend
                    enhanced_data['volume_trend'] = (enhanced_data['volume_15m'] / enhanced_data['volume_15m'].rolling(3).mean()) - 1
                    # Volume spikes
                    vol_std = enhanced_data['volume_15m'].rolling(10).std()
                    enhanced_data['volume_spike'] = (enhanced_data['volume_15m'] - enhanced_data['volume_15m'].rolling(10).mean()) / vol_std
                
                # Momentum and trend features
                for window in [3, 5, 7, 10, 14]:
                    # Momentum strength
                    col_name = f'momentum_{window}m'
                    if col_name in enhanced_data.columns:
                        enhanced_data[f'{col_name}_strength'] = enhanced_data[col_name].abs()
                        enhanced_data[f'{col_name}_acceleration'] = enhanced_data[col_name].diff()
                
                # Price pattern features
                if 'price_change_15m' in enhanced_data.columns:
                    # Price volatility
                    enhanced_data['price_volatility'] = enhanced_data['price_change_15m'].rolling(5).std()
                    # Price trend consistency
                    enhanced_data['trend_consistency'] = (
                        (enhanced_data['price_change_15m'] > 0).rolling(5).sum() / 5
                    )
                    # Price acceleration
                    enhanced_data['price_acceleration_3m'] = enhanced_data['price_change_15m'].diff(3)
                    enhanced_data['price_acceleration_5m'] = enhanced_data['price_change_15m'].diff(5)
                
                # Interaction features
                if all(col in enhanced_data.columns for col in ['momentum_7m', 'volume_15m']):
                    enhanced_data['momentum_volume_interaction'] = enhanced_data['momentum_7m'] * enhanced_data['volume_15m']
                
                # Pattern recognition features
                if 'price_change_15m' in enhanced_data.columns and 'price_change_10m' in enhanced_data.columns:
                    # Bullish momentum pattern
                    enhanced_data['bullish_pattern'] = (
                        (enhanced_data['price_change_15m'] > enhanced_data['price_change_10m']) &
                        (enhanced_data['price_change_10m'] > 0)
                    ).astype(int)
                    
                    # Acceleration pattern
                    enhanced_data['acceleration_pattern'] = (
                        enhanced_data['price_change_5m'].diff() > 0
                    ).astype(int)
                
                # Volatility features
                for window in [5, 10, 15]:
                    price_col = f'price_change_{window}m'
                    if price_col in enhanced_data.columns:
                        # Volatility ratio
                        enhanced_data[f'volatility_ratio_{window}m'] = (
                            enhanced_data[price_col].rolling(3).std() /
                            enhanced_data[price_col].rolling(10).std()
                        )
                        
                        # Trend strength
                        enhanced_data[f'trend_strength_{window}m'] = (
                            enhanced_data[price_col].rolling(window).mean() /
                            enhanced_data[price_col].rolling(window).std()
                        )
                
                # Clean up NaN values
                for col in enhanced_data.columns:
                    if col not in ['token_address', 'spike_time']:
                        enhanced_data[col] = enhanced_data[col].fillna(0)
                
                return enhanced_data
            
            # Apply feature engineering to the full dataset
            full_df = engineer_features(full_df)
            
            # Print data info
            self.logger.info(f"Total samples: {len(full_df)}")
            self.logger.info(f"Spike samples: {len(spike_data)}")
            self.logger.info(f"Non-spike samples: {len(non_spike_df)}")
            
            # Verify feature values
            self.logger.info("\nFeature value ranges:")
            for col in self.feature_thresholds.keys():
                spike_vals = full_df[full_df['is_spike'] == 1][col]
                non_spike_vals = full_df[full_df['is_spike'] == 0][col]
                self.logger.info(f"\n{col}:")
                self.logger.info(f"  Spike - min: {spike_vals.min():.4f}, max: {spike_vals.max():.4f}")
                self.logger.info(f"  Non-spike - min: {non_spike_vals.min():.4f}, max: {non_spike_vals.max():.4f}")
            
            return full_df
    
    def test_features(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
            """Test the predictive power of individual features"""
                    # Suppress numpy warnings about invalid values
            import warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            self.logger.info("\nTesting individual feature performance...")
            
            # Prepare data first
            training_df = self.prepare_training_data(df)
            
            # Get all features except metadata columns
            all_features = [col for col in training_df.columns 
                        if col not in ['token_address', 'spike_time', 'is_spike']]
            
            feature_scores = {}
            
            for feature in all_features:
                try:
                    # Create single-feature dataset
                    X = training_df[[feature]]
                    y = training_df['is_spike']
                    
                    # Handle infinities and NaNs
                    X = X.replace([np.inf, -np.inf], np.nan)
                    X = X.fillna(0)
                    
                    # Scale the feature
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    # Split data
                    train_size = int(len(X_scaled) * 0.8)
                    X_train = X_scaled[:train_size]
                    X_test = X_scaled[train_size:]
                    y_train = y[:train_size]
                    y_test = y[train_size:]
                    
                    # Train a simple model
                    model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1
                    )
                    model.fit(X_train, y_train)
                    
                    # Get predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    scores = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0),
                        'auc_roc': roc_auc_score(y_test, y_pred_proba),
                        'correlation': abs(np.corrcoef(X_test.flatten(), y_test)[0, 1])
                    }
                    
                    # Calculate spike detection rate
                    spike_mask = y_test == 1
                    if sum(spike_mask) > 0:
                        spike_detection = sum((y_pred == 1) & spike_mask) / sum(spike_mask)
                        scores['spike_detection'] = spike_detection
                    else:
                        scores['spike_detection'] = 0
                    
                    feature_scores[feature] = scores
                    
                    # Log results
                    self.logger.info(f"\nResults for {feature}:")
                    self.logger.info(f"Accuracy: {scores['accuracy']:.4f}")
                    self.logger.info(f"Precision: {scores['precision']:.4f}")
                    self.logger.info(f"Recall: {scores['recall']:.4f}")
                    self.logger.info(f"AUC-ROC: {scores['auc_roc']:.4f}")
                    self.logger.info(f"Correlation: {scores['correlation']:.4f}")
                    self.logger.info(f"Spike Detection Rate: {scores['spike_detection']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error testing feature {feature}: {str(e)}")
                    feature_scores[feature] = {'error': str(e)}
            
            # Rank features by spike detection ability
            ranked_features = sorted(
                [(f, s['spike_detection']) for f, s in feature_scores.items() if 'spike_detection' in s],
                key=lambda x: x[1],
                reverse=True
            )
            
            self.logger.info("\nFeature Rankings by Spike Detection:")
            for feature, score in ranked_features:
                self.logger.info(f"{feature}: {score:.4f}")
            
            return feature_scores

    def analyze_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze feature correlations and predictive power"""
        self.logger.info("\nAnalyzing feature relationships...")
        
        feature_cols = list(self.feature_thresholds.keys())
        analysis_results = {}
        
        # Make sure we have the target column
        if 'is_spike' not in df.columns:
            raise ValueError("DataFrame must include 'is_spike' column for analysis")
        
        # Calculate correlations with spike probability
        correlations = pd.DataFrame()
        correlations['correlation_with_spike'] = df[feature_cols].corrwith(df['is_spike'])
        correlations['abs_correlation'] = correlations['correlation_with_spike'].abs()
        correlations = correlations.sort_values('abs_correlation', ascending=False)
        
        self.logger.info("\nFeature Correlations with Spike Probability:")
        for feature in correlations.index:
            corr = correlations.loc[feature, 'correlation_with_spike']
            self.logger.info(f"{feature}: {corr:.4f}")
        
        analysis_results['correlations'] = correlations
        
        # Calculate feature distributions for spike vs non-spike
        distributions = pd.DataFrame()
        for feature in feature_cols:
            spike_stats = df[df['is_spike'] == 1][feature].describe()
            non_spike_stats = df[df['is_spike'] == 0][feature].describe()
            
            distributions[f'{feature}_spike'] = spike_stats
            distributions[f'{feature}_non_spike'] = non_spike_stats
        
        analysis_results['distributions'] = distributions
        
        # Calculate cross-correlations between features
        cross_corr = df[feature_cols].corr()
        analysis_results['cross_correlations'] = cross_corr
        
        # Identify most important feature combinations
        pairs = []
        for i, f1 in enumerate(feature_cols):
            for j, f2 in enumerate(feature_cols[i+1:], i+1):
                combined_corr = df[[f1, f2]].corrwith(df['is_spike'])
                mean_corr = combined_corr.mean()
                pairs.append({
                    'feature1': f1,
                    'feature2': f2,
                    'combined_correlation': mean_corr,
                    'cross_correlation': cross_corr.loc[f1, f2]
                })
        
        feature_pairs = pd.DataFrame(pairs).sort_values('combined_correlation', ascending=False)
        analysis_results['feature_pairs'] = feature_pairs
        
        # Log the best feature combinations
        self.logger.info("\nTop Feature Combinations:")
        for _, row in feature_pairs.head().iterrows():
            self.logger.info(f"{row['feature1']} + {row['feature2']}:")
            self.logger.info(f"  Combined correlation: {row['combined_correlation']:.4f}")
            self.logger.info(f"  Cross-correlation: {row['cross_correlation']:.4f}")
        
        return analysis_results
    

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict:
            """Train model with focus on most predictive features"""
            results = {}
            
            # Prepare training data
            training_df = self.prepare_training_data(df)
            
            # Analyze features
            feature_analysis = self.analyze_features(training_df)
            results['feature_analysis'] = feature_analysis
            
            # Get top features based on correlation
            top_features = feature_analysis['correlations']\
                .sort_values('abs_correlation', ascending=False)\
                .head(6).index.tolist()
            
            # Split data into train and final test sets
            train_size = int(len(training_df) * 0.8)
            train_df = training_df.iloc[:train_size]
            test_df = training_df.iloc[train_size:]
            
            X_train = train_df[top_features]
            y_train = train_df['is_spike']
            X_test = test_df[top_features]
            y_test = test_df['is_spike']
            
            # Print class distribution
            self.logger.info("\nClass Distribution:")
            self.logger.info(f"Training set - Spikes: {sum(y_train==1)} ({sum(y_train==1)/len(y_train)*100:.1f}%)")
            self.logger.info(f"Training set - Non-spikes: {sum(y_train==0)} ({sum(y_train==0)/len(y_train)*100:.1f}%)")
            self.logger.info(f"Test set - Spikes: {sum(y_test==1)} ({sum(y_test==1)/len(y_test)*100:.1f}%)")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Balance classes
                random_state=42,
                eval_metric='logloss'
            )
            
            # Train with evaluation set
            eval_set = [(X_test_scaled, y_test)]
            self.model.fit(
                X_train_scaled, 
                y_train,
                eval_set=eval_set,
                verbose=True
            )
            
            # Get predictions on test set
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # NEW: Store features and predictions for saving
            self.train_features = X_train
            self.test_y = y_test
            self.test_predictions = y_pred_proba
            
            # Calculate metrics
            test_scores = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }
            
            self.logger.info("\nTest Set Results:")
            for metric, value in test_scores.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            # Test different confidence thresholds
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            threshold_results = {}
            
            for threshold in thresholds:
                self.logger.info(f"\nEvaluating confidence threshold: {threshold:.1f}")
                high_conf_mask = y_pred_proba >= threshold
                
                if sum(high_conf_mask) > 0:
                    test_subset = test_df.iloc[high_conf_mask]
                    conf_scores = {
                        'num_trades': int(sum(high_conf_mask)),
                        'precision': float(precision_score(y_test[high_conf_mask], y_pred[high_conf_mask])),
                        'recall': float(recall_score(y_test[high_conf_mask], y_pred[high_conf_mask])),
                        'avg_return': float(test_subset['future_price_increase'].mean())
                    }
                    threshold_results[str(threshold)] = conf_scores
                    
                    self.logger.info(f"Number of trades: {conf_scores['num_trades']}")
                    self.logger.info(f"Precision: {conf_scores['precision']:.4f}")
                    self.logger.info(f"Recall: {conf_scores['recall']:.4f}")
                    self.logger.info(f"Average return: {conf_scores['avg_return']:.2f}x")
            
            results['test_scores'] = test_scores
            results['threshold_results'] = threshold_results
            
            return results
    
    def save_feature_test_results(self, feature_scores: Dict, run_dir: str):
            """Save feature testing results"""
            results_dir = os.path.join(run_dir, 'feature_testing')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save detailed results
            filtered_scores = {}
            for feature, scores in feature_scores.items():
                if isinstance(scores, dict) and 'error' not in scores:
                    filtered_scores[feature] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                        for k, v in scores.items()
                    }
            
            with open(os.path.join(results_dir, 'feature_scores.json'), 'w') as f:
                json.dump(filtered_scores, f, indent=4)
            
            # Create visualization
            plt.figure(figsize=(15, 8))
            features = []
            scores = []
            
            for feature, metrics in feature_scores.items():
                if isinstance(metrics, dict) and 'spike_detection' in metrics:
                    features.append(feature)
                    scores.append(metrics['spike_detection'])
            
            y_pos = np.arange(len(features))
            plt.barh(y_pos, scores)
            plt.yticks(y_pos, features)
            plt.xlabel('Spike Detection Rate')
            plt.title('Feature Performance in Spike Detection')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'feature_performance.png'))
            plt.close()
            
            # Also save as CSV for easier analysis
            feature_df = pd.DataFrame([
                {
                    'feature': feature,
                    'spike_detection': metrics['spike_detection'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'auc_roc': metrics['auc_roc']
                }
                for feature, metrics in feature_scores.items()
                if isinstance(metrics, dict) and 'spike_detection' in metrics
            ])
            feature_df.to_csv(os.path.join(results_dir, 'feature_metrics.csv'), index=False)
            
            self.logger.info(f"\nFeature test results saved to: {results_dir}")

    def save_results(self, results: Dict):
            """Save model results and feature analysis"""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(self.base_dir, 'model_results', timestamp)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save feature analysis
            feature_analysis = results['feature_analysis']
            feature_analysis['correlations'].to_csv(
                os.path.join(results_dir, 'feature_correlations.csv')
            )
            feature_analysis['cross_correlations'].to_csv(
                os.path.join(results_dir, 'feature_cross_correlations.csv')
            )
            feature_analysis['feature_pairs'].to_csv(
                os.path.join(results_dir, 'feature_pairs.csv')
            )
            
            # Save test results
            test_results = {
                'test_metrics': results['test_scores'],
                'threshold_analysis': results['threshold_results']
            }
            with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
                json.dump(test_results, f, indent=4)
            
            # Save model
            model_file = os.path.join(results_dir, 'spike_model.json')
            self.model.save_model(model_file)
            
            # NEW: Save prediction scores and features used
            if hasattr(self, 'test_predictions'):
                pred_df = pd.DataFrame({
                    'true_label': self.test_y,
                    'predicted_prob': self.test_predictions,
                    'predicted_label': (self.test_predictions >= 0.5).astype(int)
                })
                pred_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
            
            # NEW: Save features used for training
            if hasattr(self, 'train_features'):
                feature_data = pd.DataFrame(self.scaler.transform(self.train_features),
                                        columns=self.train_features.columns)
                feature_data.to_csv(os.path.join(results_dir, 'training_features.csv'), index=False)
                
                # Save feature names and importance
                feature_importance = pd.DataFrame({
                    'feature': self.train_features.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                feature_importance.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
            
            # Create analysis plots
            self._create_feature_analysis_plots(feature_analysis, results_dir)
            
            self.logger.info(f"\nResults saved to: {results_dir}")
            self.logger.info(f"Model saved as: {model_file}")

    def _create_feature_analysis_plots(self, feature_analysis: Dict, save_dir: str):
            """Create feature analysis visualizations"""
            # Feature importance plot
            plt.figure(figsize=(12, 6))
            correlations = feature_analysis['correlations']['abs_correlation']
            correlations.plot(kind='bar')
            plt.title('Feature Importance by Correlation')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
            plt.close()
            
            # Cross-correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                feature_analysis['cross_correlations'], 
                annot=True, 
                cmap='coolwarm',
                center=0
            )
            plt.title('Feature Cross-Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'feature_correlations.png'))
            plt.close()

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find most recent features file
    features_dir = os.path.join(base_dir, 'data', 'minute_level_data', 'features')
    feature_files = sorted([f for f in os.listdir(features_dir) if f.startswith('spike_features_')])
    
    if not feature_files:
        print("No feature files found!")
        return
        
    latest_features = os.path.join(features_dir, feature_files[-1])
    
    try:
        # Initialize trainer
        trainer = SpikeModelTrainer(base_dir)
        
        # Load and prepare data
        df = pd.read_csv(latest_features)
        # Convert spike_time to datetime immediately after loading
        df['spike_time'] = pd.to_datetime(df['spike_time'])
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, 'model_results', timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # Test features first
        feature_scores = trainer.test_features(df)
        trainer.save_feature_test_results(feature_scores, run_dir)
        
        # Train and evaluate
        results = trainer.train_and_evaluate(df)
        
        # Save results
        trainer.save_results(results)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()