import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import random
from dataclasses import dataclass

@dataclass
class TradeMetrics:
    total_trades: int
    successful_trades: int
    win_rate: float
    avg_return: float
    max_return: float
    confidence_scores: List[float]

class ValidatedBacktester:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Load model and training data
        self.model_dir = os.path.join(base_dir, 'model_results', '20241118_222036')
        self.model = self._load_model()
        self.training_features = self._load_training_features()
        self.test_results = self._load_test_results()
        self.feature_stats = self._calculate_feature_stats()
        
        # Updated thresholds based on best results
        self.confidence_thresholds = {
            'high': 0.9,    # Performed best in test run
            'medium': 0.85,  # Increased from 0.8 to be more selective
            'min': 0.8      # Base threshold
        }
        
        # Feature thresholds from successful trades
        self.feature_thresholds = {
            'momentum_7m': {'min': 0.05, 'max': 1.2},    # Based on successful range
            'momentum_5m': {'min': 0.05, 'max': 1.3},    # Based on successful range
            'trend_acceleration': {'min': 0.02, 'max': 0.6},
            'price_change_15m': {'min': 0.1, 'max': 1.3}
        }
        
        # Prioritize holding periods
        self.holding_periods = [45, 30]  # Best performing holding periods

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ValidatedBacktester')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_model(self) -> xgb.XGBClassifier:
        """Load the trained model and verify it"""
        model_path = os.path.join(self.model_dir, 'spike_model.json')
        self.logger.info(f"Loading model from {model_path}")
        
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        
        return model

    def _load_training_features(self) -> pd.DataFrame:
        """Load and validate training features"""
        features_path = os.path.join(self.model_dir, 'training_features.csv')
        self.logger.info(f"Loading training features from {features_path}")
        
        df = pd.read_csv(features_path)
        self.feature_columns = [col for col in df.columns 
                              if col not in ['token_address', 'timestamp', 'target']]
        
        return df

    def _load_test_results(self) -> Dict:
        """Load test results for validation"""
        results_path = os.path.join(self.model_dir, 'test_results.json')
        self.logger.info(f"Loading test results from {results_path}")
        
        with open(results_path, 'r') as f:
            return json.load(f)

    def _calculate_feature_stats(self) -> Dict:
        """Calculate feature statistics from training data"""
        stats = {}
        for col in self.feature_columns:
            stats[col] = {
                'mean': float(self.training_features[col].mean()),
                'std': float(self.training_features[col].std()),
                'min': float(self.training_features[col].min()),
                'max': float(self.training_features[col].max())
            }
        return stats

    def calculate_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features exactly matching training"""
        features = pd.DataFrame(index=price_data.index)
        price_series = price_data['value']
        
        # Calculate basic features
        features['momentum_7m'] = price_series.pct_change(7)
        features['momentum_5m'] = price_series.pct_change(5)
        features['trend_acceleration'] = features['momentum_7m'].diff()
        features['momentum_acc_7m'] = features['momentum_7m'].diff(7)
        features['momentum_acc_14m'] = features['momentum_7m'].diff(14)
        features['price_change_15m'] = price_series.pct_change(15)
        
        # Standardize features using training statistics
        for col in features.columns:
            if col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                if std > 0:
                    features[col] = (features[col] - mean) / std
                else:
                    features[col] = 0
        
        # Validate feature ranges
        for col in features.columns:
            if col in self.feature_stats:
                current_range = features[col].agg(['min', 'max'])
                training_range = [self.feature_stats[col]['min'], self.feature_stats[col]['max']]
                self.logger.debug(f"{col} ranges - Current: {current_range}, Training: {training_range}")
        
        return features

    def validate_features(self, features: pd.Series) -> bool:
            """Enhanced validation using successful trade patterns"""
            # Must meet minimum momentum requirements
            if abs(features['momentum_7m']) < self.feature_thresholds['momentum_7m']['min']:
                return False
                
            if abs(features['momentum_5m']) < self.feature_thresholds['momentum_5m']['min']:
                return False
                
            # Check for aligned momentum (both positive or both negative)
            if np.sign(features['momentum_7m']) != np.sign(features['momentum_5m']):
                return False
                
            # Acceleration check
            if abs(features['trend_acceleration']) < self.feature_thresholds['trend_acceleration']['min']:
                return False
                
            return True

    async def analyze_token(self, price_data: pd.DataFrame, holding_period: int) -> Optional[Dict]:
        """Analyze token with validation against training results"""
        try:
            self.logger.info(f"Analyzing {len(price_data)} price points")
            
            diagnostics = {
                'points_analyzed': 0,
                'features_calculated': 0,
                'signals_generated': 0,
                'trades_executed': 0,
                'price_moves_5x': 0
            }
            
            trades = []
            features = self.calculate_features(price_data)
            
            for i in range(len(price_data) - holding_period):
                diagnostics['points_analyzed'] += 1
                
                if i % 5000 == 0:
                    self.logger.info(f"Processed {i}/{len(price_data)} points")
                
                # Get current features
                current_features = features.iloc[i:i+1]
                
                if current_features.isnull().any().any():
                    continue
                
                diagnostics['features_calculated'] += 1
                
                # Validate features match training distribution
                if not self.validate_features(current_features.iloc[0]):
                    continue
                
                # Check for price move first
                entry_price = price_data['value'].iloc[i]
                future_window = price_data['value'].iloc[i:i + holding_period]
                max_price = future_window.max()
                price_multiple = max_price / entry_price
                
                if price_multiple >= 5.0:  # Looking for 5x+ moves
                    diagnostics['price_moves_5x'] += 1
                    
                    # Get model prediction
                    try:
                        confidence_score = float(self.model.predict_proba(current_features)[0][1])
                        
                        # Log features and prediction for major signals
                        if confidence_score >= self.confidence_thresholds['medium']:
                            diagnostics['signals_generated'] += 1
                            self.logger.info(
                                f"\nSignal generated:"
                                f"\nConfidence: {confidence_score:.3f}"
                                f"\nPrice multiple: {price_multiple:.2f}x"
                                f"\nFeatures: {current_features.iloc[0].to_dict()}"
                            )
                            
                            # Record trade
                            exit_price = future_window.iloc[-1]
                            trades.append({
                                'entry_time': price_data.index[i],
                                'entry_price': float(entry_price),
                                'exit_price': float(exit_price),
                                'max_price': float(max_price),
                                'return_multiple': float(price_multiple),
                                'confidence_score': confidence_score,
                                'holding_period': holding_period,
                                'features': current_features.iloc[0].to_dict()
                            })
                            
                            diagnostics['trades_executed'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Prediction error: {str(e)}")
                        continue
            
            if trades:
                results = {
                    'total_trades': len(trades),
                    'high_confidence_trades': len([t for t in trades 
                        if t['confidence_score'] >= self.confidence_thresholds['high']]),
                    'avg_return': np.mean([t['return_multiple'] for t in trades]),
                    'max_return': max([t['return_multiple'] for t in trades]),
                    'trades': trades[:10],  # Store first 10 trades
                    'diagnostics': diagnostics
                }
                
                self.logger.info("\nAnalysis Results:")
                self.logger.info(f"Total trades: {results['total_trades']}")
                self.logger.info(f"Average return: {results['avg_return']:.2f}x")
                self.logger.info(f"High confidence trades: {results['high_confidence_trades']}")
                
                return results
            
            self.logger.info(f"\nNo qualifying trades found. Diagnostics: {diagnostics}")
            return None
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return None

    def save_results(self, results: Dict, output_dir: str):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(output_dir, f'backtest_results_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(results_dir, 'backtest_results.json'), 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Save trade summary
        if results:
            trades_df = pd.DataFrame([t for period_results in results.values() 
                                    for token_results in period_results.values() 
                                    if token_results for t in token_results['trades']])
            if not trades_df.empty:
                trades_df.to_csv(os.path.join(results_dir, 'trades.csv'), index=False)
        
        # Save diagnostics summary
        with open(os.path.join(results_dir, 'diagnostics.txt'), 'w') as f:
            f.write("=== BACKTEST DIAGNOSTICS ===\n\n")
            for period, period_results in results.items():
                f.write(f"\nHolding Period: {period} minutes\n")
                f.write("-" * 50 + "\n")
                
                total_trades = sum(r['total_trades'] for r in period_results.values() if r)
                total_high_conf = sum(r['high_confidence_trades'] for r in period_results.values() if r)
                
                if total_trades > 0:
                    avg_return = np.mean([r['avg_return'] for r in period_results.values() if r])
                    f.write(f"Total Trades: {total_trades}\n")
                    f.write(f"High Confidence Trades: {total_high_conf}\n")
                    f.write(f"Average Return: {avg_return:.2f}x\n")
                else:
                    f.write("No trades found\n")

    def generate_detailed_summary(self, results: Dict, results_dir: str):
        """Generate comprehensive summary files"""
        # Summary CSV with all metrics
        summary_data = []
        for period, period_results in results.items():
            for token, token_results in period_results.items():
                if token_results:
                    summary_data.append({
                        'holding_period': period,
                        'token_address': token,
                        'total_trades': token_results['total_trades'],
                        'high_confidence_trades': token_results['high_confidence_trades'],
                        'avg_return_multiple': token_results['avg_return'],
                        'max_return_multiple': token_results['max_return'],
                        'points_analyzed': token_results['diagnostics']['points_analyzed'],
                        'price_moves_5x': token_results['diagnostics']['price_moves_5x'],
                        'signals_generated': token_results['diagnostics']['signals_generated']
                    })
        
        if summary_data:
            pd.DataFrame(summary_data).to_csv(
                os.path.join(results_dir, 'backtest_summary.csv'), index=False
            )
        
        # Detailed analysis text file
        with open(os.path.join(results_dir, 'detailed_analysis.txt'), 'w') as f:
            f.write("=== DETAILED BACKTEST ANALYSIS ===\n\n")
            
            # Overall statistics
            total_trades = sum(len(token_results['trades']) 
                            for period_results in results.values() 
                            for token_results in period_results.values() if token_results)
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Tokens Analyzed: {len(summary_data)}\n")
            f.write(f"Total Trades Found: {total_trades}\n\n")
            
            # Results by holding period
            for period in results:
                period_trades = [t for token_results in results[period].values() 
                            if token_results 
                            for t in token_results['trades']]
                
                if period_trades:
                    f.write(f"\nHOLDING PERIOD: {period} minutes\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Number of Trades: {len(period_trades)}\n")
                    f.write(f"Average Return: {np.mean([t['return_multiple'] for t in period_trades]):.2f}x\n")
                    f.write(f"Max Return: {max([t['return_multiple'] for t in period_trades]):.2f}x\n")
                    
                    # Confidence score distribution
                    conf_scores = [t['confidence_score'] for t in period_trades]
                    f.write("\nConfidence Score Distribution:\n")
                    f.write(f"Mean: {np.mean(conf_scores):.3f}\n")
                    f.write(f"High Confidence (>0.9): {len([s for s in conf_scores if s >= 0.9])}\n")
                    f.write(f"Medium Confidence (0.8-0.9): {len([s for s in conf_scores if 0.8 <= s < 0.9])}\n\n")
            
            # Feature analysis
            f.write("\nFEATURE ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            all_trades = [t for period_results in results.values() 
                        for token_results in period_results.values() 
                        if token_results 
                        for t in token_results['trades']]
            
            if all_trades:
                features_df = pd.DataFrame([t['features'] for t in all_trades])
                for col in features_df.columns:
                    f.write(f"\n{col}:\n")
                    f.write(f"Mean: {features_df[col].mean():.3f}\n")
                    f.write(f"Std: {features_df[col].std():.3f}\n")
                    f.write(f"Range: [{features_df[col].min():.3f}, {features_df[col].max():.3f}]\n")

async def run_backtest(full_run: bool = True):
    """Run optimized backtest"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backtester = ValidatedBacktester(base_dir)
    
    prices_dir = os.path.join(base_dir, 'data', 'minute_level_data', 'prices')
    price_files = [f for f in os.listdir(prices_dir) if '_minute_prices_' in f]
    
    if not full_run:
        # Use 50 tokens for validation
        price_files = random.sample(price_files, 50)
        backtester.logger.info(f"Running validation with {len(price_files)} tokens")
    else:
        backtester.logger.info(f"Running full analysis with {len(price_files)} tokens")
    
    results = {period: {} for period in backtester.holding_periods}
    
    # Process in batches to show progress
    batch_size = 20
    for i in range(0, len(price_files), batch_size):
        batch = price_files[i:i + batch_size]
        backtester.logger.info(f"\nProcessing batch {i//batch_size + 1}/{len(price_files)//batch_size + 1}")
        
        for price_file in batch:
            try:
                token_address = price_file.split('_minute_prices_')[0]
                backtester.logger.info(f"Processing {token_address}")
                
                df = pd.read_csv(os.path.join(prices_dir, price_file))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                
                if len(df) < 1000:  # Skip tokens with too little data
                    continue
                
                for period in backtester.holding_periods:
                    period_results = await backtester.analyze_token(df, period)
                    
                    if period_results and period_results['total_trades'] > 0:
                        results[period][token_address] = period_results
                        backtester.logger.info(
                            f"{period}min: {period_results['total_trades']} trades, "
                            f"Avg return: {period_results['avg_return']:.2f}x"
                        )
            
            except Exception as e:
                backtester.logger.error(f"Error processing {price_file}: {str(e)}")
                continue
        
        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(base_dir, 'backtest_results', f'batch_{i//batch_size + 1}_{timestamp}')
        os.makedirs(batch_dir, exist_ok=True)
        
        backtester.save_results(results, batch_dir)
        backtester.generate_detailed_summary(results, batch_dir)
        
        backtester.logger.info(f"Batch results saved to {batch_dir}")
    
    # Save final results
    final_dir = os.path.join(base_dir, 'backtest_results', f'final_{timestamp}')
    os.makedirs(final_dir, exist_ok=True)
    
    backtester.save_results(results, final_dir)
    backtester.generate_detailed_summary(results, final_dir)
    
    backtester.logger.info(f"\nFinal results saved to {final_dir}")
    backtester.logger.info("\nBacktest complete!")
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Run full analysis
    asyncio.run(run_backtest(full_run=True))