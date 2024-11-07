import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio

@dataclass
class TradeResult:
    token_address: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    max_price: float
    return_pct: float
    max_return_pct: float
    holding_period: timedelta
    strategy: str
    features_at_entry: Dict  # Store features that triggered the trade

class BacktestEngine:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = self._setup_logging()
        self.trades: List[TradeResult] = []
        self.feature_thresholds = self._load_feature_thresholds()
        self.holding_periods = [3, 15, 45]  # minutes
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger('BacktestEngine')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _load_feature_thresholds(self) -> Dict:
        """Load feature patterns from processed features"""
        # Find the most recent run directory
        runs_dir = os.path.join(self.base_dir, 'data', 'runs')
        run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_') and d.endswith('_full')]
        if not run_dirs:
            raise ValueError("No valid run directories found")
            
        latest_run = sorted(run_dirs)[-1]  # Get most recent run
        features_file = os.path.join(runs_dir, latest_run, 'features', 'processed_features.csv')
        
        self.logger.info(f"Loading features from: {features_file}")
        
        try:
            df = pd.read_csv(features_file)
            
            # Print feature info for verification
            self.logger.info(f"Loaded features: {df.columns.tolist()}")
            self.logger.info(f"Number of feature samples: {len(df)}")
            
            # Calculate thresholds for each feature
            thresholds = {}
            for col in df.columns:
                if col != 'future_price_increase':
                    # Calculate thresholds
                    thresholds[col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].quantile(0.25)),
                        'max': float(df[col].quantile(0.75)),
                        'median': float(df[col].median())
                    }
                    
                    # Log threshold values
                    self.logger.info(f"\nThresholds for {col}:")
                    self.logger.info(f"Mean: {thresholds[col]['mean']:.4f}")
                    self.logger.info(f"Range: {thresholds[col]['min']:.4f} to {thresholds[col]['max']:.4f}")
            
            return thresholds
            
        except Exception as e:
            self.logger.error(f"Error loading features: {str(e)}")
            raise

    def _calculate_features(self, price_data: pd.DataFrame, current_idx: int) -> Dict:
        """Calculate all features using past data only"""
        try:
            # Get lookback window
            lookback_window = price_data.iloc[max(0, current_idx-14):current_idx+1]
            
            if len(lookback_window) < 5:  # Minimum required points
                return None
                
            # Get price array
            prices = lookback_window['value'].values
            available_periods = len(prices) - 1  # Subtract 1 for returns calculation
            
            # Calculate features based on available data
            features = {
                # Price changes
                'price_change_5m': self._calculate_price_change(prices, min(5, available_periods)),
                'price_change_10m': self._calculate_price_change(prices, min(10, available_periods)),
                'price_change_15m': self._calculate_price_change(prices, min(len(prices)-1, 15)),
                
                # Momentum with safety checks
                'momentum_5m': self._calculate_momentum(prices, min(5, available_periods)),
                'momentum_10m': self._calculate_momentum(prices, min(10, available_periods)),
                'momentum_15m': self._calculate_momentum(prices, min(len(prices)-1, 15)),
                
                # RSI features
                'rsi_14': self._calculate_rsi(prices),
                'rsi_14_slope': 0.0,  # Simplified for now
                
                # Basic features
                'trend_strength_15m': self._calculate_trend_strength(prices),
                'volatility_15m': float(np.std(prices) / (np.mean(prices) + 1e-8))
            }
            
            # Log the feature calculations
            self.logger.debug(f"Calculated features using {len(prices)} price points")
            self.logger.debug(f"Feature values: {features}")
            
            return features
                
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}, prices shape: {prices.shape if 'prices' in locals() else 'N/A'}")
            return None

    def _calculate_price_change(self, prices: np.ndarray, period: int) -> float:
        """Calculate price change over specified period"""
        try:
            if len(prices) >= period + 1:
                return float((prices[-1] / prices[-period-1] - 1) * 100)
            return 0.0
        except Exception as e:
            self.logger.debug(f"Price change calculation error: {str(e)}")
            return 0.0

    def _calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calculate momentum over specified period"""
        try:
            if len(prices) >= period + 1:
                # Use only the required number of prices
                price_window = prices[-(period+1):]
                returns = np.diff(price_window) / price_window[:-1]
                return float(np.mean(returns))
            return 0.0
        except Exception as e:
            self.logger.debug(f"Momentum calculation error with {len(prices)} prices for period {period}: {str(e)}")
            return 0.0

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        try:
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            return float(slope / (np.mean(prices) + 1e-8) * 100)
        except Exception as e:
            self.logger.debug(f"Trend strength calculation error: {str(e)}")
            return 0.0

    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Calculate volume trend strength"""
        if len(volumes) > 1:
            return np.mean(np.diff(volumes)) / (np.mean(volumes) + 1e-8)
        return 0

    def _calculate_trend_r2(self, values: np.ndarray) -> float:
        """Calculate R-squared of trend"""
        x = np.arange(len(values))
        y = values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        return r2

    def _calculate_large_trade_ratio(self, volumes: np.ndarray) -> float:
        """Calculate ratio of large trades"""
        if len(volumes) > 0:
            large_trade_threshold = np.percentile(volumes, 75)
            return np.sum(volumes > large_trade_threshold) / len(volumes)
        return 0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI with flexible period"""
        if len(prices) < 2:
            return 50  # Neutral RSI if not enough data
            
        deltas = np.diff(prices)
        gain = deltas.copy()
        loss = deltas.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Use exponential averages for shorter periods
        alpha = 1.0 / period
        avg_gain = gain.mean()
        avg_loss = loss.mean()
        
        for i in range(len(gain)):
            avg_gain = (1 - alpha) * avg_gain + alpha * gain[i]
            avg_loss = (1 - alpha) * avg_loss + alpha * loss[i]
            
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_rsi_slope(self, prices: np.ndarray) -> float:
        """Calculate RSI slope"""
        rsi_values = np.array([self._calculate_rsi(prices[:i+1]) for i in range(len(prices)-1, len(prices))])
        if len(rsi_values) > 1:
            return np.diff(rsi_values)[-1]
        return 0
    
    def verify_calculated_features(self, token_address: str, price_data: pd.DataFrame):
        """Verify feature calculations at specific points of interest"""
        try:
            # Find the point of maximum price change
            price_data['pct_change'] = price_data['value'].pct_change() * 100
            max_change_idx = price_data['pct_change'].idxmax()
            max_change_loc = price_data.index.get_loc(max_change_idx)
            
            self.logger.info(f"\nFeature verification for {token_address}")
            self.logger.info(f"Time of max price change: {max_change_idx}")
            self.logger.info(f"Price change: {price_data.loc[max_change_idx, 'pct_change']:.2f}%")
            
            # Only calculate features if we have enough prior data points
            if max_change_loc >= 15:
                features = self._calculate_features(price_data, max_change_loc - 15)
                if features:
                    self.logger.info("\nFeatures 15 minutes before big move:")
                    for feature, value in features.items():
                        self.logger.info(f"{feature}: {value:.4f}")
                else:
                    self.logger.warning("Could not calculate features: insufficient data")
            else:
                self.logger.warning("Insufficient historical data before max price change")
                
        except Exception as e:
            self.logger.error(f"Error in feature verification: {str(e)}")
    
    def check_entry_conditions(self, features: Dict) -> bool:
        """Check if features match patterns that preceded huge moves"""
        if not features:
            return False
        
        # Patterns we see before 30x+ moves
        key_conditions = {
            'rsi_14': {
                'value': features.get('rsi_14', 0),
                'min': 24.0,  # Looking for oversold conditions
                'max': 35.0,
                'weight': 2.0
            },
            'trend_strength_15m': {
                'value': features.get('trend_strength_15m', 0),
                'min': 3.0,
                'max': 6.0,
                'weight': 2.0
            },
            'volatility_15m': {
                'value': features.get('volatility_15m', 0),
                'min': 0.03,
                'max': 0.05,
                'weight': 1.5
            },
            'momentum_5m': {
                'value': features.get('momentum_5m', 0),
                'min': -0.025,
                'max': 0.002,
                'weight': 1.5
            },
            'rsi_14_slope': {
                'value': features.get('rsi_14_slope', 0),
                'min': -8.5,
                'max': 3.0,
                'weight': 1.0
            }
        }
        
        total_weight = sum(c['weight'] for c in key_conditions.values())
        matched_weight = 0
        
        for feature, params in key_conditions.items():
            value = params['value']
            is_match = params['min'] <= value <= params['max']
            
            if is_match:
                matched_weight += params['weight']
                self.logger.debug(f"{feature}: {value:.4f} ✓")
            else:
                self.logger.debug(f"{feature}: {value:.4f} ✗")
        
        match_ratio = matched_weight / total_weight
        
        if match_ratio >= 0.80:
            self.logger.info(f"Found potential setup (match ratio: {match_ratio:.2f})")
            for feature, params in key_conditions.items():
                self.logger.info(f"{feature}: {params['value']:.4f}")
            return True
        
        return False
    


    def simulate_trade(self, price_data: pd.DataFrame, entry_time: datetime, 
                    token_address: str, holding_period: int, features: Dict) -> Optional[TradeResult]:
        """Simulate a single trade"""
        try:
            # Get entry data
            entry_mask = price_data['timestamp'] == entry_time
            if not any(entry_mask):
                return None
            
            entry_price = float(price_data.loc[entry_mask, 'value'].iloc[0])
            exit_time = entry_time + pd.Timedelta(minutes=holding_period)
            
            # Get prices during holding period
            trade_window = price_data[
                (price_data['timestamp'] > entry_time) & 
                (price_data['timestamp'] <= exit_time)
            ]
            
            if trade_window.empty:
                return None
                
            # Calculate results
            exit_price = float(trade_window['value'].iloc[-1])
            max_price = float(trade_window['value'].max())
            actual_exit_time = trade_window['timestamp'].iloc[-1]
            
            return_pct = ((exit_price - entry_price) / entry_price) * 100
            max_return_pct = ((max_price - entry_price) / entry_price) * 100
            
            # Log trade details
            self.logger.info(f"\nTrade opened:")
            self.logger.info(f"Entry Time: {entry_time}")
            self.logger.info(f"Entry Price: {entry_price:.8f}")
            self.logger.info(f"Exit Price: {exit_price:.8f}")
            self.logger.info(f"Return: {return_pct:.2f}%")
            self.logger.info(f"Max Return: {max_return_pct:.2f}%")
            
            return TradeResult(
                token_address=token_address,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=actual_exit_time,
                exit_price=exit_price,
                max_price=max_price,
                return_pct=return_pct,
                max_return_pct=max_return_pct,
                holding_period=actual_exit_time - entry_time,
                strategy=f"{holding_period}min_hold",
                features_at_entry=features
            )
            
        except Exception as e:
            self.logger.error(f"Error simulating trade: {str(e)}")
            self.logger.debug(f"Entry time: {entry_time}")
            self.logger.debug(f"Data range: {price_data['timestamp'].min()} to {price_data['timestamp'].max()}")
            return None


    async def backtest_token(self, price_data: pd.DataFrame, token_address: str, 
                            holding_period: int) -> List[TradeResult]:
        """Backtest looking for massive price multipliers"""
        token_trades = []
        max_spike_duration = 60  # Maximum minutes to look for spike
        
        # Need enough data for feature calculation
        if len(price_data) < 15:
            return []
        
        for i in range(15, len(price_data) - max_spike_duration):
            current_time = price_data.iloc[i]['timestamp']
            start_price = price_data.iloc[i]['value']
            
            # Calculate features at this point
            features = self._calculate_features(price_data, i)
            
            if features and self.check_entry_conditions(features):
                # Look for massive moves in next period
                future_window = price_data.iloc[i:i+max_spike_duration]
                max_price = future_window['value'].max()
                price_multiple = max_price / start_price
                
                # Look for 5x or greater moves (being conservative)
                if price_multiple >= 5.0:  # 500% minimum increase
                    peak_idx = future_window['value'].idxmax()
                    trade_result = TradeResult(
                        token_address=token_address,
                        entry_time=current_time,
                        entry_price=start_price,
                        exit_time=price_data.loc[peak_idx, 'timestamp'],
                        exit_price=max_price,
                        max_price=max_price,
                        return_pct=(price_multiple - 1) * 100,  # Convert to percentage for logging
                        max_return_pct=(price_multiple - 1) * 100,
                        holding_period=pd.Timedelta(minutes=holding_period),
                        strategy=f"{holding_period}min_hold",
                        features_at_entry=features
                    )
                    
                    self.logger.info(f"\nFound {price_multiple:.1f}x move!")
                    self.logger.info(f"Start price: {start_price:.10f}")
                    self.logger.info(f"Max price: {max_price:.10f}")
                    self.logger.info(f"Time to peak: {(peak_idx - i)} minutes")
                    
                    token_trades.append(trade_result)
        
        if token_trades:
            self.logger.info(f"\nFound {len(token_trades)} large moves for {token_address}")
            for trade in token_trades:
                self.logger.info(f"{trade.return_pct/100:.1f}x multiple in {trade.holding_period}")
        
        return token_trades
    
    async def validate_backtest(self, num_validation_tokens: int = 3) -> bool:
        """Run backtest on a small set of tokens to validate approach"""
        price_dir = os.path.join(self.base_dir, 'data', 'historical_prices_1m')
        price_files = [f for f in os.listdir(price_dir) if f.endswith('.csv')]
        
        validation_tokens = price_files[:num_validation_tokens]
        self.logger.info(f"Running validation backtest on {len(validation_tokens)} tokens")
        
        validation_trades = []
        for file in validation_tokens:
            token_address = file.split('_prices_')[0]
            df = pd.read_csv(os.path.join(price_dir, file))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            self.logger.info(f"\nValidating token: {token_address}")
            
            for holding_period in self.holding_periods:
                trades = await self.backtest_token(df, token_address, holding_period)
                if trades:
                    validation_trades.extend(trades)
                    self.logger.info(f"Holding period {holding_period}min results:")
                    self.logger.info(f"Number of trades: {len(trades)}")
                    returns = [t.return_pct for t in trades]
                    self.logger.info(f"Average return: {np.mean(returns):.2f}%")
                    self.logger.info(f"Win rate: {len([r for r in returns if r > 0]) / len(returns):.2%}")
        
        return len(validation_trades) > 0

    async def run_full_backtest(self) -> Dict:
        """Run full backtest with progress tracking"""
        # First run validation
        self.logger.info("Starting validation phase...")
        is_valid = await self.validate_backtest(num_validation_tokens=5)
        
        if not is_valid:
            self.logger.error("Validation failed - no valid trades found")
            return {}
            
        self.logger.info("Validation successful - proceeding with full backtest")
        
        # Clear validation trades
        self.trades = []
        
        # Run full backtest
        price_dir = os.path.join(self.base_dir, 'data', 'historical_prices_1m')
        price_files = [f for f in os.listdir(price_dir) if f.endswith('.csv')]
        total_tokens = len(price_files)
        
        self.logger.info(f"\nStarting full backtest on {total_tokens} tokens")
        results_by_token = {}
        
        for idx, file in enumerate(price_files, 1):
            token_address = file.split('_prices_')[0]
            df = pd.read_csv(os.path.join(price_dir, file))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            token_trades = []
            for holding_period in self.holding_periods:
                self.logger.info(f"Processing token {idx}/{total_tokens}: {token_address} - {holding_period}min hold")
                trades = await self.backtest_token(df, token_address, holding_period)
                if trades:
                    token_trades.extend(trades)
                    self.trades.extend(trades)
            
            if token_trades:
                results_by_token[token_address] = {
                    'total_trades': len(token_trades),
                    'avg_return': np.mean([t.return_pct for t in token_trades]),
                    'max_return': max([t.return_pct for t in token_trades]),
                    'trades_by_period': {
                        f"{period}min": len([t for t in token_trades if t.strategy == f"{period}min_hold"])
                        for period in self.holding_periods
                    }
                }
            
            if idx % 10 == 0:
                self.logger.info(f"Progress: {idx}/{total_tokens} tokens processed ({idx/total_tokens*100:.1f}%)")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.base_dir, 'data', 'backtest_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save overall summary
        overall_results = self.analyze_results()
        with open(os.path.join(results_dir, f'backtest_summary_{timestamp}.json'), 'w') as f:
            json.dump(overall_results, f, indent=4)
        
        # Save detailed results by token
        with open(os.path.join(results_dir, f'backtest_details_{timestamp}.json'), 'w') as f:
            json.dump(results_by_token, f, indent=4)
        
        # Save all trades data
        trades_df = pd.DataFrame([
            {
                'token_address': t.token_address,
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat(),
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'return_pct': t.return_pct,
                'max_return_pct': t.max_return_pct,
                'strategy': t.strategy,
                **t.features_at_entry
            }
            for t in self.trades
        ])
        trades_df.to_csv(os.path.join(results_dir, f'all_trades_{timestamp}.csv'), index=False)
        
        self.logger.info(f"\nBacktest complete. Results saved to {results_dir}")
        self.logger.info(f"Total trades: {len(self.trades)}")
        self.logger.info(f"Tokens with trades: {len(results_by_token)}")
        
        return overall_results

    def analyze_results(self) -> Dict:
        """Analyze backtest results with detailed metrics"""
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame([{
            'token_address': t.token_address,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'return_pct': t.return_pct,
            'max_return_pct': t.max_return_pct,
            'holding_period': t.holding_period.total_seconds() / 60,
            'strategy': t.strategy,
            **t.features_at_entry
        } for t in self.trades])
        
        results = {}
        
        # Overall metrics
        results['overall'] = {
            'total_trades': len(trades_df),
            'unique_tokens': len(trades_df['token_address'].unique()),
            'avg_return': float(trades_df['return_pct'].mean()),
            'median_return': float(trades_df['return_pct'].median()),
            'win_rate': float(len(trades_df[trades_df['return_pct'] > 0]) / len(trades_df)),
            'return_buckets': {
                '>100%': int(len(trades_df[trades_df['return_pct'] > 100])),
                '50-100%': int(len(trades_df[(trades_df['return_pct'] >= 50) & (trades_df['return_pct'] <= 100)])),
                '20-50%': int(len(trades_df[(trades_df['return_pct'] >= 20) & (trades_df['return_pct'] < 50)])),
                '0-20%': int(len(trades_df[(trades_df['return_pct'] >= 0) & (trades_df['return_pct'] < 20)])),
                '<0%': int(len(trades_df[trades_df['return_pct'] < 0]))
            }
        }
        
        # Per holding period metrics
        for period in self.holding_periods:
            period_trades = trades_df[trades_df['strategy'] == f"{period}min_hold"]
            if not period_trades.empty:
                results[f"{period}min_holding"] = {
                    'total_trades': len(period_trades),
                    'win_rate': float(len(period_trades[period_trades['return_pct'] > 0]) / len(period_trades)),
                    'avg_return': float(period_trades['return_pct'].mean()),
                    'median_return': float(period_trades['return_pct'].median()),
                    'max_return': float(period_trades['return_pct'].max()),
                    'avg_winning_trade': float(period_trades[period_trades['return_pct'] > 0]['return_pct'].mean()),
                    'avg_losing_trade': float(period_trades[period_trades['return_pct'] < 0]['return_pct'].mean()),
                    'return_buckets': {
                        '>100%': int(len(period_trades[period_trades['return_pct'] > 100])),
                        '50-100%': int(len(period_trades[(period_trades['return_pct'] >= 50) & (period_trades['return_pct'] <= 100)])),
                        '20-50%': int(len(period_trades[(period_trades['return_pct'] >= 20) & (period_trades['return_pct'] < 50)])),
                        '0-20%': int(len(period_trades[(period_trades['return_pct'] >= 0) & (period_trades['return_pct'] < 20)])),
                        '<0%': int(len(period_trades[period_trades['return_pct'] < 0]))
                    }
                }
        
        return results

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backtester = BacktestEngine(base_dir)
    
    try:
        results = await backtester.run_full_backtest()
        
        if results:
            # Save results
            results_dir = os.path.join(base_dir, 'data', 'backtest_results')
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(results_dir, f'backtest_results_{timestamp}.json')
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            backtester.logger.info(f"Results saved to {results_file}")
            
            # Log summary statistics
            backtester.logger.info("\nBacktest Results Summary:")
            backtester.logger.info("-" * 50)
            for period, metrics in results.items():
                backtester.logger.info(f"\n{period} results:")
                backtester.logger.info(f"Total trades: {metrics['total_trades']}")
                backtester.logger.info(f"Win rate: {metrics['win_rate']:.2%}")
                backtester.logger.info(f"Average return: {metrics['avg_return']:.2f}%")
                backtester.logger.info(f"Median return: {metrics['median_return']:.2f}%")
    
    except Exception as e:
        backtester.logger.error(f"Error in backtesting: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())