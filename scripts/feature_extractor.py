import pandas as pd
import numpy as np
import os
import traceback
import json
import ast
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging
import warnings
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import shutil

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')

@dataclass
class FeatureConfig:
    # Windows for different calculations (in minutes)
    FULL_WINDOW: int = 15
    SHORT_WINDOW: int = 5
    PRICE_WINDOWS: List[int] = (1, 3, 5, 10, 15)
    VOLUME_WINDOWS: List[int] = (1, 3, 5, 10, 15)
    
    # Add shorter intervals for micro-pattern detection
    MICRO_WINDOWS: List[int] = (1, 2, 3)  # 1-3 minute patterns
    
    # Add composite features
    COMPOSITE_FEATURES: bool = True
    
    # Feature importance thresholds
    MIN_FEATURE_IMPORTANCE: float = 0.05
    MAX_CORRELATION: float = 0.98

    # Pattern detection parameters
    PATTERN_LENGTH: int = 5  # Length for pattern detection
    RSI_WINDOWS: List[int] = (7, 14, 21)  # Multiple RSI periods
    BB_WINDOWS: List[int] = (20, 30)  # Multiple Bollinger Band periods
    
    # Market structure parameters
    MIN_TRADE_SIZE_PERCENTILE: float = 0.95  # For whale detection
    LIQUIDITY_IMPACT_THRESHOLD: float = 0.01  # 1% of liquidity
    
    # New feature groups
    PATTERN_DETECTION: bool = True
    MICROSTRUCTURE: bool = True
    WALLET_ANALYSIS: bool = True

class FeatureExtractor:
    def __init__(self, base_dir: str):
        """Initialize feature extractor with proper run directory creation"""
        self.base_dir = base_dir
        self.run_dir = self._create_new_run_dir()
        self._setup_logging()
        self.features_dir = os.path.join(self.run_dir, 'features')
        os.makedirs(self.features_dir, exist_ok=True)
        self.scaler = StandardScaler()

    def _create_new_run_dir(self) -> str:
        """Create a new run directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.base_dir, "data", "runs", f"run_{timestamp}")
        
        # Create required subdirectories
        for subdir in ['features', 'logs', 'price_data', 'trades', 'token_info', 'analysis', 
                      os.path.join('features', 'individual_features')]:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
        
        # Initialize run info
        run_info = {
            'timestamp': timestamp,
            'start_time': datetime.now().isoformat(),
            'status': 'initialized',
            'total_spikes_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0
        }
        
        with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)
            
        return run_dir

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = os.path.join(self.run_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Configure logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Setup file handler for general logs
        file_handler = logging.FileHandler(os.path.join(log_dir, 'feature_extractor.log'))
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup file handler for debug logs
        debug_handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))

        # Configure logger
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler, debug_handler]
        )
        self.logger = logging.getLogger('FeatureExtractor')

    def extract_price_features(self, df: pd.DataFrame, config: FeatureConfig) -> Dict:
        """Extract enhanced price-related features"""
        features = {}
        
        try:
            # Basic price statistics
            features.update({
                'price_mean': df['value'].mean(),
                'price_std': df['value'].std(),
                'price_min': df['value'].min(),
                'price_max': df['value'].max(),
                'price_last': df['value'].iloc[-1]
            })
            
            # Technical indicators
            if len(df) >= 5:  # Ensure enough data for technical indicators
                # RSI for multiple windows
                for window in config.RSI_WINDOWS:
                    if len(df) >= window:
                        rsi = ta.momentum.RSIIndicator(df['value'], window=window)
                        features[f'rsi_{window}'] = rsi.rsi().iloc[-1]
                        rsi_values = rsi.rsi().dropna()
                        if len(rsi_values) >= 2:
                            features[f'rsi_{window}_slope'] = np.gradient(rsi_values).mean()

                # MACD
                macd = ta.trend.MACD(df['value'])
                features.update({
                    'macd': macd.macd().iloc[-1],
                    'macd_signal': macd.macd_signal().iloc[-1],
                    'macd_divergence': macd.macd_diff().iloc[-1]
                })

                # Bollinger Bands for multiple windows
                for window in config.BB_WINDOWS:
                    if len(df) >= window:
                        bb = ta.volatility.BollingerBands(df['value'], window=window)
                        bb_high = bb.bollinger_hband()
                        bb_low = bb.bollinger_lband()
                        current_price = df['value'].iloc[-1]
                        
                        features[f'bb_width_{window}'] = ((bb_high - bb_low) / current_price).iloc[-1]
                        width_diff = bb_high - bb_low
                        if width_diff.iloc[-1] != 0:
                            features[f'bb_position_{window}'] = ((current_price - bb_low.iloc[-1]) / 
                                                            width_diff.iloc[-1])

            # Price changes and volatility over different windows
            for window in config.PRICE_WINDOWS:
                if len(df) >= window:
                    window_data = df.iloc[-window:]
                    features.update(self._calculate_window_price_metrics(window_data, window))

        except Exception as e:
            self.logger.error(f"Error in price feature extraction: {str(e)}")
            self.logger.debug(f"Price data shape: {df.shape}")
            self.logger.debug(f"Price data head: {df.head().to_dict()}")
            self.logger.debug(traceback.format_exc())
            
        return features

    def _calculate_window_price_metrics(self, window_data: pd.DataFrame, window: int) -> Dict:
        """Calculate price metrics for a specific window"""
        metrics = {}
        try:
            if len(window_data) >= 2:
                # Calculate returns
                returns = window_data['value'].pct_change().dropna()
                
                # Basic price change
                start_price = window_data['value'].iloc[0]
                end_price = window_data['value'].iloc[-1]
                change = (end_price - start_price) / start_price if start_price != 0 else 0
                
                metrics[f'price_change_{window}m'] = change
                
                # Volatility metrics
                if len(returns) > 0:
                    returns_std = returns.std()
                    metrics.update({
                        f'volatility_{window}m': returns_std,
                        f'momentum_{window}m': returns.mean(),
                        f'momentum_std_{window}m': returns_std,
                        f'acceleration_{window}m': returns.diff().mean()
                    })
                    
                    # Trend metrics
                    if returns_std != 0:
                        metrics[f'trend_strength_{window}m'] = abs(returns.mean()) / returns_std
                        metrics[f'trend_consistency_{window}m'] = (np.sign(returns) == 
                                                                np.sign(returns.mean())).mean()
        except Exception as e:
            self.logger.error(f"Error calculating window metrics for {window}m: {str(e)}")
            
        return metrics

    def parse_trade_amount(self, trade_data: str) -> float:
        """Parse trade amount from JSON string or dict with improved error handling"""
        try:
            if isinstance(trade_data, str):
                trade_dict = ast.literal_eval(trade_data.replace('null', 'None'))
            else:
                trade_dict = trade_data
                
            amount = float(trade_dict.get('uiAmount', 0))
            # Handle potential negative values
            return abs(amount)
            
        except Exception as e:
            self.logger.debug(f"Error parsing trade amount: {str(e)}")
            self.logger.debug(f"Trade data: {trade_data[:200]}...")  # Log first 200 chars
            return 0.0

    def calculate_trade_volume(self, row: pd.Series) -> float:
        """Calculate trade volume with improved error handling"""
        try:
            # For a buy, use the 'to' amount, for a sell use the 'from' amount
            if row['side'] == 'buy':
                volume = self.parse_trade_amount(row['to'])
            else:
                volume = self.parse_trade_amount(row['from'])
                
            # Validate volume
            if pd.isna(volume) or volume < 0:
                self.logger.debug(f"Invalid volume calculated: {volume}")
                return 0.0
                
            return float(volume)
            
        except Exception as e:
            self.logger.debug(f"Error calculating trade volume: {str(e)}")
            self.logger.debug(f"Row data: {row.to_dict()}")
            return 0.0

    def preprocess_trades_data(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess trades data with comprehensive validation"""
        if trades_df is None or trades_df.empty:
            self.logger.warning("Empty trades DataFrame provided")
            return pd.DataFrame()
            
        try:
            trades_df = trades_df.copy()
            
            # Convert timestamp
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            # Calculate volumes
            self.logger.debug("Calculating trade volumes...")
            trades_df['volume'] = trades_df.apply(self.calculate_trade_volume, axis=1)
            
            # Remove invalid entries
            initial_count = len(trades_df)
            trades_df = trades_df[trades_df['volume'] > 0]
            removed_count = initial_count - len(trades_df)
            
            if removed_count > 0:
                self.logger.warning(f"Removed {removed_count} trades with invalid volumes")
            
            # Sort by timestamp
            trades_df = trades_df.sort_values('timestamp')
            
            # Add derived columns
            if 'side' not in trades_df.columns:
                trades_df['side'] = 'unknown'
            
            self.logger.debug(f"Preprocessed {len(trades_df)} trades successfully")
            return trades_df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing trades data: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def extract_volume_features(self, trades_df: pd.DataFrame, config: FeatureConfig) -> Dict:
        """Extract enhanced volume and trading features"""
        features = {}
        
        if trades_df is None or trades_df.empty:
            self.logger.warning("No trade data available for volume feature extraction")
            return self._get_default_volume_features()
            
        try:
            # Preprocess trades
            trades_df = self.preprocess_trades_data(trades_df)
            if trades_df.empty:
                return self._get_default_volume_features()
            
            # Basic volume statistics
            features.update(self._calculate_basic_volume_metrics(trades_df))
            
            # Whale detection
            large_trades = trades_df[
                trades_df['volume'] > trades_df['volume'].quantile(config.MIN_TRADE_SIZE_PERCENTILE)
            ]
            features.update({
                'whale_trade_count': len(large_trades),
                'whale_volume_ratio': (large_trades['volume'].sum() / trades_df['volume'].sum() 
                                     if trades_df['volume'].sum() > 0 else 0)
            })
            
            # Window-based analysis
            for window in config.VOLUME_WINDOWS:
                window_features = self._calculate_window_volume_metrics(
                    trades_df, window, config.MIN_TRADE_SIZE_PERCENTILE
                )
                features.update(window_features)
            
            # Add volume trends
            trend_features = self._calculate_volume_trends(trades_df)
            features.update(trend_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in volume feature extraction: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return self._get_default_volume_features()

    def _calculate_basic_volume_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate basic volume metrics with validation"""
        metrics = {}
        try:
            metrics.update({
                'volume_total': trades_df['volume'].sum(),
                'volume_mean': trades_df['volume'].mean(),
                'volume_std': trades_df['volume'].std(),
                'volume_skew': trades_df['volume'].skew(),
                'volume_kurtosis': trades_df['volume'].kurtosis()
            })
            
            # Add trade intervals
            if len(trades_df) > 1:
                intervals = trades_df['timestamp'].diff().dt.total_seconds()
                metrics.update({
                    'avg_trade_interval': intervals.mean(),
                    'trade_interval_std': intervals.std()
                })
                
        except Exception as e:
            self.logger.error(f"Error calculating basic volume metrics: {str(e)}")
            metrics = {
                'volume_total': 0,
                'volume_mean': 0,
                'volume_std': 0,
                'volume_skew': 0,
                'volume_kurtosis': 0,
                'avg_trade_interval': 0,
                'trade_interval_std': 0
            }
            
        return metrics

    def _calculate_window_volume_metrics(self, trades_df: pd.DataFrame, 
                                       window: int, whale_threshold: float) -> Dict:
        """Calculate volume metrics for specific time windows"""
        metrics = {}
        try:
            window_data = trades_df.iloc[-window:]
            if not window_data.empty:
                prefix = f'volume_{window}m'
                
                # Basic volume metrics
                metrics.update({
                    f'{prefix}': window_data['volume'].sum(),
                    f'trades_count_{window}m': len(window_data),
                    f'avg_trade_size_{window}m': window_data['volume'].mean(),
                    f'max_trade_size_{window}m': window_data['volume'].max()
                })
                
                # Buy/Sell analysis
                buys = window_data[window_data['side'] == 'buy']['volume'].sum()
                sells = window_data[window_data['side'] == 'sell']['volume'].sum()
                total = buys + sells
                
                if total > 0:
                    metrics.update({
                        f'buy_sell_ratio_{window}m': buys / sells if sells > 0 else float('inf'),
                        f'buy_sell_imbalance_{window}m': (buys - sells) / total
                    })
                
                # Trade size analysis
                large_trades = window_data[
                    window_data['volume'] >= window_data['volume'].quantile(whale_threshold)
                ]
                metrics[f'large_trade_ratio_{window}m'] = (
                    len(large_trades) / len(window_data) if len(window_data) > 0 else 0
                )
                
                # Volume momentum
                if len(window_data) > 1:
                    vol_changes = window_data['volume'].pct_change()
                    metrics.update({
                        f'volume_momentum_{window}m': vol_changes.mean(),
                        f'volume_momentum_std_{window}m': vol_changes.std(),
                        f'volume_acceleration_{window}m': vol_changes.diff().mean()
                    })
                
        except Exception as e:
            self.logger.error(f"Error calculating window metrics for {window}m: {str(e)}")
            self.logger.debug(f"Window data head: {trades_df.head().to_dict()}")
            
        return metrics

    def _calculate_volume_trends(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate volume trend metrics"""
        trends = {}
        try:
            if len(trades_df) > 1:
                # Calculate linear regression on volumes
                x = np.arange(len(trades_df))
                y = trades_df['volume'].values
                slope, intercept, r_value, _, _ = stats.linregress(x, y)
                
                trends.update({
                    'volume_trend_slope': slope,
                    'volume_trend_r2': r_value ** 2,
                    'volume_trend_strength': abs(slope) / trades_df['volume'].std() 
                        if trades_df['volume'].std() != 0 else 0
                })
                
                # Volume acceleration
                vol_changes = trades_df['volume'].pct_change()
                if len(vol_changes) > 1:
                    trends['volume_acceleration'] = vol_changes.diff().mean()
                
        except Exception as e:
            self.logger.error(f"Error calculating volume trends: {str(e)}")
            trends = {
                'volume_trend_slope': 0,
                'volume_trend_r2': 0,
                'volume_trend_strength': 0,
                'volume_acceleration': 0
            }
            
        return trends

    def get_default_volume_features(self) -> Dict:
        """Return default values for volume features when calculation fails"""
        return {
            'volume_total': 0,
            'volume_mean': 0,
            'volume_std': 0,
            'volume_skew': 0,
            'volume_kurtosis': 0,
            'whale_trade_count': 0,
            'whale_volume_ratio': 0
        }

    def extract_market_features(self, token_info: Dict) -> Dict:
        """Extract enhanced market structure features"""
        features = {}
        
        if not token_info:
            self.logger.warning("No token info available for market feature extraction")
            return self._get_default_market_features()
            
        try:
            # Market depth metrics
            features.update({
                'liquidity': token_info.get('liquidity', 0),
                'market_count': token_info.get('numberMarkets', 0),
                'holder_count': token_info.get('holder', 0),
                'unique_wallets_24h': token_info.get('uniqueWallet24h', 0)
            })
            
            # Calculate derived metrics
            if features['holder_count'] > 0:
                features.update({
                    'active_ratio': features['unique_wallets_24h'] / features['holder_count'],
                    'average_holding': token_info.get('totalSupply', 0) / features['holder_count']
                })
            
            # Trading activity metrics
            features['trades_24h'] = token_info.get('trade24h', 0)
            if features['holder_count'] > 0:
                features.update({
                    'trades_per_holder': features['trades_24h'] / features['holder_count'],
                    'volume_per_holder': token_info.get('volume24h', 0) / features['holder_count']
                })
            
            # Market maturity indicators
            if 'creationTime' in token_info:
                creation_time = pd.to_datetime(token_info['creationTime'])
                features['token_age_days'] = (pd.Timestamp.now() - creation_time).days
                
            # Market concentration metrics
            if 'top10HolderPercent' in token_info:
                features['holder_concentration'] = token_info['top10HolderPercent']
                
            # Volatility and momentum metrics
            if 'priceChange24h' in token_info:
                features['price_change_24h'] = token_info['priceChange24h']
            if 'volumeChange24h' in token_info:
                features['volume_change_24h'] = token_info['volumeChange24h']
                
            # Add validation flags
            features.update(self._validate_market_features(token_info))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting market features: {str(e)}")
            self.logger.debug(f"Token info: {json.dumps(token_info, indent=2)}")
            return self._get_default_market_features()

    def _validate_market_features(self, token_info: Dict) -> Dict:
        """Validate market data and add quality flags"""
        validation = {
            'has_valid_liquidity': token_info.get('liquidity', 0) > 0,
            'has_valid_holders': token_info.get('holder', 0) > 0,
            'has_recent_trades': token_info.get('lastTradeUnixTime', 0) > 
                              (datetime.now() - timedelta(hours=24)).timestamp(),
            'has_price_data': 'price' in token_info and token_info['price'] is not None
        }
        return validation

    def _get_default_market_features(self) -> Dict:
        """Return default values for market features"""
        return {
            'liquidity': 0,
            'market_count': 0,
            'holder_count': 0,
            'unique_wallets_24h': 0,
            'active_ratio': 0,
            'average_holding': 0,
            'trades_24h': 0,
            'trades_per_holder': 0,
            'volume_per_holder': 0,
            'token_age_days': 0,
            'holder_concentration': 0,
            'has_valid_liquidity': False,
            'has_valid_holders': False,
            'has_recent_trades': False,
            'has_price_data': False
        }

    def extract_pattern_features(self, df: pd.DataFrame, config: FeatureConfig) -> Dict:
        """Extract pattern-based features with validation"""
        features = {}
        
        try:
            if len(df) < 5:  # Minimum data points needed for pattern detection
                self.logger.warning("Insufficient data points for pattern detection")
                return self._get_default_pattern_features()
            
            # Price patterns
            returns = df['value'].pct_change().fillna(0)
            
            # Detect trend patterns
            for window in config.MICRO_WINDOWS:
                window_features = self._extract_window_patterns(returns, window)
                features.update(window_features)
            
            # Volatility patterns
            volatility_features = self._extract_volatility_patterns(df, returns, config)
            features.update(volatility_features)
            
            # Technical patterns
            tech_features = self._extract_technical_patterns(df, config)
            features.update(tech_features)
            
            # Ensure default values for required patterns
            required_patterns = {
                'trend_strength_3m': 0.0,
                'trend_consistency_3m': 0.0,
                'volatility_regime_5m': 'unknown'
            }
            
            for key, default_value in required_patterns.items():
                if key not in features:
                    features[key] = default_value
            
            # Add pattern validation
            features['patterns_detected'] = self._validate_patterns(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern features: {str(e)}")
            return self._get_default_pattern_features()

    def _extract_window_patterns(self, returns: pd.Series, window: int) -> Dict:
        """Extract pattern features for a specific window"""
        features = {}
        try:
            if len(returns) >= window:
                rolling_mean = returns.rolling(window=window).mean()
                rolling_std = returns.rolling(window=window).std()
                
                # Skip if not enough data
                if pd.isna(rolling_mean.iloc[-1]) or pd.isna(rolling_std.iloc[-1]):
                    return {}
                    
                # Trend strength and consistency
                if rolling_std.iloc[-1] != 0:
                    features[f'trend_strength_{window}m'] = (
                        abs(rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
                    )
                features[f'trend_consistency_{window}m'] = (
                    (np.sign(returns[-window:]) == np.sign(rolling_mean.iloc[-1])).mean()
                )
                
                # Momentum features
                if rolling_std.iloc[-1] != 0:
                    features[f'momentum_strength_{window}m'] = (
                        rolling_mean.iloc[-1] / rolling_std.iloc[-1]
                    )
                
                # Pattern identification
                features[f'pattern_type_{window}m'] = self._identify_pattern(
                    returns[-window:], rolling_mean.iloc[-1]
                )
                
        except Exception as e:
            self.logger.debug(f"Error in window pattern extraction: {str(e)}")
            
        return features

    def _extract_volatility_patterns(self, df: pd.DataFrame, returns: pd.Series, 
                                   config: FeatureConfig) -> Dict:
        """Extract volatility-based patterns"""
        features = {}
        try:
            for window in config.PRICE_WINDOWS:
                if len(returns) >= window:
                    volatility = returns.rolling(window=window).std()
                    if len(volatility) >= 2 and volatility.iloc[-2] != 0:
                        features[f'volatility_acceleration_{window}m'] = (
                            volatility.iloc[-1] / volatility.iloc[-2]
                        )
                    
                    # Volatility regimes
                    if not pd.isna(volatility.iloc[-1]):
                        features[f'volatility_regime_{window}m'] = self._classify_volatility(
                            volatility.iloc[-1], volatility.mean(), volatility.std()
                        )
                        
        except Exception as e:
            self.logger.debug(f"Error in volatility pattern extraction: {str(e)}")
            
        return features

    def _extract_technical_patterns(self, df: pd.DataFrame, config: FeatureConfig) -> Dict:
        """Extract technical analysis patterns"""
        features = {}
        try:
            # RSI divergence
            for window in config.RSI_WINDOWS:
                if len(df) >= window:
                    rsi = ta.momentum.RSIIndicator(df['value'], window=window).rsi()
                    if len(rsi) >= 2:
                        features[f'rsi_divergence_{window}'] = (
                            np.sign(df['value'].diff().iloc[-1]) != 
                            np.sign(rsi.diff().iloc[-1])
                        )
                        
            # MACD patterns
            if len(df) >= 26:  # Minimum length for MACD
                macd = ta.trend.MACD(df['value'])
                features.update({
                    'macd_cross': (
                        np.sign(macd.macd_diff().iloc[-1]) != 
                        np.sign(macd.macd_diff().iloc[-2])
                    ) if len(macd.macd_diff()) >= 2 else False,
                    'macd_trend': self._classify_trend(macd.macd_diff().iloc[-5:])
                })
                
        except Exception as e:
            self.logger.debug(f"Error in technical pattern extraction: {str(e)}")
            
        return features

    def _identify_pattern(self, returns: pd.Series, trend: float) -> str:
        """Identify price pattern type"""
        if abs(trend) < 0.001:  # Near zero trend
            return 'consolidation'
        elif trend > 0:
            return 'uptrend' if (returns > 0).mean() > 0.7 else 'choppy_up'
        else:
            return 'downtrend' if (returns < 0).mean() > 0.7 else 'choppy_down'

    def _classify_volatility(self, current: float, mean: float, std: float) -> str:
        """Classify volatility regime"""
        if current < mean - std:
            return 'low'
        elif current > mean + std:
            return 'high'
        return 'normal'

    def _classify_trend(self, values: pd.Series) -> str:
        """Classify trend direction"""
        if len(values) < 2:
            return 'unknown'
        mean_change = values.mean()
        if abs(mean_change) < 0.001:
            return 'sideways'
        return 'bullish' if mean_change > 0 else 'bearish'

    def _get_default_pattern_features(self) -> Dict:
        """Return default values for pattern features"""
        return {
            'patterns_detected': False,
            'trend_strength': 0,
            'trend_consistency': 0,
            'volatility_regime': 'unknown',
            'pattern_type': 'unknown'
        }
    
    def _validate_patterns(self, features: Dict) -> bool:
        """Validate extracted patterns"""
        try:
            # Check if required pattern features are present
            required_patterns = [
                'trend_strength_3m',
                'trend_consistency_3m',
                'volatility_regime_5m'
            ]
            
            # Check for presence and non-None values
            for pattern in required_patterns:
                if pattern not in features or features[pattern] is None:
                    return False
            return True
            
        except Exception as e:
            self.logger.debug(f"Error validating patterns: {str(e)}")
            return False

    def _serialize_features(self, features: Dict) -> Dict:
        """Convert numpy types and other non-serializable types to Python native types"""
        def serialize_value(value):
            try:
                if isinstance(value, (np.int8, np.int16, np.int32, np.int64)):
                    return int(value)
                elif isinstance(value, (np.float16, np.float32, np.float64)):
                    return float(value)
                elif isinstance(value, np.ndarray):
                    return value.tolist()
                elif isinstance(value, pd.Timestamp):
                    return value.isoformat()
                elif isinstance(value, datetime):
                    return value.isoformat()
                elif isinstance(value, (bool, np.bool_)):
                    return bool(value)
                elif pd.isna(value):
                    return None
                elif isinstance(value, dict):
                    return {k: serialize_value(v) for k, v in value.items()}
                elif isinstance(value, (list, tuple)):
                    return [serialize_value(item) for item in value]
                else:
                    return value
            except Exception as e:
                self.logger.warning(f"Error serializing value {value}: {str(e)}")
                return str(value)  # Fallback to string conversion

        try:
            serialized = {}
            for key, value in features.items():
                serialized[key] = serialize_value(value)
            return serialized
                
        except Exception as e:
            self.logger.error(f"Error in feature serialization: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return a stringified version as last resort
            return {k: str(v) for k, v in features.items()}

    def extract_microstructure_features(self, trades_df: pd.DataFrame, config: FeatureConfig) -> Dict:
        """Extract market microstructure features with improved validation"""
        features = {}
        
        if trades_df is None or trades_df.empty:
            self.logger.warning("No trades data for microstructure analysis")
            return self._get_default_microstructure_features()
            
        try:
            # Basic trade size analysis
            features.update(self._analyze_trade_sizes(trades_df))
            
            # Trade timing analysis
            features.update(self._analyze_trade_timing(trades_df))
            
            # Order flow analysis
            for window in config.MICRO_WINDOWS:
                features.update(self._analyze_order_flow(trades_df, window))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in microstructure feature extraction: {str(e)}")
            return self._get_default_microstructure_features()

    def _analyze_trade_sizes(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze trade size distributions"""
        features = {}
        try:
            trade_sizes = trades_df['volume']
            features.update({
                'trade_size_mean': trade_sizes.mean(),
                'trade_size_median': trade_sizes.median(),
                'trade_size_skew': trade_sizes.skew(),
                'trade_size_kurtosis': trade_sizes.kurtosis()
            })
            
            # Size percentiles
            for pct in [25, 75, 90, 95]:
                features[f'trade_size_percentile_{pct}'] = trade_sizes.quantile(pct/100)
                
            # Large trade analysis
            large_trades = trade_sizes[trade_sizes > trade_sizes.quantile(0.95)]
            features.update({
                'large_trade_count': len(large_trades),
                'large_trade_volume_ratio': large_trades.sum() / trade_sizes.sum() 
                    if trade_sizes.sum() > 0 else 0
            })
            
        except Exception as e:
            self.logger.debug(f"Error in trade size analysis: {str(e)}")
            
        return features

    def _analyze_trade_timing(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze trade timing patterns"""
        features = {}
        try:
            trade_times = pd.to_datetime(trades_df['timestamp'])
            intervals = trade_times.diff().dt.total_seconds()
            
            features.update({
                'avg_trade_interval': intervals.mean(),
                'trade_interval_std': intervals.std(),
                'trade_interval_skew': intervals.skew()
            })
            
            # Clustering analysis
            if len(intervals) > 1:
                # Detect trade clusters
                mean_interval = intervals.mean()
                std_interval = intervals.std()
                cluster_threshold = mean_interval - std_interval
                
                clusters = (intervals < cluster_threshold).astype(int)
                cluster_starts = (clusters.diff() == 1).sum()
                
                features.update({
                    'cluster_count': cluster_starts,
                    'clustering_ratio': cluster_starts / len(intervals) if len(intervals) > 0 else 0
                })
                
        except Exception as e:
            self.logger.debug(f"Error in trade timing analysis: {str(e)}")
            
        return features

    def _analyze_order_flow(self, trades_df: pd.DataFrame, window: int) -> Dict:
        """Analyze order flow imbalance"""
        features = {}
        prefix = f'order_flow_{window}m'
        
        try:
            window_trades = trades_df.iloc[-window:]
            if not window_trades.empty:
                # Buy/Sell imbalance
                buys = window_trades[window_trades['side'] == 'buy']['volume'].sum()
                sells = window_trades[window_trades['side'] == 'sell']['volume'].sum()
                total = buys + sells
                
                if total > 0:
                    features[f'{prefix}_imbalance'] = (buys - sells) / total
                    features[f'{prefix}_buy_ratio'] = buys / total
                    
                # Trade size patterns
                buy_sizes = window_trades[window_trades['side'] == 'buy']['volume']
                sell_sizes = window_trades[window_trades['side'] == 'sell']['volume']
                
                if not buy_sizes.empty and not sell_sizes.empty:
                    features.update({
                        f'{prefix}_buy_mean_size': buy_sizes.mean(),
                        f'{prefix}_sell_mean_size': sell_sizes.mean(),
                        f'{prefix}_size_imbalance': (
                            buy_sizes.mean() / sell_sizes.mean() if sell_sizes.mean() > 0 else 0
                        )
                    })
                    
        except Exception as e:
            self.logger.debug(f"Error in order flow analysis for window {window}: {str(e)}")
            
        return features

    def _get_default_microstructure_features(self) -> Dict:
        """Return default values for microstructure features"""
        return {
            'trade_size_mean': 0,
            'trade_size_median': 0,
            'trade_size_skew': 0,
            'trade_size_kurtosis': 0,
            'large_trade_count': 0,
            'large_trade_volume_ratio': 0,
            'avg_trade_interval': 0,
            'trade_interval_std': 0,
            'trade_interval_skew': 0,
            'cluster_count': 0,
            'clustering_ratio': 0
        }

    def extract_composite_features(self, price_df: pd.DataFrame, trades_df: pd.DataFrame, 
                                 token_info: Dict, config: FeatureConfig) -> Dict:
        """Extract composite features combining multiple signals"""
        features = {}
        
        try:
            if trades_df is not None and not trades_df.empty:
                # Volume-weighted price impact
                features.update(self._calculate_volume_weighted_metrics(price_df, trades_df))
                
                # Liquidity utilization
                if token_info and token_info.get('liquidity', 0) > 0:
                    features.update(self._calculate_liquidity_metrics(trades_df, token_info))
                
                # Market impact calculations
                features.update(self._calculate_market_impact(trades_df, token_info, config))
                
                # Buy pressure metrics
                features.update(self._calculate_buy_pressure(trades_df, price_df, config))
                
            # Combined technical indicators
            features.update(self._calculate_technical_combinations(price_df, config))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting composite features: {str(e)}")
            return self._get_default_composite_features()

    def _calculate_volume_weighted_metrics(self, price_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """Calculate volume-weighted metrics"""
        features = {}
        try:
            price_changes = price_df['value'].pct_change()
            volume_changes = trades_df['volume'].pct_change()
            
            # Volume-weighted momentum
            if len(price_changes) == len(volume_changes):
                features['volume_weighted_momentum'] = (price_changes * volume_changes).mean()
                
            # Volume-price correlation
            if len(price_changes) > 1 and len(volume_changes) > 1:
                correlation = np.corrcoef(
                    price_changes.fillna(0),
                    volume_changes.fillna(0)
                )[0, 1]
                features['price_volume_correlation'] = correlation
                
        except Exception as e:
            self.logger.debug(f"Error in volume-weighted metrics: {str(e)}")
            
        return features

    def _calculate_liquidity_metrics(self, trades_df: pd.DataFrame, token_info: Dict) -> Dict:
        """Calculate liquidity-based metrics"""
        features = {}
        try:
            liquidity = token_info.get('liquidity', 0)
            if liquidity > 0:
                total_volume = trades_df['volume'].sum()
                features.update({
                    'liquidity_utilization': total_volume / liquidity,
                    'average_impact': (trades_df['volume'] / liquidity).mean()
                })
                
                # Large trade impact
                large_trades = trades_df[
                    trades_df['volume'] >= trades_df['volume'].quantile(0.95)
                ]
                if not large_trades.empty:
                    features['large_trade_impact'] = (
                        large_trades['volume'].sum() / liquidity
                    )
                    
        except Exception as e:
            self.logger.debug(f"Error in liquidity metrics: {str(e)}")
            
        return features

    def _calculate_market_impact(self, trades_df: pd.DataFrame, 
                               token_info: Dict, config: FeatureConfig) -> Dict:
        """Calculate market impact metrics"""
        features = {}
        try:
            for window in config.MICRO_WINDOWS:
                window_volume = trades_df['volume'].iloc[-window:].sum()
                if token_info and token_info.get('liquidity', 0) > 0:
                    features[f'market_impact_{window}m'] = (
                        window_volume / token_info['liquidity']
                    )
                    
        except Exception as e:
            self.logger.debug(f"Error in market impact calculations: {str(e)}")
            
        return features

    def _calculate_buy_pressure(self, trades_df: pd.DataFrame, 
                              price_df: pd.DataFrame, config: FeatureConfig) -> Dict:
        """Calculate buy pressure metrics"""
        features = {}
        try:
            for window in config.VOLUME_WINDOWS:
                window_trades = trades_df.iloc[-window:]
                buys = window_trades[window_trades['side'] == 'buy']['volume'].sum()
                sells = window_trades[window_trades['side'] == 'sell']['volume'].sum()
                
                if sells > 0:
                    # Buy pressure relative to volatility
                    vol = price_df['value'].pct_change()[-window:].std()
                    if vol > 0:
                        features[f'buy_pressure_volatility_{window}m'] = (buys/sells) / vol
                    
                    # Buy pressure momentum
                    prev_window = trades_df.iloc[-2*window:-window]
                    prev_buys = prev_window[prev_window['side'] == 'buy']['volume'].sum()
                    prev_sells = prev_window[prev_window['side'] == 'sell']['volume'].sum()
                    if prev_sells > 0:
                        features[f'buy_pressure_momentum_{window}m'] = (
                            (buys/sells) / (prev_buys/prev_sells)
                        )
                        
        except Exception as e:
            self.logger.debug(f"Error in buy pressure calculations: {str(e)}")
            
        return features

    def _calculate_technical_combinations(self, price_df: pd.DataFrame, config: FeatureConfig) -> Dict:
        """Calculate combined technical indicators"""
        features = {}
        try:
            # Volatility-adjusted momentum
            for window in config.PRICE_WINDOWS:
                price_changes = price_df['value'].pct_change()[-window:]
                if len(price_changes) >= window:
                    vol = price_changes.std()
                    if vol > 0:
                        features[f'volatility_adjusted_momentum_{window}m'] = (
                            price_changes.mean() / vol
                        )
                        
            # RSI combinations
            for rsi_window in config.RSI_WINDOWS:
                rsi = ta.momentum.RSIIndicator(price_df['value'], window=rsi_window).rsi()
                if len(rsi) >= 5:
                    features[f'rsi_trend_strength_{rsi_window}'] = (
                        abs(rsi[-5:].mean() - 50) / rsi[-5:].std()
                        if rsi[-5:].std() > 0 else 0
                    )
                    
        except Exception as e:
            self.logger.debug(f"Error in technical combinations: {str(e)}")
            
        return features

    def _get_default_composite_features(self) -> Dict:
        """Return default values for composite features"""
        return {
            'volume_weighted_momentum': 0,
            'price_volume_correlation': 0,
            'liquidity_utilization': 0,
            'average_impact': 0,
            'large_trade_impact': 0
        }
    
    def extract_all_features(self, token_address: str, spike_time: datetime) -> Optional[Dict]:
        """Extract and combine all features for a given spike"""
        try:
            # Get the data run directory from progress info
            progress_file = os.path.join(self.run_dir, 'processing_progress.json')
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                data_run_dir = progress.get('data_run_dir')
            else:
                # Find the latest data run directory
                runs_dir = os.path.join(self.base_dir, "data", "runs")
                run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
                current_run = os.path.basename(self.run_dir)
                previous_runs = [d for d in run_dirs if d < current_run]
                data_run_dir = sorted(previous_runs)[-1] if previous_runs else None

            if not data_run_dir:
                self.logger.error("Could not find data run directory")
                return None

            spike_time_str = spike_time.strftime('%Y%m%d_%H%M%S')
            
            # Load data files from the data run directory
            price_file = os.path.join(self.base_dir, "data", "runs", data_run_dir, 
                                    'price_data', 
                                    f'{token_address}_pre_spike_{spike_time_str}_price.csv')
            trades_file = os.path.join(self.base_dir, "data", "runs", data_run_dir,
                                     'trades',
                                     f'{token_address}_pre_spike_{spike_time_str}_trades.csv')
            token_info_file = os.path.join(self.base_dir, "data", "runs", data_run_dir,
                                         'token_info',
                                         f'{token_address}_pre_spike_{spike_time_str}_token_info.json')

            # Validate price file exists
            if not os.path.exists(price_file):
                self.logger.error(f"Price file not found: {price_file}")
                return None

            # Load and validate price data
            price_df = pd.read_csv(price_file)
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            
            if price_df.empty:
                self.logger.error(f"Empty price data for {token_address}")
                return None

            # Load trades data
            trades_df = None
            if os.path.exists(trades_file):
                trades_df = pd.read_csv(trades_file)
                if not trades_df.empty:
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                    self.logger.info(f"Loaded {len(trades_df)} trades for {token_address}")

            # Load token info
            token_info = None
            if os.path.exists(token_info_file):
                with open(token_info_file, 'r') as f:
                    token_info = json.load(f)

            # Initialize feature config
            config = FeatureConfig()

            # Extract features
            features = {
                'token_address': token_address,
                'spike_time': spike_time.isoformat()
            }

            # Extract each feature group with detailed logging
            self.logger.debug(f"Extracting price features for {token_address}")
            price_features = self.extract_price_features(price_df, config)
            features.update(price_features)

            if trades_df is not None and not trades_df.empty:
                self.logger.debug(f"Extracting volume features for {token_address}")
                volume_features = self.extract_volume_features(trades_df, config)
                features.update(volume_features)

                # Extract microstructure features
                microstructure_features = self.extract_microstructure_features(trades_df, config)
                features.update(microstructure_features)

            self.logger.debug(f"Extracting market features for {token_address}")
            market_features = self.extract_market_features(token_info)
            features.update(market_features)

            # Extract pattern features
            pattern_features = self.extract_pattern_features(price_df, config)
            features.update(pattern_features)

            # Extract composite features
            composite_features = self.extract_composite_features(
                price_df, trades_df, token_info, config
            )
            features.update(composite_features)

            # Add metadata
            features.update({
                'data_points_count': len(price_df),
                'trades_count': len(trades_df) if trades_df is not None else 0,
                'feature_extraction_time': datetime.now().isoformat(),
                'data_run_dir': data_run_dir
            })

            # Serialize features
            serialized_features = self._serialize_features(features)
            
            # Save individual feature file
            feature_file = os.path.join(
                self.features_dir, 
                'individual_features',
                f'{token_address}_spike_{spike_time_str}_features.json'
            )
            
            with open(feature_file, 'w') as f:
                json.dump(serialized_features, f, indent=2)

            return features

        except Exception as e:
            self.logger.error(f"Error in extract_all_features for {token_address}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def select_features(self, features_df: pd.DataFrame, target: str, config: FeatureConfig) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features using statistical tests and correlation analysis"""
        self.logger.info("Starting feature selection process...")
        
        try:
            # Store metadata columns
            metadata_columns = ['token_address', 'spike_time']
            metadata = features_df[metadata_columns].copy()
            
            # Get numeric columns only
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns 
                             if col not in metadata_columns + [target]]
            
            if not numeric_columns:
                self.logger.error("No numeric features found for selection")
                return features_df, []
            
            # Prepare numeric data
            numeric_df = features_df[numeric_columns].copy()
            numeric_df = numeric_df.fillna(0)
            
            # Calculate correlation with target
            target_correlations = abs(numeric_df.corrwith(features_df[target]))
            self.logger.info(f"Calculated correlations with target: {target}")
            
            # First selection based on correlation with target
            correlation_threshold = 0.1
            correlated_features = target_correlations[
                target_correlations > correlation_threshold
            ].index.tolist()
            
            # Remove highly intercorrelated features
            correlation_matrix = numeric_df[correlated_features].corr().abs()
            upper = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Track removed features
            removed_features = []
            to_drop = set()
            
            # For each pair of correlated features
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    val = correlation_matrix.iloc[i, j]
                    if val > config.MAX_CORRELATION:
                        feat_i = correlation_matrix.columns[i]
                        feat_j = correlation_matrix.columns[j]
                        if target_correlations[feat_i] < target_correlations[feat_j]:
                            to_drop.add(feat_i)
                            removed_features.append({
                                'removed': feat_i,
                                'kept': feat_j,
                                'correlation': val,
                                'reason': 'high_correlation'
                            })
                        else:
                            to_drop.add(feat_j)
                            removed_features.append({
                                'removed': feat_j,
                                'kept': feat_i,
                                'correlation': val,
                                'reason': 'high_correlation'
                            })
            
            # Get selected features
            selected_features = [f for f in correlated_features if f not in to_drop]
            
            # Ensure key features are included
            key_features = [
                'volume_momentum_5m',
                'volatility_3m',
                'price_change_15m',
                'rsi_14',
                'macd',
                'volume_trend_strength'
            ]
            
            for feature in key_features:
                if feature in numeric_columns and feature not in selected_features:
                    selected_features.append(feature)
                    self.logger.info(f"Added key feature: {feature}")
            
            # Sort features by importance
            feature_importance = target_correlations[selected_features]
            selected_features = feature_importance.sort_values(ascending=False).index.tolist()
            
            # Create final DataFrame
            result_df = pd.concat([
                metadata,
                features_df[selected_features],
                features_df[target]
            ], axis=1)
            
            # Log selection results
            self.logger.info(f"Selected {len(selected_features)} features")
            self._save_feature_selection_report(selected_features, removed_features, 
                                             target_correlations)
            
            return result_df, selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            self.logger.error(traceback.format_exc())
            return features_df, list(features_df.columns)

    def _save_feature_selection_report(self, selected_features: List[str], 
                                     removed_features: List[Dict],
                                     correlations: pd.Series):
        """Save detailed feature selection report"""
        report_path = os.path.join(self.features_dir, 'feature_selection_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Feature Selection Report\n\n")
            
            # Selected Features
            f.write("## Selected Features\n")
            for feature in selected_features:
                f.write(f"- {feature} (correlation: {correlations[feature]:.3f})\n")
            f.write("\n")
            
            # Removed Features
            f.write("## Removed Features\n")
            for removal in removed_features:
                f.write(f"- {removal['removed']} (corr with {removal['kept']}: "
                       f"{removal['correlation']:.3f}, reason: {removal['reason']})\n")


    def _save_progress(self, progress: Dict):
        """Save processing progress"""
        progress_file = os.path.join(self.run_dir, 'processing_progress.json')
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def _update_run_status(self, status: str, progress: Dict, error: str = None):
        """Update run status information"""
        run_info_file = os.path.join(self.run_dir, 'run_info.json')
        
        run_info = {
            'status': status,
            'end_time': datetime.now().isoformat(),
            'total_processed': progress['processed'],
            'successful': progress['successful'],
            'failed': progress['failed'],
            'error': error
        }
        
        with open(run_info_file, 'w') as f:
            json.dump(run_info, f, indent=2)

    def process_all_spikes(self):
        """Process all spikes and perform feature extraction"""
        # Get the previous run directory that contains the spikes data
        runs_dir = os.path.join(self.base_dir, "data", "runs")
        run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
        if not run_dirs:
            self.logger.error("No run directories found")
            return
            
        # Sort by timestamp to get the latest run before current
        current_run = os.path.basename(self.run_dir)
        previous_runs = [d for d in run_dirs if d < current_run]
        if not previous_runs:
            self.logger.error("No previous run directories found")
            return
            
        latest_data_run = sorted(previous_runs)[-1]
        
        # Load spikes file from the latest data run
        spikes_file = os.path.join(self.base_dir, "data", "runs", latest_data_run, '5x_price_increases.csv')
        self.logger.info(f"Loading spikes from: {spikes_file}")
        
        if not os.path.exists(spikes_file):
            self.logger.error(f"Spikes file not found: {spikes_file}")
            return
            
        try:
            # Load and process spikes
            spikes_df = pd.read_csv(spikes_file)
            spikes_df['spike_time'] = pd.to_datetime(spikes_df['spike_time'])
            
            total_spikes = len(spikes_df)
            self.logger.info(f"Processing {total_spikes} spikes")
            
            # Initialize progress tracking
            progress = {
                'total_spikes': total_spikes,
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'start_time': datetime.now().isoformat(),
                'data_run_dir': latest_data_run
            }
            
            # Save initial progress
            self._save_progress(progress)
            
            # Process spikes
            all_features = []
            failed_spikes = []
            
            for idx, spike in spikes_df.iterrows():
                try:
                    self.logger.info(f"Processing spike {idx + 1}/{total_spikes} "
                                   f"for {spike['token_address']} at {spike['spike_time']}")
                    
                    # Update path for loading data files to use previous run directory
                    spike_time_str = pd.to_datetime(spike['spike_time']).strftime('%Y%m%d_%H%M%S')
                    
                    # Load price data from previous run
                    price_file = os.path.join(self.base_dir, "data", "runs", latest_data_run, 
                                            'price_data', 
                                            f"{spike['token_address']}_pre_spike_{spike_time_str}_price.csv")
                                            
                    trades_file = os.path.join(self.base_dir, "data", "runs", latest_data_run,
                                             'trades',
                                             f"{spike['token_address']}_pre_spike_{spike_time_str}_trades.csv")
                                             
                    token_info_file = os.path.join(self.base_dir, "data", "runs", latest_data_run,
                                                 'token_info',
                                                 f"{spike['token_address']}_pre_spike_{spike_time_str}_token_info.json")
                    
                    # Extract features using data from previous run
                    features = self.extract_all_features(spike['token_address'], 
                                                       pd.to_datetime(spike['spike_time']))
                    
                    if features:
                        all_features.append(features)
                        progress['successful'] += 1
                    else:
                        failed_spikes.append({
                            'token_address': spike['token_address'],
                            'spike_time': spike['spike_time'],
                            'error': 'Feature extraction failed'
                        })
                        progress['failed'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing spike {idx}: {str(e)}")
                    failed_spikes.append({
                        'token_address': spike['token_address'],
                        'spike_time': spike['spike_time'],
                        'error': str(e)
                    })
                    progress['failed'] += 1
                
                progress['processed'] = idx + 1
                self._save_progress(progress)
            
            # Create final results
            if all_features:
                self._save_results(all_features, failed_spikes)
            
            # Update run status
            self._update_run_status('completed', progress)
            
        except Exception as e:
            self.logger.error(f"Error in spike processing: {str(e)}")
            self.logger.error(traceback.format_exc())
            self._update_run_status('failed', progress, error=str(e))

    def _save_results(self, all_features: List[Dict], failed_spikes: List[Dict]):
        """Save extracted features and analysis results"""
        try:
            # Create DataFrame from all features
            features_df = pd.DataFrame(all_features)
            
            # Save raw features
            raw_features_file = os.path.join(self.features_dir, 'raw_features.csv')
            features_df.to_csv(raw_features_file, index=False)
            self.logger.info(f"Saved raw features to {raw_features_file}")

            # Calculate feature statistics
            numeric_features = features_df.select_dtypes(include=[np.number]).columns
            feature_stats = features_df[numeric_features].describe()
            stats_file = os.path.join(self.features_dir, 'feature_statistics.csv')
            feature_stats.to_csv(stats_file)
            self.logger.info(f"Saved feature statistics to {stats_file}")

            # Calculate feature correlations
            correlations = features_df[numeric_features].corr()
            corr_file = os.path.join(self.features_dir, 'feature_correlations.csv')
            correlations.to_csv(corr_file)
            self.logger.info(f"Saved feature correlations to {corr_file}")

            # Generate correlation heatmap
            plt.figure(figsize=(20, 16))
            sns.heatmap(correlations, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlations Heatmap')
            plt.tight_layout()
            heatmap_file = os.path.join(self.run_dir, 'correlation_heatmap.png')
            plt.savefig(heatmap_file)
            plt.close()
            self.logger.info(f"Saved correlation heatmap to {heatmap_file}")

            # Identify feature patterns
            patterns = self._analyze_feature_patterns(features_df, numeric_features)
            patterns_file = os.path.join(self.features_dir, 'feature_patterns.json')
            with open(patterns_file, 'w') as f:
                json.dump(patterns, f, indent=2)
            self.logger.info(f"Saved feature patterns to {patterns_file}")

            # Save selected features
            # Assuming price_change_15m is our target variable for feature selection
            target = 'price_change_15m'
            if target in features_df.columns:
                selected_df, selected_features = self.select_features(features_df, target, FeatureConfig())
                selected_file = os.path.join(self.features_dir, 'selected_features.csv')
                selected_df.to_csv(selected_file, index=False)
                
                # Save feature metadata
                metadata = {
                    'selected_features': selected_features,
                    'feature_count': len(selected_features),
                    'total_features': len(features_df.columns),
                    'numeric_features': len(numeric_features),
                    'timestamp': datetime.now().isoformat()
                }
                metadata_file = os.path.join(self.features_dir, 'feature_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(f"Saved selected features to {selected_file}")

            # Save failed spikes
            if failed_spikes:
                failed_file = os.path.join(self.features_dir, 'failed_spikes.json')
                with open(failed_file, 'w') as f:
                    json.dump(failed_spikes, f, indent=2)
                self.logger.info(f"Saved failed spikes to {failed_file}")

            # Generate analysis report
            self._generate_analysis_report(features_df, patterns, failed_spikes)

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _analyze_feature_patterns(self, df: pd.DataFrame, numeric_features: pd.Index) -> Dict:
        """Analyze patterns in features"""
        patterns = {
            'high_correlation_pairs': [],
            'stable_features': [],
            'volatile_features': [],
            'key_statistics': {}
        }

        try:
            # Find highly correlated feature pairs
            correlations = df[numeric_features].corr()
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    correlation = correlations.iloc[i, j]
                    if abs(correlation) > 0.7:  # Threshold for high correlation
                        patterns['high_correlation_pairs'].append({
                            'feature1': str(correlations.columns[i]),
                            'feature2': str(correlations.columns[j]),
                            'correlation': float(correlation)  # Convert to native float
                        })

            # Identify stable and volatile features
            for feature in numeric_features:
                std = float(df[feature].std())  # Convert to native float
                mean = float(df[feature].mean())  # Convert to native float
                if mean != 0:
                    cv = abs(std / mean)  # Coefficient of variation
                    if cv < 0.1:  # Threshold for stability
                        patterns['stable_features'].append(str(feature))
                    elif cv > 0.5:  # Threshold for volatility
                        patterns['volatile_features'].append(str(feature))

            # Calculate key statistics for each feature
            for feature in numeric_features:
                feature_stats = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'correlation_with_target': float(correlations.get('price_change_15m', {}).get(feature, 0))
                    if 'price_change_15m' in correlations else None
                }
                # Convert NaN to None for JSON serialization
                feature_stats = {k: None if pd.isna(v) else v 
                                 for k, v in feature_stats.items()}
                patterns['key_statistics'][str(feature)] = feature_stats

        except Exception as e:
            self.logger.error(f"Error analyzing feature patterns: {str(e)}")
            self.logger.debug(traceback.format_exc())

        return patterns

    def _generate_analysis_report(self, df: pd.DataFrame, patterns: Dict, failed_spikes: List[Dict]):
        """Generate a comprehensive analysis report"""
        report_path = os.path.join(self.features_dir, 'analysis_report.md')
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Feature Analysis Report\n\n")

                # Run Summary
                f.write("## Run Summary\n\n")
                f.write(f"- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- Total Spikes Analyzed: {len(df)}\n")
                f.write(f"- Unique Tokens: {df['token_address'].nunique()}\n")
                f.write(f"- Selected Features: {len(patterns['key_statistics'])}\n\n")

                # Feature Selection Results
                f.write("## Feature Selection Results\n\n")
                f.write(f"- Total Features Extracted: {len(df.columns)}\n")
                f.write(f"- Features Selected: {len(patterns['key_statistics'])}\n")
                f.write("### Selected Features:\n")
                for feature in patterns['key_statistics'].keys():
                    f.write(f"- {feature}\n")
                f.write("\n")

                # Feature Patterns
                f.write("## Feature Patterns\n\n")
                f.write(f"- Highly Correlated Pairs: {len(patterns['high_correlation_pairs'])}\n")
                f.write(f"- Stable Features: {len(patterns['stable_features'])}\n")
                f.write(f"- Volatile Features: {len(patterns['volatile_features'])}\n\n")

                # Key Statistics
                f.write("## Key Statistics for Selected Features\n\n")
                for feature, stats in patterns['key_statistics'].items():
                    f.write(f"### {feature}\n")
                    for stat_name, stat_value in stats.items():
                        if stat_value is not None:  # Only write if value exists
                            if isinstance(stat_value, (int, float)):
                                # Format numbers to 4 decimal places if they're numbers
                                f.write(f"- {stat_name}: {stat_value:.4f}\n")
                            else:
                                # For non-numeric values, just convert to string
                                f.write(f"- {stat_name}: {str(stat_value)}\n")
                        else:
                            # Handle None values
                            f.write(f"- {stat_name}: nan\n")
                    f.write("\n")

                # Add failed spikes summary if any
                if failed_spikes:
                    f.write("## Failed Spikes Summary\n\n")
                    f.write(f"Total Failed: {len(failed_spikes)}\n\n")
                    for fail in failed_spikes[:5]:  # Show first 5 failures
                        f.write(f"- Token: {fail.get('token_address', 'Unknown')}\n")
                        f.write(f"  Time: {fail.get('spike_time', 'Unknown')}\n")
                        f.write(f"  Error: {fail.get('error', 'Unknown error')}\n\n")
                    if len(failed_spikes) > 5:
                        f.write(f"... and {len(failed_spikes) - 5} more failures\n")

                self.logger.info(f"Generated analysis report at {report_path}")

        except Exception as e:
            self.logger.error(f"Error generating analysis report: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

def main():
    """Main entry point"""
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(base_dir, 'data', 'runs')

    # Get all run directories
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith('run_')]
    if not run_dirs:
        print("No run directories found")
        return

    # Sort run directories by timestamp
    sorted_runs = sorted(run_dirs)
    
    # Find the last run directory that contains the 5x_price_increases.csv file
    data_run_dir = None
    for run_dir in reversed(sorted_runs):
        spikes_file = os.path.join(runs_dir, run_dir, '5x_price_increases.csv')
        if os.path.exists(spikes_file):
            data_run_dir = run_dir
            break

    if not data_run_dir:
        print("Could not find any run directory with 5x_price_increases.csv")
        return

    print(f"Using data from run directory: {data_run_dir}")

    # Create a new run directory for feature extraction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_dir = os.path.join(runs_dir, f'run_{timestamp}')
    os.makedirs(new_run_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ['features', 'logs', 'trades', 'price_data', 'token_info']:
        os.makedirs(os.path.join(new_run_dir, os.path.join(new_run_dir, subdir)), exist_ok=True)
    
    # Copy the spikes file from the data run directory to the new run directory
    src_file = os.path.join(runs_dir, data_run_dir, '5x_price_increases.csv')
    dst_file = os.path.join(new_run_dir, '5x_price_increases.csv')
    if os.path.exists(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"Copied spikes file from {src_file} to {dst_file}")
    else:
        print(f"Warning: Spikes file not found at {src_file}")
        return
    
    # Initialize and run feature extractor
    extractor = FeatureExtractor(base_dir)
    extractor.process_all_spikes()

if __name__ == "__main__":
    main()
