from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import json
from typing import List, Dict, Tuple, Any, Optional
import aiohttp
import asyncio
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import traceback
warnings.filterwarnings('ignore')

@dataclass
class ProcessingStats:
    total_spikes: int
    processed_spikes: int
    failed_spikes: int
    start_time: datetime
    end_time: datetime
    unique_tokens: int
    feature_count: int
    
class FeatureExtractor:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = self._setup_logging()
        
        # Import config for API key
        sys.path.append(base_dir)
        from config import API_KEY
        self.api_key = API_KEY
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        self.base_url = 'https://public-api.birdeye.so'
        
        # Feature extraction settings
        self.time_windows = [3, 5, 10, 15]  # minutes
        self.price_metrics = ['mean', 'std', 'min', 'max', 'last']
        self.volume_windows = [1, 3, 5, 10, 15]  # minutes
        
        # Rate limiting
        self.DELAY_BETWEEN_REQUESTS = 0.1

    def _setup_logging(self):
        logger = logging.getLogger('FeatureExtractor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    async def get_price_history(self, token_address: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get historical price data for a token"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_ts = int(start_time.timestamp())
                end_ts = int(end_time.timestamp())
                
                url = f"{self.base_url}/defi/history_price"
                params = {
                    "address": token_address,
                    "address_type": "token",
                    "type": "1m",
                    "time_from": start_ts,
                    "time_to": end_ts
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers, params=params, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('success') and data.get('data', {}).get('items'):
                                df = pd.DataFrame(data['data']['items'])
                                df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
                                return df
                        elif response.status == 429:  # Rate limit
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            self.logger.warning(f"API returned status {response.status} for {token_address}")
                            
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout on attempt {attempt + 1} for {token_address}")
                await asyncio.sleep(retry_delay)
            except Exception as e:
                self.logger.error(f"Error fetching price history for {token_address}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    break
                
        return pd.DataFrame()

    async def get_trade_history(self, token_address: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get trading history for a token"""
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        
        url = f"{self.base_url}/defi/txs/token"
        params = {
            "address": token_address,
            "tx_type": "swap",
            "time_from": start_ts,
            "time_to": end_ts
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and data.get('data', {}).get('items'):
                            trades_df = pd.DataFrame(data['data']['items'])
                            # Debug logging
                            self.logger.info(f"Trade data columns: {trades_df.columns.tolist()}")
                            if not trades_df.empty:
                                self.logger.info(f"Sample trade data: {trades_df.iloc[0].to_dict()}")
                            return trades_df
                        else:
                            self.logger.warning(f"No trade data found for {token_address}")
                    else:
                        self.logger.warning(f"API returned status {response.status} for {token_address}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching trade history for {token_address}: {str(e)}")
            return pd.DataFrame()

    async def get_token_info(self, token_address: str) -> Dict:
        """Get token overview data"""
        url = f"{self.base_url}/defi/token_overview"
        params = {"address": token_address}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success'):
                            return data.get('data', {})
        except Exception as e:
            self.logger.error(f"Error fetching token info for {token_address}: {str(e)}")
        return {}
    
    def verify_data_quality(self, features_df: pd.DataFrame) -> bool:
        """Verify the quality of extracted features"""
        if features_df.empty:
            self.logger.error("Empty features DataFrame")
            return False
            
        # Check for required columns
        required_cols = ['token_address', 'spike_time', 'future_price_increase']
        missing_cols = [col for col in required_cols if col not in features_df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for infinite values
        inf_cols = features_df.columns[features_df.isin([np.inf, -np.inf]).any()]
        if not inf_cols.empty:
            self.logger.warning(f"Columns with infinity values: {inf_cols.tolist()}")
            
        # Check for excessive missing values
        missing_pct = features_df.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            self.logger.warning(f"Columns with >50% missing values: {high_missing.index.tolist()}")
            
        # Check date range
        date_range = pd.to_datetime(features_df['spike_time'])
        self.logger.info(f"Date range: {date_range.min()} to {date_range.max()}")
        
        # Check unique tokens
        unique_tokens = features_df['token_address'].nunique()
        self.logger.info(f"Unique tokens: {unique_tokens}")
        
        # Verify numeric columns don't have string values
        numeric_issues = []
        for col in features_df.select_dtypes(include=[np.number]).columns:
            if features_df[col].apply(lambda x: isinstance(x, str)).any():
                numeric_issues.append(col)
        if numeric_issues:
            self.logger.warning(f"Numeric columns containing strings: {numeric_issues}")
        
        # Check for duplicate spikes
        dupes = features_df.duplicated(['token_address', 'spike_time'])
        if dupes.any():
            self.logger.warning(f"Found {dupes.sum()} duplicate spikes")
            
        # Basic reasonableness checks
        if 'future_price_increase' in features_df.columns:
            unreasonable = features_df['future_price_increase'] > 1000  # 1000x seems unreasonable
            if unreasonable.any():
                self.logger.warning(f"Found {unreasonable.sum()} spikes with >1000x price increase")
        
        return True

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators for pre-spike window"""
        features = {}
        
        if df.empty:
            return {}
        
        df = df.sort_values('timestamp')
        price_data = df['value']
        
        # RSI with trend analysis
        for period in [7, 14, 21]:
            delta = price_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if len(rsi) >= period:
                features[f'rsi_{period}'] = rsi.iloc[-1]
                features[f'rsi_{period}_slope'] = (rsi.iloc[-1] - rsi.iloc[-2]) if len(rsi) > 1 else 0
                
                # RSI trend strength
                rsi_trend = rsi.iloc[-period:].diff().mean()
                features[f'rsi_trend_strength_{period}'] = abs(rsi_trend)
        
        # Additional indicators
        if len(price_data) >= 26:  # Minimum length for MACD
            exp1 = price_data.ewm(span=12, adjust=False).mean()
            exp2 = price_data.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            features.update({
                'macd': macd.iloc[-1],
                'macd_signal': signal.iloc[-1],
                'macd_divergence': macd.iloc[-1] - signal.iloc[-1]
            })
        
        return features
    
    def get_default_volume_features(self) -> Dict[str, float]:
        """Return default values for all volume features"""
        default_features = {
            'volume_total': 0,
            'volume_mean': 0,
            'volume_std': 0,
            'volume_skew': 0,
            'volume_kurtosis': 0,
            'avg_trade_interval': 0,
            'trade_interval_std': 0,
            'trade_interval_skew': 0,
            'whale_trade_count': 0,
            'whale_volume_ratio': 0
        }
        
        # Add window-based defaults
        for window in [1, 3, 5, 10, 15]:
            default_features.update({
                f'volume_{window}m': 0,
                f'trades_count_{window}m': 0,
                f'avg_trade_size_{window}m': 0,
                f'max_trade_size_{window}m': 0,
                f'buy_sell_ratio_{window}m': 1,
                f'buy_sell_imbalance_{window}m': 0,
                f'large_trade_ratio_{window}m': 0,
                f'volume_momentum_{window}m': 0,
                f'volume_momentum_std_{window}m': 0,
                f'volume_acceleration_{window}m': 0
            })
        
        return default_features

    def calculate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-related features for pre-spike window"""
        features = {}
        
        if df.empty:
            return {f"{feat}_{window}m": 0 for window in [3, 5, 10, 15] 
                    for feat in ['price_change', 'volatility', 'momentum', 'momentum_std', 
                            'acceleration', 'trend_strength', 'trend_consistency']}
        
        # Ensure data is sorted by time
        df = df.sort_values('timestamp')
        
        # Basic price metrics for entire 15m window
        price_data = df['value']
        features.update({
            'price_mean': price_data.mean(),
            'price_std': price_data.std(),
            'price_min': price_data.min(),
            'price_max': price_data.max(),
            'price_last': price_data.iloc[-1]  # Price right before spike
        })
        
        # Calculate features for different time windows
        windows = [3, 5, 10, 15]  # minutes
        for window in windows:
            # Get window data from end (closest to spike)
            window_data = df.tail(window)
            if len(window_data) < 2:
                continue
            
            # Price changes
            start_price = window_data['value'].iloc[0]
            end_price = window_data['value'].iloc[-1]
            price_change = (end_price - start_price) / start_price
            
            # Calculate returns and volatility
            returns = window_data['value'].pct_change().dropna()
            volatility = returns.std()
            
            # Momentum metrics
            momentum = returns.mean()
            momentum_std = returns.std()
            
            # Acceleration (change in momentum)
            if len(returns) >= 4:
                first_half = returns[:len(returns)//2].mean()
                second_half = returns[len(returns)//2:].mean()
                acceleration = second_half - first_half
            else:
                acceleration = 0
                
            # Trend metrics
            trend_strength = abs(price_change) / volatility if volatility > 0 else 0
            up_moves = (returns > 0).sum()
            trend_consistency = up_moves / len(returns) if len(returns) > 0 else 0
            
            # Pattern detection
            features.update({
                f'price_change_{window}m': price_change,
                f'volatility_{window}m': volatility,
                f'momentum_{window}m': momentum,
                f'momentum_std_{window}m': momentum_std,
                f'acceleration_{window}m': acceleration,
                f'trend_strength_{window}m': trend_strength,
                f'trend_consistency_{window}m': trend_consistency,
                
                # Additional pattern metrics
                f'momentum_strength_{window}m': abs(momentum) / volatility if volatility > 0 else 0,
                f'pattern_type_{window}m': 'uptrend' if price_change > 0 and trend_consistency > 0.6 else 'neutral'
            })
            
            # Volatility regimes
            volatility_acceleration = volatility / df['value'].pct_change().std() if len(df) > window else 1
            features[f'volatility_acceleration_{window}m'] = volatility_acceleration
            features[f'volatility_regime_{window}m'] = 'normal'
            if volatility_acceleration > 1.5:
                features[f'volatility_regime_{window}m'] = 'high'
            elif volatility_acceleration < 0.5:
                features[f'volatility_regime_{window}m'] = 'low'
                
            # Risk-adjusted metrics
            features[f'volatility_adjusted_momentum_{window}m'] = momentum / volatility if volatility > 0 else 0
        
        return features

    def calculate_volume_features(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-related features for pre-spike window"""
        features = {}
        
        if trades_df.empty:
            return self.get_default_volume_features()
        
        # Ensure trades are sorted by time
        if 'timestamp' not in trades_df.columns and 'blockUnixTime' in trades_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['blockUnixTime'], unit='s')
        trades_df = trades_df.sort_values('timestamp')
        
        # Determine volume field
        volume_field = None
        for field in ['amount', 'baseAmount', 'quoteAmount', 'value']:
            if field in trades_df.columns:
                volume_field = field
                break
        
        if volume_field is None:
            self.logger.warning("No volume field found, checking nested data")
            # Try to extract from nested structure
            if 'from' in trades_df.columns:
                trades_df['amount'] = trades_df.apply(
                    lambda x: float(x['from'].get('amount', 0)) if isinstance(x['from'], dict) else 0,
                    axis=1
                )
                volume_field = 'amount'
        
        if volume_field is None:
            return self.get_default_volume_features()
        
        # Basic volume metrics
        volume_data = trades_df[volume_field].astype(float)
        features.update({
            'volume_total': volume_data.sum(),
            'volume_mean': volume_data.mean(),
            'volume_std': volume_data.std(),
            'volume_skew': volume_data.skew(),
            'volume_kurtosis': volume_data.kurtosis(),
        })
        
        # Trade intervals
        if len(trades_df) > 1:
            intervals = trades_df['timestamp'].diff().dt.total_seconds()
            features.update({
                'avg_trade_interval': intervals.mean(),
                'trade_interval_std': intervals.std(),
                'trade_interval_skew': intervals.skew(),
            })
        
        # Whale analysis
        volume_95th = volume_data.quantile(0.95)
        whale_trades = trades_df[volume_data >= volume_95th]
        features.update({
            'whale_trade_count': len(whale_trades),
            'whale_volume_ratio': whale_trades[volume_field].sum() / volume_data.sum() if volume_data.sum() > 0 else 0
        })
        
        # Analyze different time windows
        for window in [1, 3, 5, 10, 15]:  # minutes
            window_end = trades_df['timestamp'].max()
            window_start = window_end - pd.Timedelta(minutes=window)
            window_trades = trades_df[trades_df['timestamp'] >= window_start]
            
            if len(window_trades) == 0:
                continue
                
            window_volume = window_trades[volume_field].astype(float)
            
            features.update({
                f'volume_{window}m': window_volume.sum(),
                f'trades_count_{window}m': len(window_trades),
                f'avg_trade_size_{window}m': window_volume.mean(),
                f'max_trade_size_{window}m': window_volume.max(),
            })
            
            # Buy/Sell analysis
            side_field = next((f for f in ['side', 'type', 'tradeType'] if f in window_trades.columns), None)
            if side_field:
                buys = window_trades[window_trades[side_field].str.lower().str.contains('buy', na=False)]
                sells = window_trades[window_trades[side_field].str.lower().str.contains('sell', na=False)]
                
                buy_volume = buys[volume_field].astype(float).sum()
                sell_volume = sells[volume_field].astype(float).sum()
                total_volume = buy_volume + sell_volume
                
                features.update({
                    f'buy_sell_ratio_{window}m': buy_volume / sell_volume if sell_volume > 0 else float('inf'),
                    f'buy_sell_imbalance_{window}m': (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0,
                    f'large_trade_ratio_{window}m': (window_volume >= volume_95th).mean()
                })
            
            # Volume momentum
            if len(window_trades) > 1:
                volume_changes = window_volume.pct_change().dropna()
                features.update({
                    f'volume_momentum_{window}m': volume_changes.mean(),
                    f'volume_momentum_std_{window}m': volume_changes.std(),
                    f'volume_acceleration_{window}m': volume_changes.diff().mean()
                })
        
        # Calculate volume trend features first
        if len(volume_data) > 1:
            try:
                features['volume_trend_slope'] = np.polyfit(range(len(volume_data)), volume_data, 1)[0]
                features['volume_trend_r2'] = np.corrcoef(range(len(volume_data)), volume_data)[0,1] ** 2
                features['volume_trend_strength'] = abs(features['volume_trend_slope']) / volume_data.std() if volume_data.std() > 0 else 0
            except Exception as e:
                self.logger.warning(f"Error calculating trend features: {str(e)}")
                features.update({
                    'volume_trend_slope': 0,
                    'volume_trend_r2': 0,
                    'volume_trend_strength': 0
                })
        else:
            features.update({
                'volume_trend_slope': 0,
                'volume_trend_r2': 0,
                'volume_trend_strength': 0
            })
        
        return features

    def detect_trade_clusters(self, trades_df: pd.DataFrame, max_interval: int = 60) -> List[pd.DataFrame]:
        """Detect clusters of trades based on time intervals"""
        if len(trades_df) < 2:
            return [trades_df] if not trades_df.empty else []
            
        trades_df = trades_df.sort_values('timestamp')
        clusters = []
        current_cluster = [trades_df.iloc[0]]
        
        for i in range(1, len(trades_df)):
            current_trade = trades_df.iloc[i]
            last_trade = current_cluster[-1]
            
            interval = (current_trade['timestamp'] - last_trade['timestamp']).total_seconds()
            
            if interval <= max_interval:
                current_cluster.append(current_trade)
            else:
                if len(current_cluster) > 1:
                    clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [current_trade]
        
        if len(current_cluster) > 1:
            clusters.append(pd.DataFrame(current_cluster))
            
        return clusters

    async def process_spike(self, spike: Dict) -> Optional[Dict[str, Any]]:
        """Process a single spike and extract features from pre-spike window"""
        try:
            token_address = spike['token_address']
            spike_time = datetime.fromisoformat(spike['spike_time'])
            
            # Get the 15-minute window before the spike
            end_time = spike_time
            start_time = end_time - timedelta(minutes=15)
            
            self.logger.debug(f"Processing {token_address} window: {start_time} to {end_time}")
            
            # Fetch all necessary data
            price_df = await self.get_price_history(token_address, start_time, end_time)
            trades_df = await self.get_trade_history(token_address, start_time, end_time)
            token_info = await self.get_token_info(token_address)
            
            if price_df.empty:
                self.logger.warning(f"No price data for {token_address}")
                return None
            
            # Log data points for verification
            self.logger.info(f"Found {len(price_df)} price points and {len(trades_df)} trades in pre-spike window")
            
            features = {
                'token_address': token_address,
                'spike_time': spike_time.isoformat(),
                'future_price_increase': spike['increase']
            }
            
            # Add price features
            price_features = self.calculate_price_features(price_df)
            features.update(price_features)
            
            # Add volume features
            if not trades_df.empty:
                if 'from' in trades_df.columns:
                    # Extract nested data
                    trades_df['amount'] = trades_df.apply(
                        lambda x: float(x['from'].get('amount', 0)) if isinstance(x['from'], dict) else 0, 
                        axis=1
                    )
                    trades_df['side'] = trades_df.apply(
                        lambda x: 'buy' if isinstance(x['to'], dict) and x['to'].get('address') == token_address else 'sell',
                        axis=1
                    )
                
                volume_features = self.calculate_volume_features(trades_df)
                features.update(volume_features)
            
            # Add technical indicators
            tech_features = self.calculate_technical_indicators(price_df)
            features.update(tech_features)
            
            # Add token info features
            if token_info:
                token_features = {
                    'liquidity': token_info.get('liquidity', 0),
                    'market_count': token_info.get('numberMarkets', 0),
                    'holder_count': token_info.get('holder', 0),
                    'unique_wallets_24h': token_info.get('uniqueWallet24h', 0),
                    'active_ratio': token_info.get('uniqueWallet24h', 0) / token_info.get('holder', 1) if token_info.get('holder', 0) > 0 else 0,
                    'trades_24h': token_info.get('trade24h', 0),
                    'trades_per_holder': token_info.get('trade24h', 0) / token_info.get('holder', 1) if token_info.get('holder', 0) > 0 else 0
                }
                features.update(token_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing spike for {token_address}: {str(e)}")
            traceback.print_exc()
            return None

    async def process_all_spikes(self, spikes_data) -> pd.DataFrame:
        """Process all spikes with improved error handling and progress tracking"""
        try:
            # Handle both DataFrame and file path inputs
            if isinstance(spikes_data, str):
                spikes_df = pd.read_csv(spikes_data)
            else:
                spikes_df = spikes_data
                
            self.logger.info(f"Processing {len(spikes_df)} spikes")
            start_time = datetime.now()
            
            features_list = []
            failed_spikes = []
            processed_count = 0
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(5)
            
            async def process_with_rate_limit(spike):
                async with semaphore:
                    features = await self.process_spike(spike)
                    await asyncio.sleep(self.DELAY_BETWEEN_REQUESTS)
                    return features, spike['token_address']
            
            # Process spikes concurrently with rate limiting
            tasks = []
            for _, spike in spikes_df.iterrows():
                tasks.append(process_with_rate_limit(spike.to_dict()))
            
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, tuple) and result[0] is not None:
                        features_list.append(result[0])
                        processed_count += 1
                    else:
                        failed_spikes.append(result[1] if isinstance(result, tuple) else "Unknown")
                
                self.logger.info(f"Processed {processed_count}/{len(spikes_df)} spikes")
            
            if not features_list:
                self.logger.error("No features were extracted")
                return pd.DataFrame()
            
            # Create features DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Save processing stats
            self.stats = ProcessingStats(
                total_spikes=len(spikes_df),
                processed_spikes=processed_count,
                failed_spikes=len(failed_spikes),
                start_time=start_time,
                end_time=datetime.now(),
                unique_tokens=features_df['token_address'].nunique(),
                feature_count=len(features_df.columns)
            )
            
            # Verify data quality
            if not self.verify_data_quality(features_df):
                self.logger.warning("Data quality issues detected")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error processing spikes: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def select_features(self, features_df: pd.DataFrame, target_col: str = 'future_price_increase') -> pd.DataFrame:
        """Select most relevant features using correlations and domain knowledge"""
        # Prepare data
        metadata_cols = ['token_address', 'spike_time', target_col]
        X = features_df.drop(metadata_cols, axis=1, errors='ignore')
        y = features_df[target_col] if target_col in features_df.columns else None
        
        self.logger.info(f"Starting feature selection with {len(X.columns)} features")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert boolean columns to int
        bool_columns = X.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            X[col] = X[col].astype(int)
        
        # Handle categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if 'regime' in col:
                regime_map = {'low': 0, 'normal': 1, 'high': 2}
                X[col] = X[col].map(regime_map).fillna(1)
            elif 'pattern_type' in col:
                pattern_map = {'neutral': 0, 'uptrend': 1}
                X[col] = X[col].map(pattern_map).fillna(0)
            else:
                X = X.drop(columns=[col])
        
        # Handle numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Handle infinity values
            X[col] = X[col].replace([np.inf, -np.inf], [1e10, -1e10])
            
            # Log transform large values (like volume)
            if X[col].abs().max() > 1e6:
                X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))
            
            # Clip extreme values to 99th percentile
            percentile_99 = np.percentile(X[col].abs(), 99)
            X[col] = X[col].clip(-percentile_99, percentile_99)
        
        # Calculate absolute correlations with target
        correlations = X.corrwith(y).abs()
        raw_correlations = X.corrwith(y)  # Also keep raw correlations
        correlations = correlations.sort_values(ascending=False)
        
        # Select features with significant correlations
        strong_features = correlations[correlations > 0.3].index.tolist()
        
        # Always include key technical indicators
        important_features = [
            'price_change_15m', 'price_change_10m', 'price_change_5m',
            'momentum_15m', 'momentum_10m', 'momentum_5m',
            'rsi_14', 'rsi_14_slope',
            'volume_15m', 'volume_trend_strength',
            'volatility_15m', 'trend_strength_15m',
            'buy_sell_imbalance_15m', 'large_trade_ratio_15m'
        ]
        
        selected_features = list(set(strong_features + important_features))
        
        # Save feature importance scores
        scores_df = pd.DataFrame({
            'feature': correlations.index,
            'raw_correlation': raw_correlations,
            'abs_correlation': correlations,
            'selected': correlations.index.isin(selected_features)
        }).sort_values('abs_correlation', ascending=False)
        
        self.logger.info(f"Selected {len(selected_features)} features")
        
        # Print top features
        print("\nTop 30 most important features:")
        print(scores_df.head(30).to_string())
        
        # Add back metadata columns
        final_features = metadata_cols + selected_features
        return features_df[final_features]

    def analyze_features(self, features_df: pd.DataFrame, run_dir: str):
        """Analyze extracted features and save analysis results"""
        analysis_dir = os.path.join(run_dir, 'feature_analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Get only numeric columns for analysis
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_df = features_df[numeric_cols]
        
        # Calculate basic statistics
        stats = numeric_df.describe().transpose()
        
        # Calculate correlations with target (if exists)
        if 'price_change_15m' in numeric_cols:
            stats['correlation_with_target'] = numeric_df.corrwith(numeric_df['price_change_15m'])
        
        # Save statistics
        stats.to_csv(os.path.join(analysis_dir, 'feature_statistics.csv'))
        
        # Calculate and save correlations
        correlations = numeric_df.corr()
        correlations.to_csv(os.path.join(analysis_dir, 'feature_correlations.csv'))
        
        # Identify highly correlated pairs
        corr_pairs = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                if abs(correlations.iloc[i, j]) > 0.8:
                    corr_pairs.append({
                        'feature1': correlations.columns[i],
                        'feature2': correlations.columns[j],
                        'correlation': correlations.iloc[i, j]
                    })
        
        # Analyze feature patterns
        patterns = {
            'highly_correlated_pairs': len(corr_pairs),
            'stable_features': len(stats[stats['std'] < stats['std'].median()]),
            'volatile_features': len(stats[stats['std'] > stats['std'].median()]),
        }
        
        # Save patterns analysis
        with open(os.path.join(analysis_dir, 'feature_patterns.json'), 'w') as f:
            json.dump(patterns, f, indent=2)
        
        # Generate analysis report
        report = f"""# Feature Analysis Report

    ## Run Summary
    - Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Total Spikes Analyzed: {len(features_df)}
    - Total Features: {len(features_df.columns)}
    - Numeric Features: {len(numeric_cols)}

    ## Key Statistics
    - Most predictive features (by correlation with 15m price change):
    {stats.nlargest(10, 'correlation_with_target')[['correlation_with_target']].to_string() if 'correlation_with_target' in stats.columns else 'No target correlations calculated'}

    ## Feature Patterns
    - Highly correlated pairs: {patterns['highly_correlated_pairs']}
    - Stable features: {patterns['stable_features']}
    - Volatile features: {patterns['volatile_features']}
    """
        
        with open(os.path.join(analysis_dir, 'analysis_report.md'), 'w') as f:
            f.write(report)
        
        self.logger.info(f"Feature analysis saved to {analysis_dir}")

    def save_features(self, features_df: pd.DataFrame, run_dir: str):
        """Save extracted features and analysis"""
        # Create features directory
        features_dir = os.path.join(run_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        
        # Save raw features
        raw_features_file = os.path.join(features_dir, 'raw_features.csv')
        features_df.to_csv(raw_features_file, index=False)
        self.logger.info(f"Saved raw features to {raw_features_file}")
        
        # Select and save important features
        self.logger.info("Starting feature selection")
        selected_features = self.select_features(features_df)
        selected_features_file = os.path.join(features_dir, 'selected_features.csv')
        selected_features.to_csv(selected_features_file, index=False)
        self.logger.info(f"Saved selected features to {selected_features_file}")
        
        # Analyze features
        self.analyze_features(features_df, run_dir)

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spikes_file = os.path.join(base_dir, 'data', 'historical_tokens', 'spikes_20241030_123758.csv')
    
    if not os.path.exists(spikes_file):
        print(f"Spikes file not found: {spikes_file}")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, 'data', 'runs', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Loading spikes from: {spikes_file}")
    print(f"Saving results to: {run_dir}")
    
    # Create feature extractor
    extractor = FeatureExtractor(base_dir)
    
    # Load spikes data
    spikes_df = pd.read_csv(spikes_file)
    
    # Take only first 10 spikes for testing
    test_spikes_df = spikes_df.head(10)
    print(f"Testing with first {len(test_spikes_df)} spikes...")
    
    features_df = await extractor.process_all_spikes(test_spikes_df)
    
    if not features_df.empty:
        print(f"Successfully extracted features for {len(features_df)} spikes")
        extractor.save_features(features_df, run_dir)
        
        print("\nFeature columns:")
        for col in features_df.columns:
            print(f"- {col}")
            
        proceed = input("\nDo you want to process all spikes? (y/n): ")
        if proceed.lower() == 'y':
            print(f"\nProcessing all {len(spikes_df)} spikes...")
            print("Progress will be logged...")
            
            full_features_df = await extractor.process_all_spikes(spikes_df)
            if not full_features_df.empty:
                print(f"\nSuccessfully extracted features for all {len(full_features_df)} spikes")
                
                # Save full results
                full_run_dir = os.path.join(base_dir, 'data', 'runs', f'run_{timestamp}_full')
                os.makedirs(full_run_dir, exist_ok=True)
                extractor.save_features(full_features_df, full_run_dir)
                
                # Print processing stats
                if extractor.stats:
                    duration = extractor.stats.end_time - extractor.stats.start_time
                    print(f"\nProcessing Statistics:")
                    print(f"Total time: {duration}")
                    print(f"Processed: {extractor.stats.processed_spikes}/{extractor.stats.total_spikes}")
                    print(f"Failed: {extractor.stats.failed_spikes}")
                    print(f"Unique tokens: {extractor.stats.unique_tokens}")
                    print(f"Feature count: {extractor.stats.feature_count}")
    else:
        print("No features were extracted in test run")

if __name__ == "__main__":
    import traceback
    asyncio.run(main())