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

def setup_directories(base_dir: str):
    """Create directory structure for minute-level analysis"""
    # Main directories
    directories = {
        'minute_data': os.path.join(base_dir, 'data', 'minute_level_data'),
        'price_data': os.path.join(base_dir, 'data', 'minute_level_data', 'prices'),
        'features': os.path.join(base_dir, 'data', 'minute_level_data', 'features'),
        'backtest': os.path.join(base_dir, 'data', 'minute_level_data', 'backtest_results'),
        'analysis': os.path.join(base_dir, 'data', 'minute_level_data', 'analysis'),
        'logs': os.path.join(base_dir, 'data', 'minute_level_data', 'logs')
    }
    
    # Create directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped subdirectories
    run_directories = {
        'current_run': os.path.join(base_dir, 'data', 'minute_level_data', f'run_{timestamp}'),
        'features_run': os.path.join(base_dir, 'data', 'minute_level_data', 'features', f'run_{timestamp}'),
        'backtest_run': os.path.join(base_dir, 'data', 'minute_level_data', 'backtest_results', f'run_{timestamp}')
    }
    
    for dir_path in run_directories.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return directories, run_directories, timestamp

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

        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    async def fetch_with_retry(self, url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        """Fetch data with retries and error handling"""
        headers = {
            'x-api-key': self.api_key,
            'accept': 'application/json'
        }
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            await asyncio.sleep(1 * (attempt + 1))
                            continue
                        else:
                            self.logger.error(f"Request failed with status {response.status}")
                            
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                self.logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return None

    def _setup_logging(self):
        logger = logging.getLogger('FeatureExtractor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _load_spike_instances(self) -> pd.DataFrame:
            """Load the original spike instances"""
            spikes_file = os.path.join(self.base_dir, "data", "historical_tokens", "spikes_20241030_123758.csv")
            spikes_df = pd.read_csv(spikes_file)
            spikes_df['spike_time'] = pd.to_datetime(spikes_df['spike_time'])
            # Filter for significant spikes (5x or greater)
            spikes_df = spikes_df[spikes_df['increase'] >= 5]
            self.logger.info(f"Loaded {len(spikes_df)} spike instances with 5x+ increase")
            return spikes_df
    
    def _load_minute_data(self, token_address: str) -> Optional[pd.DataFrame]:
        """Load minute-level price data for a token"""
        try:
            price_dir = os.path.join(self.base_dir, "data", "minute_level_data", "prices")
            matching_files = [f for f in os.listdir(price_dir) 
                            if f.startswith(token_address) and f.endswith('.csv')]
            
            if not matching_files:
                self.logger.warning(f"No price data file found for {token_address}")
                return None
            
            # Use the most recent file if multiple exist
            price_file = os.path.join(price_dir, sorted(matching_files)[-1])
            
            self.logger.debug(f"Loading price data from: {price_file}")
            df = pd.read_csv(price_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading minute data for {token_address}: {e}")
            return None
        
    def filter_valid_spikes(self, spikes_df: pd.DataFrame) -> pd.DataFrame:
        """Filter spikes to only those with sufficient data coverage"""
        valid_spikes = []
        
        for _, spike in spikes_df.iterrows():
            token_address = spike['token_address']
            spike_time = pd.to_datetime(spike['spike_time'])
            
            # Check if we have minute data for this token
            minute_data = self._load_minute_data(token_address)
            if minute_data is None:
                continue
                
            # Check if we have sufficient data before spike
            window_start = spike_time - pd.Timedelta(minutes=15)
            window_data = minute_data[
                (minute_data['timestamp'] >= window_start) &
                (minute_data['timestamp'] < spike_time)
            ]
            
            if len(window_data) >= 15:  # Need full 15 minutes of data
                valid_spikes.append(spike)
        
        return pd.DataFrame(valid_spikes)
    
    
    def verify_features_quality(self, features_df: pd.DataFrame) -> bool:
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

    def verify_price_data_quality(self) -> pd.DataFrame:
        """Verify minute-by-minute price data quality"""
        price_dir = os.path.join(self.base_dir, "data", "minute_level_data", "prices")
        files = [f for f in os.listdir(price_dir) if f.endswith('.csv')]
        
        results = []
        for file in files:
            df = pd.read_csv(os.path.join(price_dir, file))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate time differences
            time_diffs = df['timestamp'].diff().dt.total_seconds()
            
            # Analyze data quality
            total_minutes = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
            expected_points = total_minutes
            actual_points = len(df)
            coverage = (actual_points / expected_points) * 100 if expected_points > 0 else 0
            
            results.append({
                'token': file.split('_')[0],
                'start_time': df['timestamp'].min(),
                'end_time': df['timestamp'].max(),
                'total_minutes': total_minutes,
                'data_points': actual_points,
                'coverage_percent': coverage,
                'avg_gap_seconds': time_diffs.mean(),
                'max_gap_minutes': time_diffs.max() / 60 if len(time_diffs) > 0 else 0,
                'min_price': df['value'].min(),
                'max_price': df['value'].max(),
                'price_volatility': df['value'].std() / df['value'].mean()
            })
        
        results_df = pd.DataFrame(results)
        print("\nData Quality Analysis:")
        print(f"Total files analyzed: {len(results_df)}")
        print(f"\nCoverage Statistics:")
        print(f"Average coverage: {results_df['coverage_percent'].mean():.1f}%")
        print(f"Tokens with >90% coverage: {len(results_df[results_df['coverage_percent'] > 90])}")
        print(f"Tokens with <50% coverage: {len(results_df[results_df['coverage_percent'] < 50])}")
        
        # Save quality report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(self.base_dir, "data", "minute_level_data", "analysis", f"data_quality_report_{timestamp}.csv")
        results_df.to_csv(analysis_file, index=False)
        self.logger.info(f"Saved quality report to: {analysis_file}")
        
        return results_df

    def check_data_availability(self):
        """Check data availability and quality"""
        price_dir = os.path.join(self.base_dir, "data", "historical_prices")
        # Only look at CSV files, ignore JSON
        available_tokens = {f.split('_prices_')[0] for f in os.listdir(price_dir) 
                        if f.endswith('.csv') and not f.endswith('_summary.csv')}
        
        print("\nData Availability Analysis:")
        print(f"Total available token data files: {len(available_tokens)}")
        
        # Load a few files to check data quality
        sample_files = [f for f in sorted(os.listdir(price_dir)) 
                    if f.endswith('.csv') and not f.endswith('_summary.csv')][:5]
        print("\nSample Data Analysis:")
        
        total_gaps = []
        for file in sample_files:
            try:
                df = pd.read_csv(os.path.join(price_dir, file))
                token = file.split('_prices_')[0]
                
                print(f"\nToken: {token}")
                print(f"Available columns: {df.columns.tolist()}")
                
                if 'unixTime' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
                
                # Calculate time gaps
                time_diffs = df['timestamp'].diff().dt.total_seconds()
                avg_gap = time_diffs.mean()
                max_gap = time_diffs.max()
                total_gaps.extend(time_diffs.dropna().tolist())
                
                print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"Total points: {len(df)}")
                print(f"Points per day: {len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400):.2f}")
                print(f"Average gap between points: {avg_gap:.2f} seconds")
                print(f"Maximum gap: {max_gap:.2f} seconds")
                print(f"\nSample data points:")
                print(df[['timestamp', 'value']].head())
                print("\n" + "="*50)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        # Overall statistics
        if total_gaps:
            gaps = pd.Series(total_gaps)
            print("\nOverall Data Quality Statistics:")
            print(f"Average gap across all samples: {gaps.mean():.2f} seconds")
            print(f"Median gap: {gaps.median():.2f} seconds")
            print(f"Gap distribution:")
            print(f"  < 1 minute: {(gaps < 60).mean()*100:.1f}%")
            print(f"  1-5 minutes: {((gaps >= 60) & (gaps < 300)).mean()*100:.1f}%")
            print(f"  5-15 minutes: {((gaps >= 300) & (gaps < 900)).mean()*100:.1f}%")
            print(f"  > 15 minutes: {(gaps >= 900).mean()*100:.1f}%")

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

    async def get_trade_history(self, token_address: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Get trading history for a token with proper timestamp handling"""
        try:
            params = {
                'address': token_address,
                'tx_type': 'swap',
                'time_from': int(start_time.timestamp()),
                'time_to': int(end_time.timestamp())
            }
            
            url = f"{self.base_url}/defi/txs/token"
            response_data = await self.fetch_with_retry(url, params)
            
            if response_data and response_data.get('success') and response_data.get('data', {}).get('items'):
                trades = response_data['data']['items']
                
                # Process trades into flat structure
                processed_trades = []
                for trade in trades:
                    processed_trade = {
                        'timestamp': pd.to_datetime(trade['blockUnixTime'], unit='s'),
                        'amount': float(trade['from'].get('amount', 0)) if isinstance(trade['from'], dict) else 0,
                        'price': float(trade['tokenPrice']) if trade.get('tokenPrice') is not None else 0,
                        'side': trade.get('side', 'unknown'),
                        'source': trade.get('source', 'unknown')
                    }
                    processed_trades.append(processed_trade)
                
                if processed_trades:
                    trades_df = pd.DataFrame(processed_trades)
                    trades_df = trades_df.sort_values('timestamp')
                    self.logger.debug(f"Processed {len(trades_df)} trades for {token_address}")
                    return trades_df
                
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching trade history for {token_address}: {str(e)}")
            return pd.DataFrame()
    
    
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
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate the Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0  # Default neutral value if not enough data
                
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Separate gains and losses
            gains = deltas.copy()
            losses = deltas.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0  # Return neutral RSI on error
            
    def _calculate_minute_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive minute-level features"""
        try:
            if df.empty or len(df) < 2:
                return {}
                
            # Ensure data is sorted by time
            df = df.sort_values('timestamp')
            prices = df['value'].values
            
            features = {}
            
            # Price changes at different intervals
            for minutes in [1, 3, 5, 10, 15]:
                if len(prices) >= minutes:
                    features[f'price_change_{minutes}m'] = float((prices[-1] / prices[-minutes] - 1) * 100)
                    features[f'volatility_{minutes}m'] = float(np.std(prices[-minutes:]) / np.mean(prices[-minutes:]))
                    features[f'high_{minutes}m'] = float(np.max(prices[-minutes:]))
                    features[f'low_{minutes}m'] = float(np.min(prices[-minutes:]))
                    features[f'range_{minutes}m'] = float(np.max(prices[-minutes:]) - np.min(prices[-minutes:]))
            
            # Technical indicators
            features.update({
                # RSI variations
                'rsi_14': float(self._calculate_rsi(prices, 14)),
                'rsi_7': float(self._calculate_rsi(prices, 7)),
                'rsi_5': float(self._calculate_rsi(prices, 5)),
                'rsi_change': float(self._calculate_rsi(prices[-5:]) - self._calculate_rsi(prices[:-5])) if len(prices) >= 10 else 0,
                
                # Moving averages
                'sma_5': float(np.mean(prices[-5:])) if len(prices) >= 5 else float(prices[-1]),
                'sma_10': float(np.mean(prices[-10:])) if len(prices) >= 10 else float(prices[-1]),
                'sma_15': float(np.mean(prices[-15:])) if len(prices) >= 15 else float(prices[-1]),
                
                # Momentum indicators
                'momentum_1m': float(prices[-1] - prices[-2]) if len(prices) >= 2 else 0,
                'momentum_5m': float(prices[-1] - prices[-6]) if len(prices) >= 6 else 0,
                'momentum_15m': float(prices[-1] - prices[-16]) if len(prices) >= 16 else 0,
                
                # Trend strength and acceleration
                'trend_strength': float(np.polyfit(range(len(prices)), prices, 1)[0] / np.mean(prices)),
                'price_acceleration': float(np.diff(np.diff(prices)).mean() if len(prices) > 2 else 0),
                
                # Volatility metrics
                'volatility_ratio': float(np.std(prices[-5:]) / np.std(prices) if len(prices) >= 5 else 1),
                'volatility_change': float((np.std(prices[-5:]) - np.std(prices[:-5])) / np.std(prices[:-5])) if len(prices) >= 10 else 0,
                
                # Price pattern metrics
                'price_range_ratio': float((np.max(prices) - np.min(prices)) / np.mean(prices)),
                'upper_shadow_ratio': float((np.max(prices[-5:]) - prices[-1]) / prices[-1]) if len(prices) >= 5 else 0,
                'lower_shadow_ratio': float((prices[-1] - np.min(prices[-5:])) / prices[-1]) if len(prices) >= 5 else 0,
                
                # Time-weighted metrics
                'twap_5m': float(np.average(prices[-5:], weights=range(1, 6))) if len(prices) >= 5 else float(prices[-1]),
                'vwap_ratio': float(prices[-1] / np.mean(prices)),
            })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating minute features: {str(e)}")
            return {}


    def _calculate_price_acceleration(self, df: pd.DataFrame) -> float:
        """Calculate price acceleration (change in rate of change)"""
        if len(df) < 3:
            return 0.0
        
        price_changes = df['value'].pct_change()
        acceleration = price_changes.diff().mean()
        return float(acceleration)

    def _calculate_volume_acceleration(self, df: pd.DataFrame) -> float:
        """Calculate volume acceleration"""
        if len(df) < 3 or 'volume' not in df:
            return 0.0
        
        volume_changes = df['volume'].pct_change()
        acceleration = volume_changes.diff().mean()
        return float(acceleration)

    def _detect_minute_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect specific price patterns in minute data"""
        patterns = {}
        
        if len(df) < 15:
            return patterns
            
        prices = df['value'].values
        
        # Detect trend
        linear_fit = np.polyfit(range(len(prices)), prices, 1)
        trend_direction = np.sign(linear_fit[0])
        trend_strength = abs(linear_fit[0]) / np.mean(prices)
        
        # Detect consolidation
        volatility = np.std(prices) / np.mean(prices)
        is_consolidating = volatility < 0.02  # 2% threshold
        
        # Detect volume pattern
        if 'volume' in df:
            volumes = df['volume'].values
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            volume_increasing = volume_trend > 0
            volume_strength = abs(volume_trend) / np.mean(volumes)
        else:
            volume_increasing = False
            volume_strength = 0
        
        patterns.update({
            'trend_direction': float(trend_direction),
            'trend_strength': float(trend_strength),
            'is_consolidating': float(is_consolidating),
            'consolidation_tightness': float(1.0 - volatility),
            'volume_increasing': float(volume_increasing),
            'volume_trend_strength': float(volume_strength)
        })
        
        return patterns

    def _calculate_minute_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators using minute data"""
        momentum = {}
        
        if len(df) < 15:
            return momentum
            
        prices = df['value'].values
        
        # Calculate momentum at different timeframes
        for minutes in [1, 5, 15]:
            if len(prices) >= minutes:
                momentum[f'momentum_{minutes}m'] = float((prices[-1] / prices[-minutes] - 1) * 100)
                
        # Calculate rate of change
        for minutes in [1, 5, 15]:
            if len(prices) >= minutes:
                roc = ((prices[-1] - prices[-minutes]) / prices[-minutes]) * 100
                momentum[f'roc_{minutes}m'] = float(roc)
        
        # Add momentum acceleration
        if len(prices) >= 15:
            momentum_values = np.diff(prices) / prices[:-1]
            momentum['momentum_acceleration'] = float(np.diff(momentum_values).mean())
        
        return momentum

    def _calculate_minute_technicals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators using minute data"""
        technicals = {}
        
        if len(df) < 15:
            return technicals
            
        prices = df['value'].values
        
        # RSI at different timeframes
        for period in [5, 9, 14]:
            if len(prices) >= period:
                technicals[f'rsi_{period}'] = float(self._calculate_rsi(prices[-period:]))
        
        # Moving averages
        for period in [5, 10, 15]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                technicals[f'ma_{period}'] = float(ma)
                technicals[f'price_to_ma_{period}'] = float(prices[-1] / ma)
        
        # Bollinger Bands (using 15-minute window)
        if len(prices) >= 15:
            ma_15 = np.mean(prices)
            std_15 = np.std(prices)
            upper_band = ma_15 + (2 * std_15)
            lower_band = ma_15 - (2 * std_15)
            
            technicals.update({
                'bb_upper': float(upper_band),
                'bb_lower': float(lower_band),
                'bb_width': float((upper_band - lower_band) / ma_15),
                'bb_position': float((prices[-1] - lower_band) / (upper_band - lower_band)) if upper_band != lower_band else 0.5
            })
        
        return technicals
    
    def calculate_advanced_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced momentum indicators"""
        features = {}
        
        if len(df) < 15:
            return features
            
        prices = df['value'].values
        
        # Triple momentum indicators (short/medium/long)
        for window in [3, 7, 14]:
            if len(prices) >= window:
                momentum = (prices[-1] - prices[-window]) / prices[-window]
                features[f'momentum_{window}m'] = float(momentum)
                
                # Momentum acceleration
                if len(prices) >= window + 1:
                    prev_momentum = (prices[-2] - prices[-window-1]) / prices[-window-1]
                    features[f'momentum_acc_{window}m'] = float(momentum - prev_momentum)
        
        # Momentum divergence (only if volume data exists)
        if len(prices) >= 15 and 'volume' in df.columns:
            price_momentum = (prices[-1] - prices[-15]) / prices[-15]
            volume_momentum = (df['volume'].iloc[-1] - df['volume'].iloc[-15]) / df['volume'].iloc[-15]
            features['momentum_divergence'] = float(price_momentum - volume_momentum)
        
        return features

    def calculate_volume_profile_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced volume profile metrics"""
        features = {}
        
        # Return empty features if no volume data
        if 'volume' not in df.columns or len(df) < 15:
            return features
            
        try:
            volumes = df['volume'].values
            prices = df['value'].values
            
            # Only proceed if we have valid volume data
            if len(volumes) > 0 and not np.isnan(volumes).all():
                # Volume concentration
                volume_levels = pd.qcut(volumes, q=4, labels=['low', 'medium', 'high', 'very_high'])
                vol_concentration = pd.DataFrame({'volume': volumes, 'price': prices, 'level': volume_levels})
                
                # Calculate VWAP and VAH/VAL (Volume-weighted Average High/Low)
                vwap = np.average(prices, weights=volumes)
                vah = np.percentile(prices[volumes > np.median(volumes)], 70)
                val = np.percentile(prices[volumes > np.median(volumes)], 30)
                
                features.update({
                    'vwap_distance': float((prices[-1] - vwap) / vwap),
                    'vah_distance': float((prices[-1] - vah) / vah),
                    'val_distance': float((prices[-1] - val) / val),
                    'volume_profile_skew': float(vol_concentration.groupby('level')['volume'].sum().skew())
                })
                
        except Exception as e:
            self.logger.debug(f"Error calculating volume profile features: {str(e)}")
            
        return features

    def calculate_pattern_recognition_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate pattern recognition features"""
        features = {}
        
        if len(df) < 15:
            return features
            
        try:
            prices = df['value'].values
            returns = np.diff(prices) / prices[:-1]
            
            # Detect potential breakout patterns
            sma_5 = np.mean(prices[-5:])
            sma_15 = np.mean(prices)
            
            features.update({
                'breakout_strength': float((prices[-1] - sma_15) / sma_15),
                'consolidation_tightness': float(1 - (np.std(prices[-5:]) / np.std(prices))),
                'momentum_consistency': float(np.sum(returns > 0) / len(returns)),
                'trend_acceleration': float(np.polyfit(range(len(prices)), prices, 2)[0])
            })
            
            # Detect volume patterns only if volume data exists
            if 'volume' in df.columns:
                volumes = df['volume'].values
                if len(volumes) > 0 and not np.isnan(volumes).all():
                    vol_sma_5 = np.mean(volumes[-5:])
                    vol_sma_15 = np.mean(volumes)
                    
                    features.update({
                        'volume_breakout': float((volumes[-1] - vol_sma_15) / vol_sma_15),
                        'volume_trend_strength': float(np.corrcoef(range(len(volumes)), volumes)[0,1])
                    })
                    
        except Exception as e:
            self.logger.debug(f"Error calculating pattern recognition features: {str(e)}")
        
        return features


    def calculate_enhanced_volume_features(self, trades_df: pd.DataFrame, window_minutes: int = 15) -> Dict[str, float]:
        """Calculate advanced volume features for pre-spike window with error handling"""
        features = {}
        
        try:
            if trades_df.empty:
                return features
                
            trades_df = trades_df.sort_values('timestamp')
            end_time = trades_df['timestamp'].max()
            start_time = end_time - pd.Timedelta(minutes=window_minutes)
            
            # Volume windows
            for minutes in [1, 3, 5, 10, 15]:
                window_data = trades_df[trades_df['timestamp'] >= (end_time - pd.Timedelta(minutes=minutes))]
                if not window_data.empty:
                    # Calculate USD volume
                    volume = window_data['amount'] * window_data['price']
                    
                    features.update({
                        f'volume_{minutes}m': float(volume.sum()),
                        f'trade_count_{minutes}m': len(window_data),
                        f'avg_trade_size_{minutes}m': float(volume.mean()) if len(window_data) > 0 else 0.0
                    })
                    
                    # Buy/Sell analysis
                    buys = window_data[window_data['side'] == 'buy']
                    sells = window_data[window_data['side'] == 'sell']
                    
                    buy_volume = float(buys['amount'].sum() * buys['price'].mean()) if not buys.empty else 0.0
                    sell_volume = float(sells['amount'].sum() * sells['price'].mean()) if not sells.empty else 0.0
                    total_volume = buy_volume + sell_volume
                    
                    features.update({
                        f'buy_volume_{minutes}m': buy_volume,
                        f'sell_volume_{minutes}m': sell_volume,
                        f'buy_sell_ratio_{minutes}m': float(buy_volume / sell_volume) if sell_volume > 0 else 10.0,
                        f'buy_pressure_{minutes}m': float(buy_volume / total_volume) if total_volume > 0 else 0.5
                    })
                    
                    # Trade size distribution
                    if len(volume) > 0:
                        volume_95th = volume.quantile(0.95)
                        large_trades = window_data[volume >= volume_95th]
                        features.update({
                            f'large_trades_{minutes}m': len(large_trades),
                            f'large_volume_ratio_{minutes}m': float(large_trades['amount'].sum() / window_data['amount'].sum()) if not window_data.empty else 0.0
                        })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating volume features: {str(e)}")
            return features
        

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

    async def process_spike_instance(self, spike: pd.Series) -> Optional[Dict[str, Any]]:
            """Process a single spike instance with enhanced features"""
            try:
                token_address = spike['token_address']
                spike_time = pd.to_datetime(spike['spike_time'])
                
                # Get minute data
                minute_data = self._load_minute_data(token_address)
                if minute_data is None:
                    return None
                    
                # Get 15-minute window before spike
                window_end = spike_time
                window_start = window_end - pd.Timedelta(minutes=15)
                price_window = minute_data[
                    (minute_data['timestamp'] >= window_start) &
                    (minute_data['timestamp'] < window_end)
                ].copy()
                
                if len(price_window) < 15:
                    return None
                
                # Calculate all feature sets
                features = {
                    'token_address': token_address,
                    'spike_time': spike_time.isoformat(),
                    'future_price_increase': spike['increase'],
                    'spike_duration': spike['duration_minutes']
                }
                
                # Add base features
                features.update(self._calculate_minute_features(price_window))
                
                # Add advanced features
                features.update(self.calculate_advanced_momentum_features(price_window))
                features.update(self.calculate_volume_profile_features(price_window))
                features.update(self.calculate_pattern_recognition_features(price_window))
                
                # Add volume features if available
                trades_window = await self.get_trade_history(token_address, window_start, window_end)
                if trades_window is not None and not trades_window.empty:
                    features.update(self.calculate_enhanced_volume_features(trades_window))
                
                return features
                
            except Exception as e:
                self.logger.error(f"Error processing spike for {token_address}: {str(e)}")
                return None
    
    async def process_all_spikes(self, spikes_df: pd.DataFrame = None) -> pd.DataFrame:
        """Process provided spike instances with minute-level data"""
        if spikes_df is None:
            spikes_df = self._load_spike_instances()
        
        self.logger.info(f"Processing {len(spikes_df)} spike instances")
        
        features_list = []
        errors = []
        processed = 0
        start_time = datetime.now()
        
        for idx, spike in spikes_df.iterrows():
            try:
                if idx % 100 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = idx / elapsed if elapsed > 0 else 0
                    estimated_remaining = (len(spikes_df) - idx) / rate if rate > 0 else 0
                    self.logger.info(f"Progress: {idx}/{len(spikes_df)} spikes ({rate:.2f} spikes/sec)")
                    self.logger.info(f"Estimated time remaining: {timedelta(seconds=int(estimated_remaining))}")
                
                features = await self.process_spike_instance(spike)
                if features:
                    features_list.append(features)
                    processed += 1
                    
            except Exception as e:
                error_msg = f"Error processing spike {idx} for {spike['token_address']}: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                continue
        
        # Create features DataFrame
        if features_list:
            features_df = pd.DataFrame(features_list)
            
            # Save error log
            if errors:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_file = os.path.join(self.base_dir, "data", "minute_level_data", "logs", f"errors_{timestamp}.txt")
                with open(error_file, 'w') as f:
                    f.write('\n'.join(errors))
            
            self.logger.info(f"\nFeature extraction complete:")
            self.logger.info(f"Processed spikes: {processed}/{len(spikes_df)}")
            self.logger.info(f"Features extracted: {len(features_df.columns)}")
            self.logger.info(f"Errors encountered: {len(errors)}")
            
            return features_df
        
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
        
        # Calculate correlations with future price increase
        if 'future_price_increase' in numeric_cols:
            correlations = numeric_df.corr()['future_price_increase'].sort_values(ascending=False)
            print("\nTop 10 Features Correlated with Price Spikes:")
            print(correlations.head(10))
            correlations.to_csv(os.path.join(analysis_dir, 'feature_correlations.csv'))
        
        # Group spikes by magnitude
        features_df['spike_magnitude'] = pd.qcut(features_df['future_price_increase'], 
                                            q=5, 
                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Calculate average feature values for each magnitude group using only numeric columns
        magnitude_profiles = features_df.groupby('spike_magnitude')[numeric_cols].mean()
        magnitude_profiles.to_csv(os.path.join(analysis_dir, 'magnitude_profiles.csv'))
        
        # Generate analysis report
        report = f"""# Feature Analysis Report

    ## Overview
    - Total Spikes Analyzed: {len(features_df)}
    - Number of Features: {len(numeric_cols)}
    - Spike Range: {features_df['future_price_increase'].min():.2f}x to {features_df['future_price_increase'].max():.2f}x

    ## Key Indicators
    Most Predictive Features (correlation with price increase):
    {correlations.head(10).to_string()}

    ## Feature Patterns by Spike Magnitude
    {magnitude_profiles[['rsi_14', 'volatility_15m', 'momentum_15m', 'trend_strength']].to_string()}

    ## Statistical Summary
    - Median Spike Size: {features_df['future_price_increase'].median():.2f}x
    - Average RSI before spike: {features_df['rsi_14'].mean():.2f}
    - Average Volatility: {features_df['volatility_15m'].mean():.4f}
    """
        
        # Save report
        with open(os.path.join(analysis_dir, 'analysis_report.md'), 'w') as f:
            f.write(report)
        
        return correlations, magnitude_profiles
    
    def optimize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Optimize features for model training"""
        # Select most predictive features
        key_features = [
            'price_change_15m', 'price_change_10m', 'momentum_5m',
            'range_10m', 'volatility_15m', 'trend_strength', 
            'rsi_14', 'vwap_ratio', 'volume_15m', 'buy_pressure_15m',
            'large_volume_ratio_15m', 'volume_acceleration_15m',
            'future_price_increase'
        ]
        
        # Ensure all key features exist
        existing_features = [f for f in key_features if f in features_df.columns]
        df = features_df[existing_features].copy()
        
        # Add derived features
        df['volatility_trend'] = df['volatility_15m'] / df['volatility_10m']
        df['momentum_acceleration'] = df['momentum_5m'].diff()
        
        # Create categorical features
        df['rsi_zone'] = pd.cut(df['rsi_14'], 
                            bins=[0, 30, 45, 55, 70, 100],
                            labels=['oversold', 'weak', 'neutral', 'strong', 'overbought'])
        
        # Normalize numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Remove outliers
        for col in numeric_cols:
            if col != 'future_price_increase':  # Keep target variable unchanged
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q3)
        
        # Group spike sizes
        df['spike_category'] = pd.cut(df['future_price_increase'],
                                    bins=[0, 10, 20, 50, 100, float('inf')],
                                    labels=['5-10x', '10-20x', '20-50x', '50-100x', '100x+'])
        
        return df

    def balance_dataset(self, df: pd.DataFrame, target_col: str = 'spike_category') -> pd.DataFrame:
        """Balance dataset across spike categories"""
        min_samples = df[target_col].value_counts().min()
        balanced_dfs = []
        
        for category in df[target_col].unique():
            category_df = df[df[target_col] == category]
            if len(category_df) > min_samples:
                balanced_dfs.append(category_df.sample(n=min_samples, random_state=42))
            else:
                balanced_dfs.append(category_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)

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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode with 5 samples')
    parser.add_argument('--check', action='store_true', help='Check data quality only')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directories, run_directories, timestamp = setup_directories(base_dir)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(base_dir)
    proceed = 'n'
    
    if args.check:
        print("\nVerifying data quality...")
        quality_results = extractor.verify_price_data_quality()
        print("\nDetailed results saved to analysis directory")
        return
    
    # Load spike instances
    spikes_df = extractor._load_spike_instances()
    print(f"\nLoaded {len(spikes_df)} spike instances")
    
    # Filter valid spikes first
    valid_spikes_df = extractor.filter_valid_spikes(spikes_df)
    print(f"Found {len(valid_spikes_df)} spikes with sufficient data")
    print(f"Spike time range: {valid_spikes_df['spike_time'].min()} to {valid_spikes_df['spike_time'].max()}")
    
    try:
        if args.test:
            print("\nRunning in test mode with 5 samples...")
            test_spikes = valid_spikes_df.head(5)
            print("\nTest samples:")
            for _, row in test_spikes.iterrows():
                print(f"Token: {row['token_address']}, Spike time: {row['spike_time']}")
            
            features_df = await extractor.process_all_spikes(test_spikes)
            
            if not features_df.empty:
                print("\nTest run feature extraction results:")
                print(f"Total test spikes processed: {len(features_df)}")
                print("\nFeature columns:")
                for col in features_df.columns:
                    print(f"- {col}")
                
                print("\nSample of extracted features:")
                print(features_df.head())
                
                proceed = input("\nTest run complete. Would you like to process the full dataset? (y/n): ")
        
        # Process full dataset
        if not args.test or proceed.lower() == 'y':
            print("\nProcessing full dataset...")
            features_df = await extractor.process_all_spikes(valid_spikes_df)
            
            if not features_df.empty:
                print("\nFeature extraction results:")
                print(f"Total spikes processed: {len(features_df)}")
                
                # Save features
                features_file = os.path.join(directories['features'], f'spike_features_{timestamp}.csv')
                features_df.to_csv(features_file, index=False)
                print(f"\nSaved features to: {features_file}")
                
                # Analyze features
                analysis_dir = os.path.join(directories['features'], 'feature_analysis')
                os.makedirs(analysis_dir, exist_ok=True)
                patterns, magnitude_profiles = extractor.analyze_features(features_df, analysis_dir)
                print("\nFeature analysis complete. Results saved to feature_analysis directory.")
            else:
                print("No features were extracted")
                
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())