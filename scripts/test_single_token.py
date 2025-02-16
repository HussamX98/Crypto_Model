import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import asyncio
import logging
import json
import requests
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

# Add root directory to Python path for config import
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from config import API_KEY

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and pandas types"""
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

class TestSessionSummary:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.tokens_found = []
        self.attempted_trades = []
        self.features_detected = []
        self.session_start = datetime.now()
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

    def add_token(self, token_data: Dict):
        self.tokens_found.append({
            'timestamp': datetime.now(),
            'symbol': token_data['symbol'],
            'address': token_data['address'],
            'liquidity': token_data.get('liquidity', 0),
            'market_cap': token_data.get('market_cap', 0),
            'monitoring_duration': 0,
            'price_data_received': False
        })

    def update_token_status(self, address: str, **kwargs):
        for token in self.tokens_found:
            if token['address'] == address:
                token.update(kwargs)

    def add_feature_detection(self, token_address: str, features: Dict):
        self.features_detected.append({
            'timestamp': datetime.now(),
            'token_address': token_address,
            'features': features
        })

    def add_trade_attempt(self, token_address: str, confidence: float, features: Dict):
        self.attempted_trades.append({
            'timestamp': datetime.now(),
            'token_address': token_address,
            'confidence': confidence,
            'features': features
        })

    def generate_summary(self):
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        summary = {
            'session_start': self.session_start.isoformat(),
            'session_duration_minutes': session_duration,
            'tokens_analyzed': len(self.tokens_found),
            'features_detected': len(self.features_detected),
            'trade_attempts': len(self.attempted_trades),
            'tokens_with_price_data': sum(1 for t in self.tokens_found if t['price_data_received']),
            'avg_monitoring_duration': sum(t['monitoring_duration'] for t in self.tokens_found) / len(self.tokens_found) if self.tokens_found else 0
        }

        # Save summary files
        with open(os.path.join(self.results_dir, 'session_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4, cls=CustomJSONEncoder)

        # Save detailed token list
        pd.DataFrame(self.tokens_found).to_csv(
            os.path.join(self.results_dir, 'tokens_analyzed.csv'), 
            index=False
        )

        # Save feature detections
        if self.features_detected:
            pd.DataFrame(self.features_detected).to_csv(
                os.path.join(self.results_dir, 'features_detected.csv'),
                index=False
            )

        # Save trade attempts
        if self.attempted_trades:
            pd.DataFrame(self.attempted_trades).to_csv(
                os.path.join(self.results_dir, 'trade_attempts.csv'),
                index=False
            )

        # Generate report
        report_path = os.path.join(self.results_dir, 'test_run_report.txt')
        with open(report_path, 'w') as f:
            f.write("=== TEST RUN REPORT ===\n\n")
            f.write(f"Session Start: {self.session_start}\n")
            f.write(f"Duration: {session_duration:.1f} minutes\n\n")
            
            f.write("SUMMARY METRICS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Tokens Analyzed: {len(self.tokens_found)}\n")
            f.write(f"Tokens with Price Data: {sum(1 for t in self.tokens_found if t['price_data_received'])}\n")
            f.write(f"Feature Detections: {len(self.features_detected)}\n")
            f.write(f"Trade Attempts: {len(self.attempted_trades)}\n\n")
            
            f.write("TOKEN ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            for token in self.tokens_found:
                f.write(f"\nToken: {token['symbol']}\n")
                f.write(f"Liquidity: ${token['liquidity']:,.2f}\n")
                f.write(f"Market Cap: ${token['market_cap']:,.2f}\n")
                f.write(f"Got Price Data: {'Yes' if token['price_data_received'] else 'No'}\n")
                f.write(f"Monitoring Duration: {token['monitoring_duration']:.1f} minutes\n")

@dataclass
class TokenState:
    address: str
    symbol: str
    last_price: float
    feature_values: Dict[str, float]
    last_update: datetime
    monitoring_start: datetime
    trades: List[Dict]

class SingleTokenTester:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = self._setup_logging()
        
        # Create results directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(base_dir, 'test_results', f'test_run_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize summary
        self.summary = TestSessionSummary(self.results_dir)
        
        # Load model and configuration
        self.model_dir = os.path.join(base_dir, 'model_results', '20241118_222036')
        self.model = self._load_model()
        self.training_features = self._load_training_features()
        self.feature_stats = self._calculate_feature_stats()
        
        # Feature thresholds from successful backtesting
        self.feature_thresholds = {
            'momentum_7m': {'min': 0.05, 'max': 1.2},
            'momentum_5m': {'min': 0.05, 'max': 1.3},
            'trend_acceleration': {'min': 0.02, 'max': 0.6},
            'price_change_15m': {'min': 0.1, 'max': 1.3}
        }
        
        self.confidence_threshold = 0.9
        
        # Track active tokens
        self.active_tokens: Dict[str, TokenState] = {}
        self.blacklisted_tokens: Set[str] = set()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('SingleTokenTester')
        logger.setLevel(logging.INFO)
        
        # Console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = os.path.join(self.base_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

    def _load_model(self) -> xgb.XGBClassifier:
        """Load the trained model"""
        model_path = os.path.join(self.model_dir, 'spike_model.json')
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    def _load_training_features(self) -> pd.DataFrame:
        """Load training features"""
        features_path = os.path.join(self.model_dir, 'training_features.csv')
        return pd.read_csv(features_path)

    def _calculate_feature_stats(self) -> Dict:
        """Calculate feature statistics from training data"""
        stats = {}
        for col in self.training_features.columns:
            stats[col] = {
                'mean': float(self.training_features[col].mean()),
                'std': float(self.training_features[col].std()),
                'min': float(self.training_features[col].min()),
                'max': float(self.training_features[col].max())
            }
        return stats

    async def get_new_token(self) -> Optional[Dict]:
        """Get new token with good liquidity"""
        url = "https://public-api.birdeye.so/defi/v2/tokens/new_listing?limit=10"
        headers = {
            "accept": "application/json",
            "x-api-key": API_KEY
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return None
                
            data = response.json()
            if not data.get('success'):
                return None
                
            tokens = data['data']['items']
            self.logger.info(f"Found {len(tokens)} new tokens")
            
            for token in tokens:
                if token.get('liquidity', 0) > 1000:
                    token_data = await self.get_token_data(token['address'])
                    if token_data and token_data.get('data', {}).get('marketcap', float('inf')) < 10_000_000:
                        self.logger.info(f"\nFound suitable token:")
                        self.logger.info(f"Symbol: {token['symbol']}")
                        self.logger.info(f"Address: {token['address']}")
                        self.logger.info(f"Liquidity: ${token.get('liquidity', 0):,.2f}")
                        
                        # Add to summary
                        self.summary.add_token({
                            'symbol': token['symbol'],
                            'address': token['address'],
                            'liquidity': token.get('liquidity', 0),
                            'market_cap': token_data['data']['marketcap']
                        })
                        
                        return token
                        
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching new tokens: {str(e)}")
            return None

    async def get_token_data(self, address: str) -> Optional[Dict]:
        """Get token market data"""
        url = f"https://public-api.birdeye.so/defi/v3/token/market-data?address={address}"
        headers = {
            "accept": "application/json",
            "x-api-key": API_KEY
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return None
                
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching token data: {str(e)}")
            return None

    async def get_price_data(self, token_address: str) -> pd.DataFrame:
        """Get price data using working implementation from project knowledge"""
        url = f"https://public-api.birdeye.so/defi/ohlcv?address={token_address}&type=1m"
        headers = {
            "accept": "application/json",
            "x-api-key": API_KEY
        }
        
        try:
            response = requests.get(url, headers=headers)
            self.logger.debug(f"Price API response status: {response.status_code}")
            
            if response.status_code != 200:
                return pd.DataFrame()
                
            data = response.json()
            self.logger.debug(f"Price data response: {data}")
            
            if not data.get('success'):
                return pd.DataFrame()
            
            items = data.get('data', {}).get('items', [])
            if not items:
                return pd.DataFrame()
            
            df = pd.DataFrame(items)
            df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
            df = df.set_index('timestamp')
            df['value'] = df['c']  # Use closing price
            
            if not df.empty:
                self.logger.info(f"Got {len(df)} minutes of price data")
                self.logger.info(f"Price range: ${df['value'].min():.6f} to ${df['value'].max():.6f}")
                
                # Update summary
                self.summary.update_token_status(
                    token_address,
                    price_data_received=True,
                    last_price=float(df['value'].iloc[-1])
                )
            
            return df[['value']].sort_index()
            
        except Exception as e:
            self.logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()

    def calculate_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate features exactly matching backtesting"""
        features = {}
        price_series = price_data['value']
        
        try:
            # Calculate core features
            if len(price_series) >= 7:
                features['momentum_7m'] = float(price_series.pct_change(7).iloc[-1])
            
            if len(price_series) >= 5:
                features['momentum_5m'] = float(price_series.pct_change(5).iloc[-1])
                price_changes = price_series.pct_change()
                features['trend_acceleration'] = float(price_changes.diff().iloc[-1])
            
            if len(price_series) >= 15:
                features['price_change_15m'] = float(price_series.pct_change(15).iloc[-1])
            
            # Log raw features
            for feature, value in features.items():
                self.logger.info(f"{feature}: {value:.4f}")
            
            # Standardize features using training stats
            standardized = {}
            for feature, value in features.items():
                if feature in self.feature_stats:
                    mean = self.feature_stats[feature]['mean']
                    std = self.feature_stats[feature]['std']
                    if std > 0:
                        standardized[feature] = (value - mean) / std
                    else:
                        standardized[feature] = 0
            
            if standardized:
                # Add to summary
                self.summary.add_feature_detection(price_data.index[-1], standardized)
            
            return standardized
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            return {}

    def validate_features(self, features: Dict[str, float]) -> bool:
        """Validate features against thresholds"""
        for feature, value in features.items():
            if feature in self.feature_thresholds:
                thresholds = self.feature_thresholds[feature]
                if not (thresholds['min'] <= value <= thresholds['max']):
                    return False
        return True

    def get_trade_confidence(self, features: Dict[str, float]) -> float:
        """Get model confidence score"""
        try:
            feature_array = pd.DataFrame([features])
            confidence = float(self.model.predict_proba(feature_array)[0][1])
            self.logger.info(f"Confidence Score: {confidence:.4f}")
            return confidence
        except Exception as e:
            self.logger.error(f"Error getting confidence score: {str(e)}")
            return 0.0

    async def monitor_token(self, duration_minutes: int = 30):
        """Monitor new tokens with summary tracking"""
        self.logger.info(f"\nStarting {duration_minutes} minute monitoring session")
        start_time = datetime.now()
        
        while (datetime.now() - start_time) < timedelta(minutes=duration_minutes):
            try:
                token = await self.get_new_token()
                if not token:
                    self.logger.info("No suitable token found, retrying...")
                    await asyncio.sleep(60)
                    continue
                
                token_start_time = datetime.now()
                token_address = token['address']
                token_symbol = token['symbol']
                
                self.logger.info(f"\nMonitoring {token_symbol} ({token_address})")
                
                # Monitor token for up to 5 minutes
                for _ in range(5):
                    price_data = await self.get_price_data(token_address)
                    if price_data.empty:
                        await asyncio.sleep(60)
                        continue
                    
                    # Calculate and validate features
                    features = self.calculate_features(price_data)
                    if features:
                        if self.validate_features(features):
                            confidence = self.get_trade_confidence(features)
                            
                            if confidence >= self.confidence_threshold:
                                self.logger.info("\n!!! TRADE SIGNAL DETECTED !!!")
                                self.logger.info(f"Token: {token_symbol}")
                                self.logger.info(f"Current Price: ${price_data['value'].iloc[-1]:.6f}")
                                
                                # Add trade attempt to summary
                                self.summary.add_trade_attempt(token_address, confidence, features)
                    
                    await asyncio.sleep(60)
                
                # Update monitoring duration
                monitoring_duration = (datetime.now() - token_start_time).total_seconds() / 60
                self.summary.update_token_status(
                    token_address,
                    monitoring_duration=monitoring_duration
                )
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)
        
        # Generate final summary
        self.summary.generate_summary()

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tester = SingleTokenTester(base_dir)
    await tester.monitor_token(duration_minutes=30)

if __name__ == "__main__":
    asyncio.run(main())