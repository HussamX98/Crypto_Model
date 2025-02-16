import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from config import API_KEY
except ImportError:
    raise ImportError(f"Failed to import API_KEY from config.py in {project_root}")

@dataclass
class TradePosition:
    """Enhanced trade position tracking"""
    token_address: str
    token_symbol: str
    entry_time: datetime
    entry_price: float
    initial_volume: float
    position_size: float
    features_at_entry: Dict[str, float]
    trade_id: str
    confidence_score: float
    highest_price: float = 0
    peak_return: float = 0
    time_to_peak: float = 0
    reduced_position: bool = False  # Track if position was reduced at 100% profit
    loss_start_time: Optional[datetime] = None  # Track when losses started
    current_position_size: float = 0  # Track current position size after reductions

@dataclass
class TokenState:
    """Track token monitoring state"""
    address: str
    symbol: str
    price_history: List[Tuple[datetime, float]]
    volume_history: List[Tuple[datetime, float]]
    trade_history: List[Dict]
    first_seen: datetime
    last_update: datetime
    features_available: Dict[str, bool]
    features_values: Dict[str, float]
    minutes_monitored: int
    active_trade: Optional[TradePosition]

class MarketMonitor:
    def __init__(self):
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.headers = {
            "accept": "application/json",
            "x-chain": "solana"
        }
        
        # Create output directories
        self.output_dir = os.path.join(project_root, 'output', self.session_timestamp)
        self.trades_dir = os.path.join(self.output_dir, 'trades')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.trades_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize state
        self.active_tokens: Dict[str, TokenState] = {}
        self.monitored_tokens: Set[str] = set()
        self.trade_positions: List[TradePosition] = []
        self.total_tokens_seen = 0
        self.profitable_trades = 0
        self.stop_losses_hit = 0
        
        # Updated monitoring parameters
        self.MAX_TOKENS = 1200
        self.POSITION_SIZE = 100
        self.MIN_CONFIDENCE = 0.60  # Updated confidence threshold
        self.TRADE_COUNT = 0
        self.TOKEN_AGE_THRESHOLD = 180  # 3 hours in minutes
        self.INACTIVITY_THRESHOLD = 15  # Minutes before removing inactive tokens
        
        # Updated feature thresholds
        self.FEATURE_THRESHOLDS = {
            'momentum_5m': {'min': -0.0474, 'max': 0.0437},
            'momentum_7m': {'min': -0.1719, 'max': -0.0935},
            'trend_acceleration': {'min': -0.2438, 'max': 0.0355},
            'price_change_15m': {'min': -0.1537, 'max': 1.0833}
        }

    def _setup_logger(self):
            """Setup enhanced logging"""
            logger = logging.getLogger('MarketMonitor')
            logger.setLevel(logging.INFO)
            
            log_dir = os.path.join(project_root, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(
                os.path.join(log_dir, f'monitor_{self.session_timestamp}.log'),
                encoding='utf-8'
            )
            
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
            return logger

    async def fetch_data(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with enhanced error handling"""
        url = f"https://public-api.birdeye.so/defi/{endpoint}"
        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": API_KEY
        }
        
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        else:
                            self.logger.error(f"API error {response.status} for {endpoint}")
                            if attempt < 2:
                                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Request failed for {endpoint}: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        return None

    async def get_new_tokens(self) -> List[Dict]:
        """Get new token listings, filtering for pump.fun tokens only"""
        try:
            response = await self.fetch_data(
                "v2/tokens/new_listing",
                params={
                    "limit": "20",
                    "meme_platform_enabled": "true"
                }
            )
            
            if not response or not response.get('success'):
                self.logger.error("Failed to get new tokens list")
                return []

            all_tokens = response.get('data', {}).get('items', [])
            pump_tokens = [
                token for token in all_tokens
                if token.get('address', '').endswith('pump')
            ]
            
            self.logger.info(f"Retrieved {len(pump_tokens)} pump.fun tokens")
            return pump_tokens

        except Exception as e:
            self.logger.error(f"Error getting new tokens: {str(e)}")
            return []

    async def get_token_data(self, token_address: str) -> Optional[Dict]:
        """Get current token price and volume data"""
        try:
            data = await self.fetch_data(
                "price_volume/single",
                params={
                    "address": token_address,
                    "type": "24h"
                }
            )
            
            if not data:
                self.logger.debug(f"No response for token {token_address}")
                return None

            if not data.get('success'):
                self.logger.debug(f"Unsuccessful response for token {token_address}")
                return None

            token_data = data.get('data', {})
            price = token_data.get('price')
            volume = token_data.get('volumeUSD', 0)
            
            if price is None:
                self.logger.debug(f"No price data for token {token_address}")
                return None
                
            return {
                'price': float(price),
                'volume': float(volume or 0),
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.debug(f"Error getting token data: {str(e)}")
            return None

    def calculate_features(self, token_state: TokenState) -> Dict[str, float]:
        """Calculate features from price and volume history"""
        features = {}
        
        df = pd.DataFrame(token_state.price_history, columns=['timestamp', 'price'])
        df = df.set_index('timestamp')
        
        try:
            if len(df) >= 5:
                window_5m = df.last('5min')
                if len(window_5m) >= 5:
                    mom_5m = float(window_5m['price'].iloc[-1] / window_5m['price'].iloc[0] - 1)
                    features['momentum_5m'] = mom_5m
                    features['volatility_5m'] = float(window_5m['price'].pct_change().std())
            
            if len(df) >= 7:
                window_7m = df.last('7min')
                if len(window_7m) >= 7:
                    mom_7m = float(window_7m['price'].iloc[-1] / window_7m['price'].iloc[0] - 1)
                    features['momentum_7m'] = mom_7m
            
            if len(df) >= 10:
                window_10m = df.last('10min')
                if len(window_10m) >= 10:
                    changes = window_10m['price'].pct_change()
                    trend_acc = float(changes.diff().mean())
                    features['trend_acceleration'] = trend_acc
            
            if len(df) >= 15:
                window_15m = df.last('15min')
                if len(window_15m) >= 15:
                    price_change = float(window_15m['price'].iloc[-1] / window_15m['price'].iloc[0] - 1)
                    features['price_change_15m'] = price_change
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
        
        return features

    def calculate_confidence_score(self, features: Dict[str, float]) -> float:
        """Calculate confidence score based on feature values"""
        if not features:
            return 0.0
        
        score = 0.0
        weights = {
            'momentum_7m': 0.12,
            'momentum_5m': 0.62,
            'trend_acceleration': 0.12,
            'price_change_15m': 0.05
        }
        
        for feature, value in features.items():
            if feature in self.FEATURE_THRESHOLDS:
                threshold = self.FEATURE_THRESHOLDS[feature]
                if threshold['min'] <= value <= threshold['max']:
                    score += weights.get(feature, 0.1)
        
        return min(score, 1.0)


    def get_token_age_minutes(self, token_state: TokenState) -> float:
        """Calculate token age in minutes"""
        return (datetime.now() - token_state.first_seen).total_seconds() / 60
    
    async def monitor_token(self, token_state: TokenState):
            """Monitor individual token with age consideration"""
            self.logger.info(f"Starting monitoring for {token_state.symbol}")
            last_feature_log = datetime.now()
            
            while True:
                try:
                    # Get current data
                    data = await self.get_token_data(token_state.address)
                    current_time = datetime.now()
                    
                    if not data:
                        await asyncio.sleep(60)
                        continue
                    
                    # Update histories
                    token_state.price_history.append((current_time, data['price']))
                    token_state.volume_history.append((current_time, data['volume']))
                    token_state.last_update = current_time
                    token_state.minutes_monitored += 1
                    
                    # Calculate features and check conditions
                    features = self.calculate_features(token_state)
                    if features:
                        confidence_score = self.calculate_confidence_score(features)
                        token_age = self.get_token_age_minutes(token_state)
                        
                        # Log feature values periodically
                        time_since_last_log = (current_time - last_feature_log).total_seconds() / 60
                        if time_since_last_log >= 5:  # Log every 5 minutes
                            self.logger.info(
                                f"\nToken: {token_state.symbol}"
                                f"\nAge: {token_age:.2f} minutes"
                                f"\nPrice: ${data['price']:.8f}"
                                f"\nConfidence Score: {confidence_score:.2f}"
                            )
                            last_feature_log = current_time
                        
                        # Only execute trade if token is mature enough and meets confidence threshold
                        if (token_age >= self.TOKEN_AGE_THRESHOLD and 
                            confidence_score >= self.MIN_CONFIDENCE and 
                            not token_state.active_trade):
                            await self.execute_trade(token_state, features, confidence_score)
                    
                    # Clean up old data
                    cutoff = current_time - timedelta(hours=1)
                    token_state.price_history = [
                        (t, p) for t, p in token_state.price_history if t > cutoff
                    ]
                    token_state.volume_history = [
                        (t, v) for t, v in token_state.volume_history if t > cutoff
                    ]
                    
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring {token_state.symbol}: {str(e)}")
                    await asyncio.sleep(60)

    async def execute_trade(self, token_state: TokenState, features: Dict[str, float], confidence: float):
        """Execute trade with enhanced entry validation"""
        try:
            data = await self.get_token_data(token_state.address)
            if not data:
                return
            
            # Calculate entry price with slippage
            entry_price = data['price'] * 1.15  # 15% slippage
            
            self.TRADE_COUNT += 1
            trade_id = f"TRADE_{self.session_timestamp}_{self.TRADE_COUNT}"
            
            # Create trade position
            position = TradePosition(
                token_address=token_state.address,
                token_symbol=token_state.symbol,
                entry_time=datetime.now(),
                entry_price=entry_price,
                initial_volume=data['volume'],
                position_size=self.POSITION_SIZE,
                features_at_entry=features.copy(),
                trade_id=trade_id,
                confidence_score=confidence,
                current_position_size=self.POSITION_SIZE
            )
            
            # Log trade entry
            entry_log = {
                'trade_id': trade_id,
                'token_symbol': token_state.symbol,
                'entry_time': position.entry_time.isoformat(),
                'market_price': data['price'],
                'entry_price': entry_price,
                'position_size': position.position_size,
                'confidence_score': confidence,
                'features': features
            }
            
            with open(os.path.join(self.trades_dir, f'{trade_id}_entry.json'), 'w') as f:
                json.dump(entry_log, f, indent=2)
            
            self.logger.info(
                f"\nTrade executed for {token_state.symbol}"
                f"\nTrade ID: {trade_id}"
                f"\nEntry Price: ${entry_price:.8f}"
                f"\nConfidence Score: {confidence:.2f}"
            )
            
            # Start monitoring
            token_state.active_trade = position
            asyncio.create_task(self.monitor_exits(token_state))
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")

    async def monitor_exits(self, token_state: TokenState):
        """Enhanced exit monitoring with sophisticated stop loss"""
        if not token_state.active_trade:
            return
        
        position = token_state.active_trade
        entry_price = position.entry_price
        highest_price = entry_price
        monitoring_start = datetime.now()
        price_action_log = []
        
        try:
            while token_state.active_trade:
                data = await self.get_token_data(token_state.address)
                if not data:
                    await asyncio.sleep(60)
                    continue
                
                current_time = datetime.now()
                current_price = data['price']
                current_return = (current_price / entry_price - 1) * 100
                time_in_trade = (current_time - position.entry_time).total_seconds() / 60
                
                # Update highest price
                if current_price > highest_price:
                    highest_price = current_price
                    position.highest_price = highest_price
                    position.peak_return = (highest_price / entry_price - 1) * 100
                    position.time_to_peak = time_in_trade
                
                # Log price action
                price_action_log.append({
                    'timestamp': current_time.isoformat(),
                    'price': current_price,
                    'return_pct': current_return,
                    'time_in_trade_minutes': time_in_trade
                })
                
                # Exit conditions
                
                # 1. Immediate 20% drawdown stop loss
                if current_return <= -20:
                    await self.exit_trade(token_state, "stop_loss_20", current_price)
                    break
                
                # 2. Check for persistent 10% loss (15 minutes)
                if current_return <= -10:
                    if position.loss_start_time is None:
                        position.loss_start_time = current_time
                    elif (current_time - position.loss_start_time).total_seconds() >= 900:  # 15 minutes
                        await self.exit_trade(token_state, "persistent_loss", current_price)
                        break
                else:
                    position.loss_start_time = None
                
                # 3. Take partial profits at 100% gain
                if current_return >= 100 and not position.reduced_position:
                    position.current_position_size *= 0.5
                    position.reduced_position = True
                    self.logger.info(
                        f"Reduced position for {token_state.symbol} by 50% at {current_return:.2f}% return"
                    )
                
                # 4. Trailing stop (10%)
                trailing_stop_price = highest_price * 0.9
                if current_price <= trailing_stop_price and current_return > 0:
                    await self.exit_trade(token_state, "trailing_stop", current_price)
                    break
                
                # Log monitoring status
                self.logger.info(
                    f"\nMonitoring {token_state.symbol}:"
                    f"\nTime in trade: {time_in_trade:.1f} minutes"
                    f"\nCurrent Return: {current_return:.2f}%"
                    f"\nPeak Return: {position.peak_return:.2f}%"
                )
                
                await asyncio.sleep(60)
                
        except Exception as e:
            self.logger.error(f"Error monitoring exits: {str(e)}")
        finally:
            # Save price action log
            log_file = os.path.join(self.trades_dir, f'{position.trade_id}_price_action.json')
            with open(log_file, 'w') as f:
                json.dump(price_action_log, f, indent=2)

    async def exit_trade(self, token_state: TokenState, exit_type: str, exit_price: float):
        """Handle trade exit with performance tracking"""
        if not token_state.active_trade:
            return
            
        position = token_state.active_trade
        current_time = datetime.now()
        
        try:
            return_pct = (exit_price / position.entry_price - 1) * 100
            time_in_trade = (current_time - position.entry_time).total_seconds() / 60
            
            exit_log = {
                'trade_id': position.trade_id,
                'exit_type': exit_type,
                'exit_time': current_time.isoformat(),
                'exit_price': exit_price,
                'return_pct': return_pct,
                'time_in_trade_minutes': time_in_trade,
                'peak_return': position.peak_return
            }
            
            # Save exit data
            with open(os.path.join(self.trades_dir, f'{position.trade_id}_exit.json'), 'w') as f:
                json.dump(exit_log, f, indent=2)
            
            # Update performance metrics
            if return_pct > 0:
                self.profitable_trades += 1
            if exit_type.startswith('stop_loss'):
                self.stop_losses_hit += 1
            
            self.logger.info(
                f"\nTrade Exit for {token_state.symbol}"
                f"\nType: {exit_type}"
                f"\nReturn: {return_pct:.2f}%"
                f"\nTime in Trade: {time_in_trade:.2f} minutes"
            )
            
            # Clear active trade
            token_state.active_trade = None
            
        except Exception as e:
            self.logger.error(f"Error exiting trade: {str(e)}")

    async def monitor_market(self):
            """Main market monitoring loop"""
            self.logger.info(f"Starting market monitor at {self.session_timestamp}")
            self.logger.info(f"Using Feature Thresholds: {json.dumps(self.FEATURE_THRESHOLDS, indent=2)}")
            self.logger.info(f"Minimum Confidence: {self.MIN_CONFIDENCE}")
            
            # Initial token collection phase
            while len(self.active_tokens) < self.MAX_TOKENS:
                try:
                    self.logger.info(f"Current token count: {len(self.active_tokens)}")
                    new_tokens = await self.get_new_tokens()
                    
                    for token in new_tokens:
                        if len(self.active_tokens) >= self.MAX_TOKENS:
                            break
                            
                        address = token['address']
                        if address in self.monitored_tokens:
                            continue

                        data = await self.get_token_data(address)
                        if not data:
                            continue
                        
                        # Initialize token state
                        token_state = TokenState(
                            address=address,
                            symbol=token['symbol'],
                            price_history=[(datetime.now(), data['price'])],
                            volume_history=[(datetime.now(), data['volume'])],
                            trade_history=[],
                            first_seen=datetime.now(),
                            last_update=datetime.now(),
                            features_available={},
                            features_values={},
                            minutes_monitored=0,
                            active_trade=None
                        )
                        
                        self.monitored_tokens.add(address)
                        self.active_tokens[address] = token_state
                        self.total_tokens_seen += 1
                        
                        # Start monitoring task
                        asyncio.create_task(self.monitor_token(token_state))
                        self.logger.info(f"Added token {token_state.symbol} ({len(self.active_tokens)}/{self.MAX_TOKENS})")
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error gathering tokens: {str(e)}")
                    await asyncio.sleep(2)

            self.logger.info(f"Initial token collection complete. Monitoring {len(self.active_tokens)} tokens")
            
            # Main monitoring loop
            while True:
                try:
                    # Remove inactive tokens
                    tokens_to_remove = []
                    for address, token_state in self.active_tokens.items():
                        minutes_inactive = (datetime.now() - token_state.last_update).total_seconds() / 60
                        if minutes_inactive >= self.INACTIVITY_THRESHOLD:
                            tokens_to_remove.append((address, token_state.symbol))
                    
                    # Remove and log
                    for address, symbol in tokens_to_remove:
                        del self.active_tokens[address]
                        self.monitored_tokens.remove(address)
                        self.logger.info(f"Removed inactive token: {symbol}")
                    
                    # Log monitoring status
                    active_trades = sum(1 for t in self.active_tokens.values() if t.active_trade)
                    self.logger.info(
                        f"\nMonitoring Status:"
                        f"\n  Active Tokens: {len(self.active_tokens)}"
                        f"\n  Active Trades: {active_trades}"
                        f"\n  Total Trades: {self.TRADE_COUNT}"
                        f"\n  Profitable Trades: {self.profitable_trades}"
                        f"\n  Stop Losses: {self.stop_losses_hit}"
                    )
                    
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(60)

    def save_monitoring_stats(self):
        """Save comprehensive monitoring statistics"""
        stats = {
            'session_timestamp': self.session_timestamp,
            'total_tokens_monitored': len(self.monitored_tokens),
            'total_trades_executed': self.TRADE_COUNT,
            'active_tokens': len(self.active_tokens),
            'inactive_tokens_removed': self.total_tokens_seen - len(self.active_tokens),
            'active_trades': sum(1 for t in self.active_tokens.values() if t.active_trade),
            'profitable_trades': self.profitable_trades,
            'stop_losses_hit': self.stop_losses_hit,
            'monitoring_duration_hours': (
                datetime.now() - datetime.strptime(self.session_timestamp, "%Y%m%d_%H%M%S")
            ).total_seconds() / 3600
        }
        
        stats_file = os.path.join(self.output_dir, 'monitoring_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

async def main():
    """Main entry point"""
    monitor = MarketMonitor()
    
    try:
        await monitor.monitor_market()
    except KeyboardInterrupt:
        print("\nShutting down market monitor...")
        monitor.save_monitoring_stats()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        monitor.logger.error(f"Fatal error: {str(e)}")
    finally:
        monitor.save_monitoring_stats()

if __name__ == "__main__":
    asyncio.run(main())