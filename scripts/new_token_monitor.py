import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Optional, Tuple
import logging
import json
import time
import traceback

class TokenDataCollector:
    """Collects and analyzes token data for price spike detection."""
    
    def __init__(self, base_dir: str):
        """Initialize the collector with configuration and constants."""
        self.base_dir = base_dir
        self.logger = self._setup_logging()
        
        # Import config for API key
        sys.path.append(base_dir)
        from config import API_KEY
        self.api_key = API_KEY
        
        # API Configuration
        self.headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        self.base_url = 'https://public-api.birdeye.so'
        
        # Rate limiting configuration
        self.MAX_REQUESTS_PER_MINUTE = 60
        self.DELAY_BETWEEN_REQUESTS = 1
        self.request_timestamps = []
        
        # Token filtering thresholds
        self.MAX_TOKENS_TO_ANALYZE = 1000
        self.MIN_MARKET_CAP = 10000
        self.MIN_LIQUIDITY = 500
        self.MIN_HOLDER_COUNT = 10
        self.MIN_VOLUME_24H = 100
        self.MAX_AGE_DAYS = 30
        
        # Progress tracking
        self.progress_counter = 0
        self.start_time = None

    # 1. Setup and Utility Methods
    def _setup_logging(self):
        """Configure logging for the collector."""
        logger = logging.getLogger('TokenDataCollector')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    async def _handle_rate_limit(self):
        """Handle API rate limiting."""
        current_time = time.time()
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        
        if len(self.request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
            self.request_timestamps = []
        
        self.request_timestamps.append(current_time)

    def _update_progress(self):
        """Update and display progress information."""
        self.progress_counter += 1
        if self.start_time is None:
            self.start_time = time.time()
            
        elapsed = time.time() - self.start_time
        tokens_per_second = self.progress_counter / elapsed if elapsed > 0 else 0
        
        if self.progress_counter % 10 == 0:
            self.logger.info(
                f"Progress: {self.progress_counter}/{self.MAX_TOKENS_TO_ANALYZE} tokens"
                f" ({tokens_per_second:.1f} tokens/sec)"
                f" - Elapsed: {elapsed/60:.1f} minutes"
            )

    # 2. Token Data Fetching Methods
    async def get_historical_tokens(self, min_liquidity: float = 100) -> List[Dict]:
        """Get list of historical tokens with validation."""
        url = f"{self.base_url}/defi/tokenlist"
        headers = {
            **self.headers,
            'x-chain': 'solana'
        }
        
        all_tokens = []
        offset = 0
        batch_size = 50
        
        self.logger.info(f"Fetching up to {self.MAX_TOKENS_TO_ANALYZE} tokens...")
        
        while len(all_tokens) < self.MAX_TOKENS_TO_ANALYZE:
            try:
                await self._handle_rate_limit()
                
                params = {
                    "sort_by": "v24hUSD",
                    "sort_type": "desc",
                    "offset": offset,
                    "limit": batch_size,
                    "min_liquidity": min_liquidity
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if not data.get('success') or not data.get('data', {}).get('tokens'):
                                break
                                
                            batch_tokens = data['data']['tokens']
                            if not batch_tokens:
                                break
                                
                            # Validate tokens
                            valid_tokens = []
                            for token in batch_tokens:
                                is_valid, reason = await self._validate_token(token)
                                if is_valid:
                                    valid_tokens.append(token)
                                    
                            all_tokens.extend(valid_tokens)
                            self.logger.info(f"Retrieved {len(all_tokens)} total tokens")
                            
                            if len(batch_tokens) < batch_size:
                                break
                                
                            offset += batch_size
                        else:
                            self.logger.error(f"Token list request failed with status {response.status}")
                            break
                            
                await asyncio.sleep(self.DELAY_BETWEEN_REQUESTS)
                
            except Exception as e:
                self.logger.error(f"Error fetching token list: {str(e)}")
                break
                
        return all_tokens[:self.MAX_TOKENS_TO_ANALYZE]

    async def get_token_creation_info(self, token_address: str) -> Optional[Dict]:
        """Get token creation information."""
        url = f"{self.base_url}/defi/token_creation_info"
        headers = {
            **self.headers,
            'x-chain': 'solana'
        }
        params = {'address': token_address}
        
        try:
            await self._handle_rate_limit()
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success'):
                            return data.get('data')
        except Exception as e:
            self.logger.debug(f"Error fetching creation info for {token_address}: {str(e)}")
        return None

    async def get_token_price_history(self, token_address: str, days_back: int = 30) -> List[Dict]:
        """Get price history for a token."""
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (days_back * 24 * 3600 * 1000)
        
        url = f"{self.base_url}/defi/history_price"
        params = {
            "address": token_address,
            "address_type": "token",
            "type": "1m",
            "time_from": str(int(start_time/1000)),
            "time_to": str(int(end_time/1000))
        }
        
        try:
            await self._handle_rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and data.get('data', {}).get('items'):
                            items = data['data']['items']
                            self.logger.debug(f"Got {len(items)} price points for {token_address}")
                            return items
                        else:
                            self.logger.warning(f"No price data found for {token_address}")
                    else:
                        self.logger.error(f"Error {response.status} fetching price history")
                        
        except Exception as e:
            self.logger.error(f"Error fetching price history: {str(e)}")
        
        return []

    # 3. Token Validation Methods
    async def get_token_list(self, sort_by: str = 'v24hUSD', min_liquidity: float = 100) -> List[Dict]:
        """Get list of tokens sorted by 24h volume."""
        url = f"{self.base_url}/defi/tokenlist"
        headers = {
            **self.headers,
            'x-chain': 'solana'
        }
        
        all_tokens = []
        offset = 0
        batch_size = 50
        
        self.logger.info(f"Fetching up to {self.MAX_TOKENS_TO_ANALYZE} tokens...")
        
        while len(all_tokens) < self.MAX_TOKENS_TO_ANALYZE:
            try:
                await self._handle_rate_limit()
                
                params = {
                    "sort_by": sort_by,
                    "sort_type": "desc",
                    "offset": offset,
                    "limit": batch_size,
                    "min_liquidity": min_liquidity
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if not data.get('success') or not data.get('data', {}).get('tokens'):
                                break
                                
                            batch_tokens = data['data']['tokens']
                            if not batch_tokens:
                                break
                                
                            # Validate tokens
                            valid_tokens = []
                            for token in batch_tokens:
                                is_valid, reason = await self._validate_token(token)
                                if is_valid:
                                    valid_tokens.append(token)
                                    
                            all_tokens.extend(valid_tokens)
                            self.logger.info(f"Retrieved {len(all_tokens)} total tokens")
                            
                            if len(batch_tokens) < batch_size:
                                break
                                
                            offset += batch_size
                        else:
                            self.logger.error(f"Token list request failed with status {response.status}")
                            break
                            
                await asyncio.sleep(self.DELAY_BETWEEN_REQUESTS)
                
            except Exception as e:
                self.logger.error(f"Error fetching token list: {str(e)}")
                break
                
        return all_tokens[:self.MAX_TOKENS_TO_ANALYZE]

    async def _validate_token(self, token: Dict) -> Tuple[bool, str]:
        """Validate if a token should be included in analysis."""
        try:
            # Check basic token info
            token_address = token.get('address')
            if not token_address:
                return False, "Missing token address"
                
            # Check liquidity
            liquidity = token.get('liquidity', 0)
            if liquidity < self.MIN_LIQUIDITY:
                return False, f"Insufficient liquidity (${liquidity:.2f})"

            # Check volume
            volume_24h = token.get('v24hUSD', 0)
            if volume_24h < self.MIN_VOLUME_24H:
                return False, f"Insufficient 24h volume (${volume_24h:.2f})"
                
            # Calculate volume/liquidity ratio
            if liquidity > 0:
                volume_liquidity_ratio = volume_24h / liquidity
                self.logger.info(f"Token {token.get('symbol')} ({token_address}) passed validation:")
                self.logger.info(f"  Liquidity: ${liquidity:.2f}")
                self.logger.info(f"  24h Volume: ${volume_24h:.2f}")
                self.logger.info(f"  Volume/Liquidity: {volume_liquidity_ratio:.2f}")
                return True, "Valid"
            else:
                return False, "Zero liquidity"

        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    # 4. Price Analysis Methods
    def detect_price_spikes(self, price_data: List[Dict], window_minutes: int = 60) -> List[Dict]:
        """
        Detect instances of 5x price increases within 1 hour window.
        Based on the working logic from data_collector.py
        """
        spikes = []
        
        if not price_data:
            return spikes
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            if df.empty:
                return spikes
                
            # Sort by time and convert values
            df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.sort_values('timestamp')
            df = df.dropna()
            
            # Need at least window_size records
            if len(df) < 2:
                return spikes
                
            # Check each potential starting point
            for i in range(len(df) - 1):
                start_value = df.iloc[i]['value']
                start_time = df.iloc[i]['timestamp']
                
                # Skip if start value is zero
                if start_value == 0:
                    continue
                
                # Get end time for window
                end_time = start_time + pd.Timedelta(minutes=window_minutes)
                
                # Get price data in the window
                window_data = df[
                    (df['timestamp'] > start_time) & 
                    (df['timestamp'] <= end_time)
                ]
                
                if len(window_data) == 0:
                    continue
                    
                max_value = window_data['value'].max()
                
                # Check for 5x increase
                if max_value >= (5 * start_value):
                    # Get exact time of peak
                    peak_time = window_data.loc[window_data['value'] == max_value, 'timestamp'].iloc[0]
                    
                    spike_info = {
                        'spike_time': peak_time,
                        'start_price': float(start_value),
                        'max_price': float(max_value),
                        'increase': float(max_value / start_value),
                        'duration_minutes': (peak_time - start_time).total_seconds() / 60
                    }
                    
                    spikes.append(spike_info)
                    
                    self.logger.info(
                        f"Found {spike_info['increase']:.1f}x spike "
                        f"from ${spike_info['start_price']:.6f} to ${spike_info['max_price']:.6f} "
                        f"in {spike_info['duration_minutes']:.1f} minutes"
                    )
                    
                    # Skip to end of this window to avoid duplicate spikes
                    while i < len(df) and df.iloc[i]['timestamp'] <= end_time:
                        i += 1
            
        except Exception as e:
            self.logger.error(f"Error in spike detection: {str(e)}\n{traceback.format_exc()}")
        
        return spikes

    async def collect_historical_spikes(self, days_to_check: int = 30) -> pd.DataFrame:
        """Collect spikes from historical price data."""
        spikes = []
        total_tokens = 0
        processed_tokens = 0
        
        try:
            tokens = await self.get_token_list()
            total_tokens = len(tokens)
            self.logger.info(f"Found {total_tokens} tokens to analyze")
            
            for token in tokens:
                try:
                    token_address = token['address']
                    symbol = token.get('symbol', 'Unknown')
                    
                    self.logger.info(f"Checking {symbol} ({token_address}) for spikes")
                    
                    # Get price history
                    price_data = await self.get_token_price_history(token_address, days_to_check)
                    
                    # Skip if no price data
                    if not price_data:
                        continue
                    
                    # Find spikes
                    token_spikes = self.detect_price_spikes(price_data)
                    
                    # Add metadata to spikes
                    for spike in token_spikes:
                        spike['token_address'] = token_address
                        spike['symbol'] = symbol
                        spike['collection_time'] = datetime.now().isoformat()
                        spikes.append(spike)
                    
                    processed_tokens += 1
                    if processed_tokens % 10 == 0:
                        self.logger.info(f"Processed {processed_tokens}/{total_tokens} tokens")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            if spikes:
                self.logger.info(f"\nFound {len(spikes)} spikes across {processed_tokens} tokens")
                df = pd.DataFrame(spikes)
                
                # Convert spike times to ISO format strings
                df['spike_time'] = df['spike_time'].apply(lambda x: x.isoformat())
                
                return df
            else:
                self.logger.warning("No spikes found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in collection: {str(e)}")
            return pd.DataFrame()
    
    def save_results(self, df: pd.DataFrame):
        """Save collected data to CSV."""
        if df.empty:
            self.logger.warning("No spikes found to save")
            return
            
        output_dir = os.path.join(self.base_dir, 'data', 'historical_tokens')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'spikes_{timestamp}.csv')
        
        df.to_csv(output_file, index=False)
        self.logger.info(f"Saved {len(df)} spike records to {output_file}")

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    collector = TokenDataCollector(base_dir)
    
    try:
        collector.logger.info("Starting historical spike collection")
        spikes_df = await collector.collect_historical_spikes(days_to_check=30)
        collector.save_results(spikes_df)
        
    except Exception as e:
        collector.logger.error(f"Error in collection: {str(e)}")
    except KeyboardInterrupt:
        collector.logger.info("Collection cancelled by user")
        # Still try to save any results we have
        if 'spikes_df' in locals() and not spikes_df.empty:
            collector.save_results(spikes_df)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript stopped by user")