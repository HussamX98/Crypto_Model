import aiohttp
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import os
import importlib.util
from typing import List, Optional, Dict
import logging
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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


# Get absolute path to config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
config_path = os.path.join(project_root, "config.py")

# Import config.py using spec
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

API_KEY = config.API_KEY

# Debug prints
print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"Config path: {config_path}")
print(f"API Key loaded: {API_KEY is not None}")

class HistoricalDataCollector:
    def __init__(self, base_url: str, headers: dict, max_concurrent: int = 5):
        self.base_url = base_url
        self.headers = headers
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.logger = logging.getLogger("HistoricalDataCollector")
        self.logger.setLevel(logging.INFO)
        # Add handler to output logs to console
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)  # Set a total timeout of 30 seconds
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_with_retry(self, url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"Attempting request to {url} (attempt {attempt})")
                async with self.session.get(url, headers=self.headers, params=params) as response:
                    self.logger.info(f"Response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        self.logger.error(f"Request failed with status {response.status}")
            except asyncio.TimeoutError:
                self.logger.error(f"Attempt {attempt} failed due to timeout.")
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed with exception: {e}")
            await asyncio.sleep(0.5 * attempt)  # Exponential back-off
        self.logger.error(f"All {max_retries} attempts failed for URL {url}")
        return None

    async def get_token_price_history(self, token_address: str) -> tuple[pd.DataFrame, Optional[dict]]:
        """Get minute-by-minute price history for a token"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=30)
        
        self.logger.info(f"Fetching minute-by-minute price history for {token_address} from {start_time} to {end_time}")
        
        # Break into 2-hour chunks
        chunk_size = timedelta(hours=2)
        current_start = start_time
        all_data = []
        
        while current_start < end_time:
            current_end = min(current_start + chunk_size, end_time)
            
            params = {
                'address': token_address,
                'address_type': 'token',
                'type': '1m',  # Changed from '1' to '1m'
                'time_from': str(int(current_start.timestamp())),  # Convert to string
                'time_to': str(int(current_end.timestamp()))  # Convert to string
            }
            
            url = f"{self.base_url}/defi/history_price"
            
            # Debug log the params
            self.logger.debug(f"Request params: {params}")
            
            response_data = await self.fetch_with_retry(url, params)
            
            if response_data and response_data.get('success'):
                items = response_data['data'].get('items', [])
                if items:
                    chunk_df = pd.DataFrame(items)
                    chunk_df['timestamp'] = pd.to_datetime(chunk_df['unixTime'], unit='s')
                    all_data.append(chunk_df)
            else:
                self.logger.warning(f"Failed to get data for chunk: {current_start} to {current_end}")
                if response_data:
                    self.logger.debug(f"Response: {response_data}")
            
            current_start = current_end
            await asyncio.sleep(1)  # Rate limiting
        
        if all_data:
            df = pd.concat(all_data)
            df = df.sort_values('timestamp')
            df = df.drop_duplicates(subset=['timestamp'])
            
            summary = {
                'token_address': token_address,
                'start_time': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(df),
                'points_per_hour': len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600)
            }
            
            return df, summary
        
        return pd.DataFrame(), None
    
    async def save_minute_data(self, df: pd.DataFrame, token_address: str, directories: Dict[str, str], timestamp: str):
        """Save minute-level price data"""
        if df.empty:
            return
            
        # Save raw price data
        price_file = os.path.join(
            directories['price_data'], 
            f'{token_address}_minute_prices_{timestamp}.csv'
        )
        df.to_csv(price_file, index=False)
        
        # Convert timestamps to string format for JSON serialization
        summary = {
            'token_address': token_address,
            'start_time': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
            'total_minutes': len(df),
            'avg_price': float(df['value'].mean()),
            'min_price': float(df['value'].min()),
            'max_price': float(df['value'].max()),
            'price_volatility': float(df['value'].std() / df['value'].mean())
        }
        
        summary_file = os.path.join(
            directories['price_data'], 
            f'{token_address}_summary_{timestamp}.json'
        )
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, cls=CustomJSONEncoder)
            
        self.logger.info(f"Saved minute data for {token_address}")
        self.logger.info(f"Total minutes: {len(df)}")
        self.logger.info(f"Time range: {summary['start_time']} to {summary['end_time']}")

    async def collect_historical_data(self, token_addresses: List[str]) -> List[tuple[pd.DataFrame, Optional[dict]]]:
        self.logger.info(f"Starting collection for {len(token_addresses)} tokens")
        tasks = [self.fetch_with_semaphore(addr) for addr in token_addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                self.logger.error(f"Task for token {token_addresses[i]} failed with error: {r}")
            elif r[0] is not None and not r[0].empty and r[1] is not None:
                valid_results.append(r)
            else:
                self.logger.warning(f"No valid data for token {token_addresses[i]}")
        return valid_results

    async def fetch_with_semaphore(self, token_address: str) -> tuple[pd.DataFrame, Optional[dict]]:
        async with self.semaphore:
            self.logger.info(f"Fetching data for token {token_address}")
            result = await self.get_token_price_history(token_address)
            return result

def load_token_addresses() -> List[str]:
    """Load tokens from existing spikes file"""
    spikes_file = os.path.join(project_root, "data", "historical_tokens", "spikes_20241030_123758.csv")
    if os.path.exists(spikes_file):
        df = pd.read_csv(spikes_file)
        # Filter for significant spikes (5x or greater)
        df = df[df['increase'] >= 5]
        # Get unique token addresses
        tokens = df['token_address'].unique().tolist()
        print(f"Loaded {len(tokens)} unique tokens with 5x+ spikes")
        return tokens
    else:
        raise FileNotFoundError(f"Spikes file not found at: {spikes_file}")

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("\nStarting data collection process...")
    print(f"Base directory: {base_dir}")
    
    try:
        # Setup directories
        directories, run_directories, timestamp = setup_directories(base_dir)
        print("\nCreated directories:")
        for name, path in directories.items():
            print(f"{name}: {path}")
        
        base_url = "https://public-api.birdeye.so"
        headers = {
            "x-api-key": API_KEY
        }
        print("\nAPI Configuration:")
        print(f"Base URL: {base_url}")
        print(f"Headers configured: {'x-api-key' in headers}")
        
        # Test with a known token first (SOL)
        test_token = "So11111111111111111111111111111111111111112"
        print(f"\nTesting with SOL token: {test_token}")
        
        async with HistoricalDataCollector(base_url, headers) as collector:
            # Try the test token first
            result, summary = await collector.get_token_price_history(test_token)
            if not result.empty:
                print("\nSuccessfully retrieved SOL price history!")
                print(f"Data points: {len(result)}")
                if len(result) > 0:
                    print("\nFirst few rows of data:")
                    print(result.head())
                await collector.save_minute_data(result, test_token, directories, timestamp)
                print(f"Saved SOL data to {directories['price_data']}")
            else:
                print("\nFailed to retrieve SOL price history")
                return
            
            # Then proceed with your token list
            token_addresses = load_token_addresses()
            print(f"\nFound {len(token_addresses)} tokens to process")
            
            results = await collector.collect_historical_data(token_addresses)
            
            successful_collections = 0
            for token_data, token_summary in results:
                if token_data is not None and not token_data.empty:
                    token_address = token_summary['token_address']
                    await collector.save_minute_data(token_data, token_address, directories, timestamp)
                    successful_collections += 1
                    print(f"Processed {successful_collections}/{len(token_addresses)} tokens")
            
            print(f"\nData collection complete. Results saved in: {directories['price_data']}")
            print("\nSummary of collection:")
            print(f"Total tokens processed: {len(results)}")
            print(f"Successful collections: {successful_collections}")
            
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\nStarting script...")
    asyncio.run(main())