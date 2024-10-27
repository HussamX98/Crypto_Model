import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
from typing import Optional, Dict, List, Any
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import get_headers, API_KEY

class DataCollector:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.base_url = "https://public-api.birdeye.so"
        self.session = self._create_session()
        self.run_dir = self._create_run_dir()
        self._setup_logging()
        
    def _create_run_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.base_dir, "data", "runs", f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        # Create subdirectories for different types of data
        for subdir in ['spikes', 'price_data', 'trades', 'token_info']:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
        return run_dir

    def _setup_logging(self):
        log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'data_collector.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataCollector')

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, headers=get_headers(), params=params)
            response.raise_for_status()
            
            if not response.text.strip():
                self.logger.warning(f"Empty response received for {url}")
                return None
                
            data = response.json()
            if not data.get('success', False):
                self.logger.warning(f"API returned success=false for {url}")
                return None
                
            return data.get('data')
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response for {url}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {str(e)}")
            return None

    def find_5x_increases(self, token_address: str, days_back: int = 30) -> List[Dict]:
        """Find instances of 5x price increases within 1 hour"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Fetch OHLCV data
        endpoint = "/defi/history_price"
        params = {
            "address": token_address,
            "address_type": "token",
            "type": "1m",
            "time_from": int(start_time.timestamp()),
            "time_to": int(end_time.timestamp())
        }
        
        data = self._make_request(endpoint, params)
        if not data or 'items' not in data:
            self.logger.warning(f"No price history data found for {token_address}")
            return []
            
        df = pd.DataFrame(data['items'])
        if df.empty:
            return []
            
        df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
        df = df.sort_values('timestamp')
        
        spikes = []
        window_size = 60  # 1 hour in minutes
        
        # Save price data for token
        price_file = os.path.join(self.run_dir, 'price_data', f'{token_address}_price_history.csv')
        df.to_csv(price_file, index=False)
        
        # Scan through the data using a rolling window
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i+window_size]
            start_value = window.iloc[0]['value']
            
            if start_value == 0 or pd.isna(start_value):
                continue
                
            max_value = window['value'].max()
            
            if max_value >= 5 * start_value:
                spike_idx = window['value'].idxmax()
                spike_time = df.loc[spike_idx, 'timestamp']
                
                spike_info = {
                    'token_address': token_address,
                    'spike_time': spike_time,
                    'start_time': window.iloc[0]['timestamp'],
                    'price_increase': max_value / start_value,
                    'start_price': start_value,
                    'end_price': max_value
                }
                
                self.logger.info(f"Found 5x spike for {token_address}:")
                self.logger.info(f"  Time: {spike_time}")
                self.logger.info(f"  Increase: {spike_info['price_increase']:.2f}x")
                self.logger.info(f"  Start price: {start_value}")
                self.logger.info(f"  End price: {max_value}")
                
                spikes.append(spike_info)
                
                # Save individual spike data
                self._save_spike_data(token_address, spike_info)
                
                # Collect and save pre-spike data
                self.collect_pre_spike_data(token_address, spike_time)
                
                # Skip ahead to avoid counting the same spike multiple times
                i += window_size - 1
        
        return spikes

    def _save_spike_data(self, token_address: str, spike_info: Dict):
        """Save individual spike data to file"""
        spike_time_str = spike_info['spike_time'].strftime('%Y%m%d_%H%M%S')
        filename = f"{token_address}_spike_{spike_time_str}.json"
        filepath = os.path.join(self.run_dir, 'spikes', filename)
        
        serializable_info = spike_info.copy()
        serializable_info['spike_time'] = spike_info['spike_time'].isoformat()
        serializable_info['start_time'] = spike_info['start_time'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(serializable_info, f, indent=2)

    def collect_pre_spike_data(self, token_address: str, spike_time: datetime) -> Dict[str, Any]:
        """Collect comprehensive data for 15 minutes before a spike"""
        start_time = spike_time - timedelta(minutes=15)
        
        # Fetch price data
        endpoint = "/defi/history_price"
        params = {
            "address": token_address,
            "address_type": "token",
            "type": "1m",
            "time_from": int(start_time.timestamp()),
            "time_to": int(spike_time.timestamp())
        }
        price_data = self._make_request(endpoint, params)
        
        # Fetch trades data
        endpoint = "/defi/txs/token"
        params = {
            "address": token_address,
            "tx_type": "swap",
            "time_from": int(start_time.timestamp()),
            "time_to": int(spike_time.timestamp())
        }
        trades_data = self._make_request(endpoint, params)
        
        # Fetch token overview
        endpoint = "/defi/token_overview"
        params = {"address": token_address}
        token_data = self._make_request(endpoint, params)
        
        data = {}
        spike_time_str = spike_time.strftime('%Y%m%d_%H%M%S')
        
        if price_data and 'items' in price_data:
            df = pd.DataFrame(price_data['items'])
            df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
            data['price'] = df
            
            # Save price data
            filename = f"{token_address}_pre_spike_{spike_time_str}_price.csv"
            df.to_csv(os.path.join(self.run_dir, 'price_data', filename), index=False)
            
        if trades_data and 'items' in trades_data:
            df = pd.DataFrame(trades_data['items'])
            if 'blockUnixTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['blockUnixTime'], unit='s')
            data['trades'] = df
            
            # Save trades data
            filename = f"{token_address}_pre_spike_{spike_time_str}_trades.csv"
            df.to_csv(os.path.join(self.run_dir, 'trades', filename), index=False)
            
        if token_data:
            data['token_info'] = token_data
            
            # Save token info
            filename = f"{token_address}_pre_spike_{spike_time_str}_token_info.json"
            with open(os.path.join(self.run_dir, 'token_info', filename), 'w') as f:
                json.dump(token_data, f, indent=2)
            
        return data

    def process_tokens(self, token_list: List[str]) -> None:
        """Process all tokens to find and analyze 5x increases"""
        spikes_all = []
        processed_tokens = []
        failed_tokens = []
        
        total_tokens = len(token_list)
        self.logger.info(f"Starting processing of {total_tokens} tokens")
        
        for i, token_address in enumerate(token_list, 1):
            self.logger.info(f"Processing token {i}/{total_tokens}: {token_address}")
            
            try:
                # Find 5x increases
                spikes = self.find_5x_increases(token_address)
                
                if spikes:
                    self.logger.info(f"Found {len(spikes)} spikes for {token_address}")
                    spikes_all.extend(spikes)
                
                processed_tokens.append(token_address)
                
            except Exception as e:
                self.logger.error(f"Error processing {token_address}: {str(e)}")
                failed_tokens.append(token_address)
            
            time.sleep(1)  # Rate limiting
        
        # Save all spikes to a single file
        if spikes_all:
            spikes_df = pd.DataFrame(spikes_all)
            output_file = os.path.join(self.run_dir, '5x_price_increases.csv')
            spikes_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved {len(spikes_all)} spike instances to {output_file}")
        
        # Save processing summary
        summary = {
            'total_tokens': total_tokens,
            'processed_tokens': len(processed_tokens),
            'failed_tokens': len(failed_tokens),
            'total_spikes': len(spikes_all),
            'failed_token_addresses': failed_tokens
        }
        
        with open(os.path.join(self.run_dir, 'processing_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("\nProcessing Summary:")
        self.logger.info(f"Total tokens processed: {summary['processed_tokens']}/{summary['total_tokens']}")
        self.logger.info(f"Failed tokens: {summary['failed_tokens']}")
        self.logger.info(f"Total spikes found: {summary['total_spikes']}")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    
    token_list_path = os.path.join(base_dir, 'token_list.yaml')
    with open(token_list_path, 'r') as f:
        tokens = yaml.safe_load(f)['tokens']
    
    token_addresses = [token.split('#')[0].strip() for token in tokens]
    collector = DataCollector(base_dir)
    collector.process_tokens(token_addresses)

if __name__ == "__main__":
    main()