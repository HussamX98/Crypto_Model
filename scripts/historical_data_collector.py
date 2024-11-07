import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Optional
import logging
import json

class HistoricalDataCollector:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = self._setup_logging()
        sys.path.append(base_dir)
        from config import API_KEY
        self.api_key = API_KEY
        self.headers = {
            'X-API-KEY': self.api_key,
            'accept': 'application/json',
            'x-chain': 'solana'
        }
        self.base_url = 'https://public-api.birdeye.so'

    def _setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger('HistoricalDataCollector')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('data_collection.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger

    async def get_token_price_history(self, token_address: str, 
                                    start_time: datetime, 
                                    end_time: datetime) -> pd.DataFrame:
        """Get historical minute-by-minute price data"""
        url = f"{self.base_url}/defi/history_price"
        
        params = {
            "address": token_address,
            "address_type": "token",
            "type": "1m",  # 1-minute intervals
            "time_from": int(start_time.timestamp()),
            "time_to": int(end_time.timestamp())
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and data.get('data', {}).get('items'):
                            df = pd.DataFrame(data['data']['items'])
                            df['timestamp'] = pd.to_datetime(df['unixTime'], unit='s')
                            df = df.sort_values('timestamp')
                            if not df.empty:
                                self.logger.info(f"Successfully collected {len(df)} minute-by-minute price points for {token_address}")
                                return df
                        
                        # More detailed error logging
                        self.logger.warning(f"No price data found for {token_address} between "
                                        f"{start_time.strftime('%Y-%m-%d %H:%M')} and "
                                        f"{end_time.strftime('%Y-%m-%d %H:%M')}")
                    else:
                        self.logger.error(f"API returned status {response.status} for {token_address}")
                    
                    return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error collecting price history for {token_address}: {str(e)}")
            return pd.DataFrame()

    def _save_token_data(self, token_address: str, data: pd.DataFrame, 
                        start_time: datetime, end_time: datetime):
        """Save price history data to file"""
        if data.empty:
            return

        # Create separate directory for minute data
        price_dir = os.path.join(self.base_dir, 'data', 'historical_prices_1m')
        os.makedirs(price_dir, exist_ok=True)
        
        # Save with date range in filename
        start_str = start_time.strftime('%Y%m%d')
        end_str = end_time.strftime('%Y%m%d')
        filename = f'{token_address}_prices_{start_str}_{end_str}.csv'
        file_path = os.path.join(price_dir, filename)
        
        # Ensure we have all required columns
        required_cols = ['unixTime', 'value', 'timestamp', 'address']
        if not all(col in data.columns for col in required_cols):
            data['address'] = token_address  # Add address column if missing
        
        data.to_csv(file_path, index=False)
        self.logger.info(f"Saved minute data to {filename}")
        
        # Save summary stats in same directory
        stats = {
            'token_address': token_address,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'data_points': len(data),
            'data_interval': '1m',
            'min_price': float(data['value'].min()),
            'max_price': float(data['value'].max()),
            'avg_price': float(data['value'].mean()),
            'time_coverage': f"{len(data)} minutes out of {(end_time - start_time).total_seconds() / 60:.0f} possible minutes"
        }
        
        stats_file = os.path.join(price_dir, f'{token_address}_summary.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)

    async def collect_historical_data(self, token_addresses: List[str], 
                                    days_back: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect historical data for multiple tokens"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        all_data = {}
        successful_tokens = 0
        failed_tokens = 0
        
        for i, token in enumerate(token_addresses, 1):
            try:
                self.logger.info(f"Processing token {i}/{len(token_addresses)}: {token}")
                
                # Make a single request for the entire period first
                price_data = await self.get_token_price_history(token, start_time, end_time)
                
                if not price_data.empty:
                    all_data[token] = price_data
                    self._save_token_data(token, price_data, start_time, end_time)
                    successful_tokens += 1
                    self.logger.info(f"Successfully collected data for {token}")
                else:
                    failed_tokens += 1
                    self.logger.warning(f"No data found for {token}")
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Failed to process token {token}: {str(e)}")
                failed_tokens += 1
                continue
                
            # Progress update every 10 tokens
            if i % 10 == 0:
                self.logger.info(f"\nProgress Update:")
                self.logger.info(f"Processed: {i}/{len(token_addresses)} tokens")
                self.logger.info(f"Successful: {successful_tokens}")
                self.logger.info(f"Failed: {failed_tokens}\n")
        
        # Final summary
        self.logger.info(f"\nCollection Complete:")
        self.logger.info(f"Total tokens: {len(token_addresses)}")
        self.logger.info(f"Successful collections: {successful_tokens}")
        self.logger.info(f"Failed collections: {failed_tokens}")
        
        return all_data

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    collector = HistoricalDataCollector(base_dir)
    
    try:
        # Load tokens from spikes data
        spikes_file = os.path.join(base_dir, 'data', 'historical_tokens', 'spikes_20241030_123758.csv')
        spikes_df = pd.read_csv(spikes_file)
        unique_tokens = spikes_df['token_address'].unique()
        
        collector.logger.info(f"Found {len(unique_tokens)} unique tokens")
        
        # Create the new directory
        minute_data_dir = os.path.join(base_dir, 'data', 'historical_prices_1m')
        os.makedirs(minute_data_dir, exist_ok=True)
        
        # Collect minute data
        data = await collector.collect_historical_data(unique_tokens, days_back=30)
        
        # Save collection summary
        summary = {
            'collection_time': datetime.now().isoformat(),
            'total_tokens': len(unique_tokens),
            'successful_collections': len(data),
            'data_interval': '1m',
            'tokens_with_data': list(data.keys())
        }
        
        summary_file = os.path.join(minute_data_dir, 'minute_collection_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

    except Exception as e:
        collector.logger.error(f"Error in data collection: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())