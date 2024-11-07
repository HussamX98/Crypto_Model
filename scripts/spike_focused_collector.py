import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional

class SpikeFocusedCollector:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = self._setup_logging()
        self.headers = {
            "accept": "application/json",
            "x-chain": "solana"
        }
        self.base_url = "https://public-api.birdeye.so"

    def _setup_logging(self):
        logger = logging.getLogger('SpikeFocusedCollector')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    async def get_token_price_history(self, token_address: str, 
                                    start_time: datetime, 
                                    end_time: datetime) -> pd.DataFrame:
        """Get minute-by-minute price data for a specific time window"""
        url = f"{self.base_url}/defi/history_price"
        
        params = {
            "address": token_address,
            "type": "1m",
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
                            return df
                    
                    self.logger.warning(f"No data for {token_address} between {start_time} and {end_time}")
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"Error collecting data: {str(e)}")
            return pd.DataFrame()

    async def collect_spike_focused_data(self, spike_window_hours: int = 2):
        """Collect data focused around known spike periods"""
        # Load spikes data
        spikes_file = os.path.join(self.base_dir, 'data', 'historical_tokens', 'spikes_20241030_123758.csv')
        spikes_df = pd.read_csv(spikes_file)
        spikes_df['spike_time'] = pd.to_datetime(spikes_df['spike_time'])
        
        # Filter to more recent spikes
        recent_cutoff = datetime.now() - timedelta(days=7)  # Last 7 days
        recent_spikes = spikes_df[spikes_df['spike_time'] >= recent_cutoff]
        
        self.logger.info(f"Found {len(recent_spikes)} recent spikes to analyze")
        
        output_dir = os.path.join(self.base_dir, 'data', 'historical_prices_1m_spikes')
        os.makedirs(output_dir, exist_ok=True)
        
        successful = 0
        failed = 0
        
        for _, spike in recent_spikes.iterrows():
            try:
                token_address = spike['token_address']
                spike_time = spike['spike_time']
                
                # Get data for window around spike
                start_time = spike_time - timedelta(hours=spike_window_hours)
                end_time = spike_time + timedelta(hours=spike_window_hours)
                
                self.logger.info(f"Collecting data for {token_address} around {spike_time}")
                
                price_data = await self.get_token_price_history(token_address, start_time, end_time)
                
                if not price_data.empty:
                    # Save data
                    output_file = os.path.join(output_dir, 
                                            f"{token_address}_spike_{spike_time.strftime('%Y%m%d_%H%M')}.csv")
                    price_data.to_csv(output_file, index=False)
                    
                    # Save summary with verification
                    max_increase = (price_data['value'].max() / price_data['value'].min() - 1) * 100
                    summary = {
                        'token_address': token_address,
                        'spike_time': spike_time.isoformat(),
                        'data_points': len(price_data),
                        'min_price': float(price_data['value'].min()),
                        'max_price': float(price_data['value'].max()),
                        'expected_increase': float(spike['increase']),
                        'actual_increase': float(max_increase),
                        'data_coverage_minutes': len(price_data),
                        'coverage_percent': (len(price_data) / (spike_window_hours * 2 * 60)) * 100
                    }
                    
                    summary_file = os.path.join(output_dir, 
                                            f"{token_address}_spike_{spike_time.strftime('%Y%m%d_%H%M')}_summary.json")
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=4)
                    
                    successful += 1
                    self.logger.info(f"Successfully collected data showing {max_increase:.2f}% increase")
                else:
                    failed += 1
                
                # More aggressive rate limiting
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Error processing {token_address}: {str(e)}")
                failed += 1
                continue
            
        self.logger.info(f"Collection complete. Successful: {successful}, Failed: {failed}")

async def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    collector = SpikeFocusedCollector(base_dir)
    await collector.collect_spike_focused_data()

if __name__ == "__main__":
    asyncio.run(main())