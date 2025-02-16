import os
import sys
import json
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from config import API_KEY
except ImportError:
    raise ImportError(f"Failed to import API_KEY from config.py in {project_root}")

@dataclass
class EndpointMetrics:
    """Store metrics for each endpoint"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_response_time: float = 0
    errors: List[str] = None
    data_completeness: Dict[str, int] = None
    last_update_time: Optional[datetime] = None
    
    def __post_init__(self):
        self.errors = []
        self.data_completeness = {}

class BirdeyeAPITester:
    def __init__(self, test_duration_minutes: int = 10):
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_duration = timedelta(minutes=test_duration_minutes)
        self.test_start_time = None
        self.setup_logging()
        self.test_tokens = []
        self.headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "x-api-key": API_KEY
        }
        
        # Track metrics for each endpoint
        self.endpoint_metrics = {
            'new_listing': EndpointMetrics(),
            'token_list': EndpointMetrics(),
            'market_data': EndpointMetrics(),
            'trade_data': EndpointMetrics(),
            'ohlcv': EndpointMetrics(),
            'price_history': EndpointMetrics(),
            'price_volume_single': EndpointMetrics()  # Add this endpoint
        }
        
        # Store price update data
        self.price_updates = {}

    def setup_logging(self):
        """Configure logging with detailed output"""
        log_dir = os.path.join(project_root, 'api_test_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create test results directory
        self.results_dir = os.path.join(log_dir, self.session_timestamp)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger = logging.getLogger('BirdeyeAPITester')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(
            os.path.join(self.results_dir, 'api_test.log')
        )
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    async def fetch_with_retry(self, endpoint_type: str, url: str, 
                             params: Optional[Dict] = None,
                             max_retries: int = 3) -> Optional[Dict]:
        """Make API request with metrics tracking"""
        metrics = self.endpoint_metrics[endpoint_type]
        metrics.total_calls += 1
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers, params=params) as response:
                        response_time = time.time() - start_time
                        metrics.total_response_time += response_time
                        
                        response_text = await response.text()
                        
                        if response.status == 200:
                            metrics.successful_calls += 1
                            metrics.last_update_time = datetime.now()
                            return json.loads(response_text)
                        else:
                            metrics.failed_calls += 1
                            error_msg = f"Status {response.status}: {response_text}"
                            metrics.errors.append(error_msg)
                            self.logger.error(f"Request failed: {error_msg}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                            continue
                            
            except Exception as e:
                metrics.failed_calls += 1
                metrics.errors.append(str(e))
                self.logger.error(f"Request error: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
        return None

    def check_data_completeness(self, endpoint_type: str, data: Dict):
        """Check completeness of response data"""
        metrics = self.endpoint_metrics[endpoint_type]
        
        def check_nested(d: Dict, path: str = ""):
            for k, v in d.items():
                full_path = f"{path}.{k}" if path else k
                if v is None or v == "":
                    metrics.data_completeness[full_path] = \
                        metrics.data_completeness.get(full_path, 0) + 1
                elif isinstance(v, dict):
                    check_nested(v, full_path)
                elif isinstance(v, list) and v and isinstance(v[0], dict):
                    check_nested(v[0], f"{full_path}[0]")
        
        check_nested(data)


    async def test_token_discovery(self):
        try:
            self.logger.info("\n=== Testing Token Discovery Endpoints ===")
            
            new_listing_response = await self.fetch_with_retry(
                'new_listing',
                "https://public-api.birdeye.so/defi/v2/tokens/new_listing",
                params={"limit": 10}
            )
            
            if new_listing_response and new_listing_response.get('success'):
                self.check_data_completeness('new_listing', new_listing_response)
                tokens = new_listing_response.get('data', {}).get('items', [])
                self.logger.info(f"\nNew Listing Endpoint returned {len(tokens)} tokens")
                
                # Store tokens first
                for token in tokens[:5]:
                    try:
                        token_symbol = token.get('symbol', '').encode('ascii', 'ignore').decode()
                        if token_symbol and token.get('address'):
                            self.test_tokens.append({
                                'address': token.get('address'),
                                'symbol': token_symbol,
                                'name': token.get('name', '').encode('ascii', 'ignore').decode(),
                                'source': 'new_listing'
                            })
                    except Exception as e:
                        self.logger.debug(f"Skipping token due to encoding: {str(e)}")

            # Now validate tokens with current prices
            validated_tokens = []
            for token in self.test_tokens:
                price_data = await self.fetch_with_retry(
                    'price_volume_single',  # Updated endpoint key
                    "https://public-api.birdeye.so/defi/v3/token/market-data",  # Use market-data endpoint instead
                    params={"address": token['address']}
                )
                
                if price_data and price_data.get('success'):
                    data = price_data.get('data', {})
                    if data and float(data.get('price', 0)) > 0:
                        validated_tokens.append(token)
                        self.logger.info(f"Validated token: {token['symbol']}")
                
                await asyncio.sleep(0.5)  # Rate limiting
            
            self.test_tokens = validated_tokens
            self.logger.info(f"Found {len(self.test_tokens)} valid trading tokens")
            
        except Exception as e:
            self.logger.error(f"Error in token discovery: {str(e)}")
            

    async def test_market_data(self, token_address: str):
        """Test market data endpoints with improved error handling"""
        try:
            self.logger.info(f"\n=== Testing Market Data for {token_address} ===")
            
            # Get market data
            market_data = await self.fetch_with_retry(
                'market_data',
                "https://public-api.birdeye.so/defi/v3/token/market-data",
                params={"address": token_address}
            )
            
            if market_data and market_data.get('success'):
                data = market_data.get('data', {})
                if data:
                    try:
                        price = float(data.get('price', 0))
                        marketcap = float(data.get('marketcap', 0))
                        liquidity = float(data.get('liquidity', 0))
                        
                        self.logger.info("\nMarket Data:")
                        self.logger.info(f"Price: ${price:.8f}")
                        self.logger.info(f"Market Cap: ${marketcap:.2f}")
                        self.logger.info(f"Liquidity: ${liquidity:.2f}")
                        
                        # Store price update
                        if token_address not in self.price_updates:
                            self.price_updates[token_address] = []
                        self.price_updates[token_address].append({
                            'timestamp': datetime.now(),
                            'price': price,
                            'source': 'market_data'
                        })
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"Error processing market data values: {str(e)}")
                else:
                    self.logger.error("No data in market data response")
            else:
                self.logger.error("Failed to get market data")
        
        except Exception as e:
            self.logger.error(f"Error in market data test: {str(e)}")


    async def test_price_data(self, token_address: str):
        """Test price data endpoints with enhanced error handling"""
        try:
            self.logger.info(f"\n=== Testing Price Data for {token_address} ===")
            
            current_time = int(datetime.now().timestamp())
            start_time = current_time - (30 * 60)  # 30 minutes history
            
            # Test OHLCV endpoint
            ohlcv_response = await self.fetch_with_retry(
                'ohlcv',
                "https://public-api.birdeye.so/defi/ohlcv",
                params={
                    "address": token_address,
                    "type": "1m",
                    "time_from": start_time,
                    "time_to": current_time
                }
            )
            
            if ohlcv_response and ohlcv_response.get('success'):
                items = ohlcv_response.get('data', {}).get('items', [])
                if items:
                    try:
                        self.logger.info(f"\nOHLCV Data Points: {len(items)}")
                        self.logger.info("Latest OHLCV:")
                        latest = items[-1]
                        self.logger.info(f"Time: {datetime.fromtimestamp(latest['unixTime'])}")
                        self.logger.info(f"Open: ${float(latest['o']):.8f}")
                        self.logger.info(f"Close: ${float(latest['c']):.8f}")
                        
                        # Store valid prices
                        if token_address not in self.price_updates:
                            self.price_updates[token_address] = []
                        for item in items:
                            self.price_updates[token_address].append({
                                'timestamp': datetime.fromtimestamp(item['unixTime']),
                                'price': float(item['c']),
                                'source': 'ohlcv'
                            })
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"Error processing OHLCV data: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error in price data test: {str(e)}")

    def analyze_results(self):
        """Analyze test results and generate report"""
        report = []
        report.append("\n=== BIRDEYE API TEST RESULTS ===\n")
        
        # Endpoint Performance
        report.append("Endpoint Performance:")
        for endpoint, metrics in self.endpoint_metrics.items():
            success_rate = (metrics.successful_calls / metrics.total_calls * 100 
                          if metrics.total_calls > 0 else 0)
            avg_response_time = (metrics.total_response_time / metrics.total_calls 
                               if metrics.total_calls > 0 else 0)
            
            report.append(f"\n{endpoint}:")
            report.append(f"  Success Rate: {success_rate:.1f}%")
            report.append(f"  Avg Response Time: {avg_response_time:.3f}s")
            report.append(f"  Total Calls: {metrics.total_calls}")
            report.append(f"  Failed Calls: {metrics.failed_calls}")
            
            if metrics.errors:
                report.append("  Common Errors:")
                for error in set(metrics.errors[:5]):
                    report.append(f"    - {error}")
        
        # Price Update Analysis
        report.append("\nPrice Update Analysis:")
        for token_address, updates in self.price_updates.items():
            df = pd.DataFrame(updates)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Analyze update frequency by source
                report.append(f"\n{token_address}:")
                for source in df['source'].unique():
                    source_data = df[df['source'] == source]
                    if len(source_data) > 1:
                        update_intervals = source_data['timestamp'].diff().dropna()
                        avg_interval = update_intervals.mean().total_seconds()
                        report.append(f"  {source}:")
                        report.append(f"    Avg Update Interval: {avg_interval:.1f}s")
                        report.append(f"    Updates Received: {len(source_data)}")
                
# Check price consistency across sources
                    if len(df['source'].unique()) > 1:
                        report.append("  Price Consistency:")
                        pivot = df.pivot(columns='source', values='price')
                        for source1 in pivot.columns:
                            for source2 in pivot.columns:
                                if source1 < source2:  # Avoid duplicate comparisons
                                    diff_pct = abs(pivot[source1] - pivot[source2]) / pivot[source1] * 100
                                    avg_diff = diff_pct.mean()
                                    report.append(f"    {source1} vs {source2}: {avg_diff:.2f}% avg difference")

        # Data Completeness Analysis
        report.append("\nData Completeness Analysis:")
        for endpoint, metrics in self.endpoint_metrics.items():
            if metrics.data_completeness:
                report.append(f"\n{endpoint}:")
                for field, missing_count in metrics.data_completeness.items():
                    missing_rate = missing_count / metrics.successful_calls * 100
                    report.append(f"  {field}: {missing_rate:.1f}% missing")

        # Save report
        report_text = '\n'.join(report)
        report_path = os.path.join(self.results_dir, 'test_results.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        self.logger.info(report_text)
        return report_text

    async def run_tests(self):
        """Run comprehensive API endpoint tests"""
        self.logger.info(f"Starting Birdeye API Tests (Duration: {self.test_duration.total_seconds()/60:.1f} minutes)...")
        self.test_start_time = datetime.now()
        
        while datetime.now() - self.test_start_time < self.test_duration:
            # Test token discovery
            await self.test_token_discovery()
            
            # Test other endpoints for each test token
            for token in self.test_tokens:
                self.logger.info(f"\nTesting endpoints for {token['symbol']} ({token['address']})")
                await self.test_market_data(token['address'])
                await self.test_price_data(token['address'])
                await asyncio.sleep(1)  # Rate limiting
            
            # Wait between test cycles
            await asyncio.sleep(10)
        
        # Analyze results
        self.analyze_results()
        self.logger.info("\nAPI Tests completed!")

async def main():
    # Run 10-minute test
    tester = BirdeyeAPITester(test_duration_minutes=10)
    await tester.run_tests()

if __name__ == "__main__":
    asyncio.run(main())