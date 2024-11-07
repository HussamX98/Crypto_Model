import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

def verify_minute_data(base_dir: str, sample_token: str = None):
    """Verify the quality of collected minute data"""
    # Load spikes data
    spikes_file = os.path.join(base_dir, 'data', 'historical_tokens', 'spikes_20241030_123758.csv')
    spikes_df = pd.read_csv(spikes_file)
    
    print("\nSpikes data columns:", spikes_df.columns.tolist())
    
    # Load minute data directory
    minute_data_dir = os.path.join(base_dir, 'data', 'historical_prices_1m')
    
    if sample_token is None:
        # Just take the first token from the spikes data
        sample_token = spikes_df['token_address'].iloc[0]
    
    print(f"\nAnalyzing data for token: {sample_token}")
    
    # Load minute data for this token
    price_files = [f for f in os.listdir(minute_data_dir) if f.startswith(sample_token) and f.endswith('.csv')]
    if not price_files:
        print(f"No price data found for token {sample_token}")
        return None
        
    price_file = price_files[0]
    price_data = pd.read_csv(os.path.join(minute_data_dir, price_file))
    
    # Convert timestamp
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    price_data = price_data.sort_values('timestamp')
    
    # Calculate basic metrics
    price_data['minute_change'] = price_data['value'].pct_change()
    price_data['5min_change'] = (price_data['value'] / price_data['value'].shift(5)) - 1
    price_data['15min_change'] = (price_data['value'] / price_data['value'].shift(15)) - 1
    
    # Find largest moves
    largest_1min = price_data.nlargest(5, 'minute_change')
    largest_5min = price_data.nlargest(5, '5min_change')
    largest_15min = price_data.nlargest(5, '15min_change')
    
    print("\nData Quality Report")
    print("-" * 50)
    print(f"Total minutes of data: {len(price_data)}")
    print(f"Date range: {price_data['timestamp'].min()} to {price_data['timestamp'].max()}")
    
    print(f"\nLargest 1-minute price increases:")
    for _, row in largest_1min.iterrows():
        print(f"Time: {row['timestamp']}, Change: {row['minute_change']*100:.2f}%")
    
    print(f"\nLargest 5-minute price increases:")
    for _, row in largest_5min.iterrows():
        print(f"Time: {row['timestamp']}, Change: {row['5min_change']*100:.2f}%")
    
    print(f"\nLargest 15-minute price increases:")
    for _, row in largest_15min.iterrows():
        print(f"Time: {row['timestamp']}, Change: {row['15min_change']*100:.2f}%")
    
    # Plot price movement
    plt.figure(figsize=(15, 7))
    plt.plot(price_data['timestamp'], price_data['value'])
    plt.title(f'Price Movement Over Time for {sample_token}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(minute_data_dir, f'{sample_token}_price_movement.png'))
    plt.close()
    
    # Plot distribution of time gaps
    time_gaps = price_data['timestamp'].diff().dt.total_seconds() / 60
    plt.figure(figsize=(10, 5))
    plt.hist(time_gaps[time_gaps < 10], bins=50)
    plt.title('Distribution of Time Gaps Between Data Points')
    plt.xlabel('Gap (minutes)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(minute_data_dir, f'{sample_token}_time_gaps.png'))
    plt.close()
    
    # Plot the largest price move period
    max_move_time = largest_15min.iloc[0]['timestamp']
    move_window = price_data[
        (price_data['timestamp'] >= max_move_time - pd.Timedelta(hours=1)) &
        (price_data['timestamp'] <= max_move_time + pd.Timedelta(hours=1))
    ]
    
    plt.figure(figsize=(12, 6))
    plt.plot(move_window['timestamp'], move_window['value'])
    plt.title('Price Movement Around Largest 15min Increase')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(minute_data_dir, f'{sample_token}_largest_move.png'))
    plt.close()
    
    return price_data

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = verify_minute_data(base_dir)