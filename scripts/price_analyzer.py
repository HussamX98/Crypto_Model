# price_analyzer.py
import pandas as pd
import os
from datetime import timedelta
from scripts.utils import load_token_list, get_headers, get_latest_run_dir

def identify_5x_increases(data_dir, output_file):
    latest_run_dir = get_latest_run_dir(data_dir)
    results = []

    for filename in os.listdir(latest_run_dir):
        if filename.endswith("_trade_data.csv"):
            token_address = filename.split("_trade_data")[0]
            df = pd.read_csv(os.path.join(latest_run_dir, filename))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            for i in range(len(df) - 96):  # Check every 24-hour window (96 15-minute intervals)
                start_price = df.iloc[i]['o']
                end_price = df.iloc[i+96]['c']
                
                if end_price >= 5 * start_price:
                    spike_time = df.iloc[i+96]['timestamp']
                    results.append({
                        'token_address': token_address,
                        'spike_time': spike_time,
                        'start_price': start_price,
                        'end_price': end_price,
                        'volume': df.iloc[i:i+97]['v'].sum()
                    })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"5x price increase instances saved to {output_file}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "token_data")
    output_file = os.path.join(base_dir, "data", "5x_price_increases.csv")
    identify_5x_increases(data_dir, output_file)
