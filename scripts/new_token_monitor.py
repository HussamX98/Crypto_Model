# new_token_monitor.py
import requests
import pandas as pd
import time
import os
from scripts.utils import get_headers, get_latest_run_dir

def fetch_new_listings(limit=10):
    url = f"https://public-api.birdeye.so/defi/v2/tokens/new_listing?limit={limit}"
    response = requests.get(url, headers=get_headers())
    if response.status_code == 200:
        return response.json()['data']['items']
    else:
        print(f"Failed to fetch new listings: {response.status_code}")
        return None

def monitor_new_tokens(interval=300, base_dir=None, max_iterations=5):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    token_data_dir = os.path.join(base_dir, "data", "token_data")
    latest_run_dir = get_latest_run_dir(token_data_dir)
    output_file = os.path.join(latest_run_dir, "new_tokens.csv")
    
    all_new_tokens = []
    
    for _ in range(max_iterations):
        new_listings = fetch_new_listings()
        if new_listings:
            for token in new_listings:
                if token['address'] not in [t['address'] for t in all_new_tokens]:
                    all_new_tokens.append(token)
                    print(f"New token detected: {token['symbol']} ({token['address']})")
        
        df = pd.DataFrame(all_new_tokens)
        df.to_csv(output_file, index=False)
        print(f"Updated new tokens list saved to {output_file}")
        
        time.sleep(interval)

    print("New token monitoring completed.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monitor_new_tokens(base_dir=base_dir)

    monitor_new_tokens()