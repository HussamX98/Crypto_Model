# scripts/data_collector.py

import os
import time  # Don't forget to import time
import pandas as pd
from utils import make_api_request

DATA_DIR = "data/price_data"

def collect_price_data(token_id):
    """Collect historical price data for a given token."""
    endpoint = "/defi/history/price"  # Verify if this is the correct endpoint
    params = {
        "address": token_id,
        "interval": "1m",  # 1-minute intervals
        "startTime": int(time.time()) - 60 * 60 * 24 * 30,  # Last 30 days
        "endTime": int(time.time())
    }
    data = make_api_request(endpoint, params)
    return data

def save_price_data(token_id, data):
    """Save price data to a CSV file."""
    df = pd.DataFrame(data)
    filepath = os.path.join(DATA_DIR, f"{token_id}.csv")
    df.to_csv(filepath, index=False)

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    with open("data/token_list.txt", "r") as f:
        token_list = [line.strip() for line in f.readlines()]

    for token_id in token_list:
        print(f"Collecting data for {token_id}...")
        data = collect_price_data(token_id)
        save_price_data(token_id, data)

if __name__ == "__main__":
    main()
