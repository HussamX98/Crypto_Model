# scripts/price_analyzer.py

import os
import pandas as pd

PRICE_DATA_DIR = "data/price_data"
FEATURES_DIR = "data/features"

def find_price_spikes(token_id):
    """Find instances where the price increased by 5x within 24 hours."""
    filepath = os.path.join(PRICE_DATA_DIR, f"{token_id}.csv")
    df = pd.read_csv(filepath)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    spikes = []
    for start_time in df.index:
        end_time = start_time + pd.Timedelta(hours=24)
        if end_time > df.index[-1]:
            continue
        start_price = df.at[start_time, 'price']
        end_price = df.at[end_time, 'price']
        if end_price >= 5 * start_price:
            spikes.append((start_time, end_time))
    return spikes

def save_spikes(token_id, spikes):
    """Save spike times to a CSV file."""
    df = pd.DataFrame(spikes, columns=['start_time', 'end_time'])
    filepath = os.path.join(FEATURES_DIR, f"{token_id}_spikes.csv")
    df.to_csv(filepath, index=False)

def main():
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)

    token_files = os.listdir(PRICE_DATA_DIR)
    for file in token_files:
        token_id = file.replace('.csv', '')
        print(f"Analyzing price data for {token_id}...")
        spikes = find_price_spikes(token_id)
        if spikes:
            save_spikes(token_id, spikes)
        else:
            print(f"No spikes found for {token_id}.")

if __name__ == "__main__":
    main()
