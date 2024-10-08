# scripts/feature_extractor.py

import os
import pandas as pd
from utils import make_api_request

PRICE_DATA_DIR = "data/price_data"
FEATURES_DIR = "data/features"
DATASET_FILE = "data/dataset.csv"

def extract_features(token_id):
    """Extract features before each price spike."""
    spike_file = os.path.join(FEATURES_DIR, f"{token_id}_spikes.csv")
    if not os.path.exists(spike_file):
        return None

    spikes_df = pd.read_csv(spike_file)
    price_file = os.path.join(PRICE_DATA_DIR, f"{token_id}.csv")
    price_df = pd.read_csv(price_file)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='s')
    price_df.set_index('timestamp', inplace=True)

    features_list = []
    for _, row in spikes_df.iterrows():
        spike_time = pd.to_datetime(row['start_time'])
        window_start = spike_time - pd.Timedelta(minutes=30)
        window_end = spike_time - pd.Timedelta(minutes=15)
        window_data = price_df.loc[window_start:window_end]

        if window_data.empty:
            continue

        # Extract features (example: average price, volume, etc.)
        avg_price = window_data['price'].mean()
        total_volume = window_data['volume'].sum()
        # Add more features as needed

        features = {
            'token_id': token_id,
            'spike_time': spike_time,
            'avg_price': avg_price,
            'total_volume': total_volume,
            # Add more features here
        }
        features_list.append(features)

    return features_list

def main():
    all_features = []
    token_files = os.listdir(PRICE_DATA_DIR)
    for file in token_files:
        token_id = file.replace('.csv', '')
        print(f"Extracting features for {token_id}...")
        features = extract_features(token_id)
        if features:
            all_features.extend(features)

    if all_features:
        dataset_df = pd.DataFrame(all_features)
        dataset_df.to_csv(DATASET_FILE, index=False)
        print(f"Dataset saved to {DATASET_FILE}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()
