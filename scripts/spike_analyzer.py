import os
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List

class SpikeAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.spikes_dir = os.path.join(run_dir, 'spikes')
        self.price_data_dir = os.path.join(run_dir, 'price_data')
        self.trades_dir = os.path.join(run_dir, 'trades')
        print(f"Analyzing data in: {run_dir}")
        print(f"Spikes directory: {self.spikes_dir}")
        
    def load_spike_files(self) -> List[Dict]:
        """Load all spike files and return as list"""
        spikes = []
        if not os.path.exists(self.spikes_dir):
            print(f"Spikes directory not found: {self.spikes_dir}")
            return spikes
            
        for filename in os.listdir(self.spikes_dir):
            if '_spike_' in filename and filename.endswith('.json'):
                full_path = os.path.join(self.spikes_dir, filename)
                print(f"Loading spike file: {filename}")
                try:
                    with open(full_path, 'r') as f:
                        spike = json.load(f)
                        spikes.append(spike)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        # If no spikes found in spikes directory, try loading from CSV
        if not spikes:
            csv_path = os.path.join(self.run_dir, '5x_price_increases.csv')
            if os.path.exists(csv_path):
                print(f"Loading spikes from CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                spikes = df.to_dict('records')
        
        return spikes

def analyze_spike(self, spike: Dict):
    """Analyze a single spike in detail"""
    token_address = spike['token_address']
    spike_time = datetime.fromisoformat(spike['spike_time'])
    start_time = datetime.fromisoformat(spike['start_time'])
    
    # Load price data
    price_file = os.path.join(self.price_data_dir, f'{token_address}_price_history.csv')
    if not os.path.exists(price_file):
        print(f"No price data found for {token_address} at {price_file}")
        matching_files = [f for f in os.listdir(self.price_data_dir) if token_address in f]
        if matching_files:
            print(f"Found alternative price files: {matching_files}")
            price_file = os.path.join(self.price_data_dir, matching_files[0])
        else:
            return
            
    prices_df = pd.read_csv(price_file)
    prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])
    
    # Get price data around spike
    window_start = start_time - timedelta(minutes=30)
    window_end = spike_time + timedelta(minutes=30)
    window_data = prices_df[
        (prices_df['timestamp'] >= window_start) & 
        (prices_df['timestamp'] <= window_end)
    ]
    
    # Print analysis
    print(f"\nAnalyzing spike for {token_address}")
    print(f"Spike time: {spike_time}")
    print(f"Reported increase: {spike['price_increase']:.2f}x")
    print(f"Start price: {spike['start_price']:.10f}")
    print(f"End price: {spike['end_price']:.10f}")
    
    # Verify price movement
    if not window_data.empty:
        actual_start_price = window_data.iloc[0]['value']
        actual_max_price = window_data['value'].max()
        actual_increase = actual_max_price / actual_start_price
        print(f"\nPrice verification:")
        print(f"Actual start price: {actual_start_price:.10f}")
        print(f"Actual max price: {actual_max_price:.10f}")
        print(f"Actual increase: {actual_increase:.2f}x")
        
        # Plot price movement
        plt.figure(figsize=(12, 6))
        plt.plot(window_data['timestamp'], window_data['value'], label='Price')
        
        # Add vertical lines for spike window
        plt.axvline(x=start_time, color='g', linestyle='--', label='Start Time')
        plt.axvline(x=spike_time, color='r', linestyle='--', label='Spike Time')
        
        plt.title(f'Price Movement for {token_address}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
def main():
    # Get the most recent run directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(base_dir, 'data', 'runs')
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
    if not run_dirs:
        print("No run directories found")
        return
        
    latest_run = sorted(run_dirs)[-1]  # Get the most recent run
    run_dir = os.path.join(runs_dir, latest_run)
    print(f"Analyzing most recent run: {latest_run}")
    
    analyzer = SpikeAnalyzer(run_dir)
    spikes = analyzer.load_spike_files()
    
    print(f"Found {len(spikes)} spikes to analyze")
    for i, spike in enumerate(spikes[:5], 1):  # Analyze first 5 spikes
        print(f"\n{'='*50}")
        print(f"Analyzing spike {i} of 5")
        analyzer.analyze_spike(spike)
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()