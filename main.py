# main.py
import os
import time
import threading
import logging
from scripts.data_collector import collect_data
from scripts.price_analyzer import identify_5x_increases
from scripts.feature_extractor import main as extract_features
from scripts.new_token_monitor import monitor_new_tokens

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "token_data")
    spike_file = os.path.join(base_dir, "data", "5x_price_increases.csv")
    features_file = os.path.join(base_dir, "data", "extracted_features.csv")
    new_tokens_file = os.path.join(base_dir, "data", "new_tokens.csv")

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)

    # Start new token monitoring in a separate thread
    new_token_thread = threading.Thread(target=monitor_new_tokens, args=(300, new_tokens_file))
    new_token_thread.start()

    try:
        while True:
            logging.info("Starting data collection cycle")
            
            try:
                # Collect token data
                logging.info("Collecting token data...")
                collect_data(data_dir, spike_file)
                logging.info("Token data collection completed")
                
                # Identify 5x price increases
                logging.info("Identifying 5x price increases...")
                identify_5x_increases(data_dir, spike_file)
                logging.info("5x price increase identification completed")
                
                # Extract features
                logging.info("Extracting features...")
                extract_features()
                logging.info("Feature extraction completed")
                
                logging.info("Data processing cycle completed. Waiting for next cycle...")
            except Exception as e:
                logging.error(f"An error occurred during the data processing cycle: {str(e)}")
            
            time.sleep(3600)  # Wait for 1 hour before next cycle
    except KeyboardInterrupt:
        logging.info("Script interrupted by user. Exiting...")
    finally:
        logging.info("Script execution completed.")

if __name__ == "__main__":
    main()
