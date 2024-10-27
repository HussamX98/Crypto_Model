import yaml
import os
from config import API_KEY
import logging

BASE_URL = "https://public-api.birdeye.so"

def load_token_list(file_name='token_list.yaml'):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(current_dir)
    # Construct the full path to token_list.yaml
    file_path = os.path.join(project_root, file_name)
    
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return [{'address': token.split('#')[0].strip()} for token in data['tokens']]
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file exists in the project root directory.")
        return []

def get_headers():
    headers = {
        "X-API-KEY": API_KEY,
        "Accept": "application/json",
    }
    masked_headers = headers.copy()
    masked_headers["X-API-KEY"] = "****" + API_KEY[-4:]  # Only show last 4 characters
    logging.debug(f"Request Headers: {masked_headers}")
    return headers

def get_url(endpoint):
    url = f"{BASE_URL}/v1{endpoint}"
    logging.debug(f"Constructed URL: {url}")
    return url

def get_latest_run_dir(base_dir):
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    if not run_dirs:
        raise ValueError("No run directories found in the base directory.")
    latest_run = max(run_dirs)
    return os.path.join(base_dir, latest_run)
