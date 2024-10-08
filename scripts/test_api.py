# scripts/test_api.py

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import make_api_request
from config import API_KEY  # Make sure this import is present

def test_api():
    print(f"Using API Key: {API_KEY}")  # Add this line to print the API key
    if not API_KEY:
        print("Error: API_KEY is not set. Please add your API key to config.py")
        return

    # Try the /public/token endpoint
    endpoint = "/public/token"
    params = {"address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"}  # USDC token address
    data = make_api_request(endpoint, params)
    print("API request successful. Data received:")
    print(data)

if __name__ == "__main__":
    test_api()
