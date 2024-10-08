# scripts/utils.py

import requests
from config import API_KEY

API_BASE_URL = "https://public-api.birdeye.so"

def make_api_request(endpoint, params=None):
    url = f"{API_BASE_URL}{endpoint}"
    headers = {
        "X-API-KEY": API_KEY,
        "accept": "application/json"
    }
    params = params or {}

    print(f"Request URL: {url}")
    print(f"Request Headers: {headers}")
    print(f"Request Params: {params}")

    response = requests.get(url, headers=headers, params=params)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Content: {response.text}")

    if response.status_code != 200:
        print(f"HTTP {response.status_code} Error: {response.reason}")
    response.raise_for_status()
    return response.json()

