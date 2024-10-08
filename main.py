# main.py

import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name])

def main():
    run_script("scripts/data_collector.py")
    run_script("scripts/price_analyzer.py")
    run_script("scripts/feature_extractor.py")
    print("Data collection and preprocessing complete.")

if __name__ == "__main__":
    main()
