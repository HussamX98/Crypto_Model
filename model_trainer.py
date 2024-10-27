# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scripts.utils import get_latest_run_dir

def load_and_prepare_data(base_dir):
    token_data_dir = os.path.join(base_dir, "data", "token_data")
    latest_run_dir = get_latest_run_dir(token_data_dir)
    features_file = os.path.join(latest_run_dir, "extracted_features.csv")
    price_increases_file = os.path.join(base_dir, "data", "5x_price_increases.csv")

    print(f"Loading data from {features_file} and {price_increases_file}")
    
    # Load the extracted features
    features_df = pd.read_csv(features_file)
    print(f"Extracted features shape: {features_df.shape}")
    print("Columns with null values in features_df:")
    print(features_df.isnull().sum())
    
    # Load the 5x price increases data
    price_increases_df = pd.read_csv(price_increases_file)
    print(f"Price increases shape: {price_increases_df.shape}")
    print("Columns with null values in price_increases_df:")
    print(price_increases_df.isnull().sum())
    
    # Merge the dataframes
    merged_df = pd.merge(features_df, price_increases_df, on=['token_address', 'spike_time'], how='inner')
    print(f"Merged dataframe shape: {merged_df.shape}")
    print("Columns with null values in merged_df:")
    print(merged_df.isnull().sum())
    
    if merged_df.empty:
        print("Error: Merged dataframe is empty. Check if 'token_address' and 'spike_time' columns exist and match in both files.")
        return None
    
    # Create the target variable (1 for 5x increase, 0 for no increase)
    merged_df['target'] = 1
    
    # Instead of dropping NA values, let's fill them with a placeholder value
    merged_df = merged_df.fillna(-999)  # You can choose a different placeholder if needed
    print(f"Dataframe shape after filling NA: {merged_df.shape}")
    
    return merged_df

def select_features(df):
    # Select relevant features for training
    feature_columns = [
        'price_mean', 'price_std', 'price_min', 'price_max',
        'volume_mean', 'volume_std', 'price_change_mean', 'price_change_std',
        'rsi', 'macd', 'signal', 'tx_count', 'unique_wallets',
        'liquidity', 'holder_count', 'market_cap'
    ]
    
    # Check if all feature columns exist in the dataframe
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: The following columns are missing from the dataframe: {missing_columns}")
        return None, None
    
    X = df[feature_columns]
    y = df['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

def train_model(X, y):
    if X is None or y is None:
        print("Error: Features or target is None. Cannot train the model.")
        return None, None
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model and scaler
    joblib.dump(model, 'crypto_price_predictor_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    return model, scaler

def analyze_feature_importance(model, X):
    if model is None or X is None:
        print("Error: Model or features is None. Cannot analyze feature importance.")
        return None
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    return feature_importances

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load and prepare the data
    df = load_and_prepare_data(base_dir)
    if df is None:
        return
    
    # Print the first few rows of the dataframe
    print("\nFirst few rows of the prepared dataframe:")
    print(df.head())
    
    # Select features
    X, y = select_features(df)
    if X is None or y is None:
        return
    
    # Train the model
    model, scaler = train_model(X, y)
    if model is None or scaler is None:
        return
    
    # Analyze feature importance
    feature_importances = analyze_feature_importance(model, X)
    if feature_importances is not None:
        print("Top 10 most important features:")
        print(feature_importances.head(10))

if __name__ == "__main__":
    main()
