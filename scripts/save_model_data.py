import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def save_training_data():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'model_results', '20241118_222036')
    features_dir = os.path.join(base_dir, 'data', 'minute_level_data', 'features')
    
    # Load original feature file
    feature_files = sorted([f for f in os.listdir(features_dir) if f.startswith('spike_features_')])
    df = pd.read_csv(os.path.join(features_dir, feature_files[-1]))
    
    # Use the top 6 features from correlations
    features = [
        'momentum_7m',
        'momentum_5m',
        'trend_acceleration',
        'momentum_acc_7m',
        'momentum_acc_14m',
        'price_change_15m'
    ]
    
    print("Using these features:")
    print(features)
    
    # Prepare and scale features
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load the model
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, 'spike_model.json'))
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Save everything
    # 1. Feature data
    feature_data = pd.DataFrame(X_scaled, columns=features)
    feature_data.to_csv(os.path.join(model_dir, 'training_features.csv'), index=False)
    
    # 2. Predictions
    predictions_df = pd.DataFrame({
        'timestamp': df['spike_time'],
        'token_address': df['token_address'],
        'confidence_score': y_pred_proba,
        'initial_price': df['initial_price'] if 'initial_price' in df.columns else None,
        'future_price_increase': df['future_price_increase']
    })
    predictions_df.to_csv(os.path.join(model_dir, 'model_predictions.csv'), index=False)
    
    print(f"\nSaved training features and predictions to {model_dir}")
    print(f"Number of samples: {len(predictions_df)}")
    print("\nFeature statistics:")
    print(feature_data.describe())
    
    print("\nPrediction statistics:")
    print(predictions_df['confidence_score'].describe())

if __name__ == "__main__":
    save_training_data()