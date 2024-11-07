import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple
from glob import glob

class ModelAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        
    def analyze_results(self):
        """Comprehensive analysis of model training results"""
        # Load all results
        cv_results = self._load_cv_results()
        feature_importance = self._load_feature_importance()
        predictions = self._load_predictions()
        
        # 1. Overall Performance Metrics
        print("\n=== Overall Model Performance ===")
        metrics_avg = pd.DataFrame(cv_results).mean()
        metrics_std = pd.DataFrame(cv_results).std()
        
        for metric in ['rmse', 'mae', 'r2', 'spearman_corr']:
            if metric in metrics_avg:
                print(f"{metric.upper()}: {metrics_avg[metric]:.4f} ± {metrics_std[metric]:.4f}")
        
        # 2. Spike Detection Performance
        print("\n=== Spike Detection Performance ===")
        for threshold in [5, 10, 20, 50, 100]:
            print(f"\nThreshold {threshold}x:")
            print(f"Accuracy: {metrics_avg[f'accuracy_{threshold}x']:.4f} ± {metrics_std[f'accuracy_{threshold}x']:.4f}")
            print(f"Precision: {metrics_avg[f'precision_{threshold}x']:.4f} ± {metrics_std[f'precision_{threshold}x']:.4f}")
            print(f"Recall: {metrics_avg[f'recall_{threshold}x']:.4f} ± {metrics_std[f'recall_{threshold}x']:.4f}")
            print(f"F1: {metrics_avg[f'f1_{threshold}x']:.4f} ± {metrics_std[f'f1_{threshold}x']:.4f}")
        
        # 3. Feature Importance Analysis
        print("\n=== Top 10 Most Important Features ===")
        top_features = feature_importance.head(10)
        for _, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # 4. Prediction Analysis
        all_predictions = pd.concat(predictions)
        correlation = np.corrcoef(all_predictions['true_value'], all_predictions['predicted_value'])[0,1]
        print(f"\n=== Prediction Analysis ===")
        print(f"Overall correlation: {correlation:.4f}")
        
        # 5. Generate Visualizations
        self._create_visualizations(all_predictions, feature_importance)
        
    def _load_cv_results(self) -> List[Dict]:
        """Load cross-validation results"""
        results_path = os.path.join(self.run_dir, 'model_results', 'cv_results.json')
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def _load_feature_importance(self) -> pd.DataFrame:
        """Load feature importance results"""
        importance_path = os.path.join(self.run_dir, 'feature_importance', 'feature_importance.csv')
        return pd.read_csv(importance_path)
    
    def _load_predictions(self) -> List[pd.DataFrame]:
        """Load predictions from all folds"""
        predictions = []
        for pred_file in glob(os.path.join(self.run_dir, 'predictions', 'fold_*_predictions.csv')):
            predictions.append(pd.read_csv(pred_file))
        return predictions
    
    def _create_visualizations(self, predictions: pd.DataFrame, feature_importance: pd.DataFrame):
        """Create and save analysis visualizations"""
        plot_dir = os.path.join(self.run_dir, 'analysis_plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions['true_value'], predictions['predicted_value'], alpha=0.5)
        plt.plot([0, predictions['true_value'].max()], [0, predictions['true_value'].max()], 'r--')
        plt.xlabel('Actual Spike Magnitude')
        plt.ylabel('Predicted Spike Magnitude')
        plt.title('Actual vs Predicted Spike Magnitudes')
        plt.savefig(os.path.join(plot_dir, 'actual_vs_predicted.png'))
        plt.close()
        
        # 2. Feature Importance Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
        plt.close()
        
        # 3. Prediction Error Distribution
        plt.figure(figsize=(10, 6))
        errors = predictions['predicted_value'] - predictions['true_value']
        sns.histplot(errors, bins=50)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.savefig(os.path.join(plot_dir, 'error_distribution.png'))
        plt.close()
        
        # 4. Performance by Magnitude Range
        magnitude_ranges = [(5, 10), (10, 20), (20, 50), (50, 100), (100, float('inf'))]
        accuracies = []
        for low, high in magnitude_ranges:
            mask = (predictions['true_value'] >= low) & (predictions['true_value'] < high)
            if mask.any():
                accuracy = np.mean(
                    (predictions.loc[mask, 'predicted_value'] >= low) & 
                    (predictions.loc[mask, 'predicted_value'] < high)
                )
                accuracies.append({'range': f'{low}-{high}', 'accuracy': accuracy})
        
        acc_df = pd.DataFrame(accuracies)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=acc_df, x='range', y='accuracy')
        plt.title('Prediction Accuracy by Magnitude Range')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'accuracy_by_magnitude.png'))
        plt.close()

def analyze_latest_run():
    """Analyze the latest model run results"""
    # Find the latest run directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(base_dir, "data", "runs")
    run_dirs = [d for d in os.listdir(runs_dir) if d.startswith('run_')]
    latest_run = sorted(run_dirs)[-1]
    run_dir = os.path.join(runs_dir, latest_run)
    
    # Create analyzer and run analysis
    analyzer = ModelAnalyzer(run_dir)
    analyzer.analyze_results()

if __name__ == "__main__":
    analyze_latest_run()