import json
import numpy as np
from typing import Dict, List
import logging
from collections import defaultdict

class XGBoostRuleExtractor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.output_file = "model_rules.txt"
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('XGBoostRuleExtractor')
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.output_file)
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        return logger
        
    def extract_rules(self):
        """Extract decision rules from each tree"""
        with open(self.model_path, 'r') as f:
            model_data = json.load(f)
            
        learner = model_data['learner']
        gbm = learner.get('gradient_booster', {})
        model = gbm.get('model', {})
        trees = model.get('trees', [])
        
        # Extract rules from each tree
        feature_thresholds = defaultdict(list)
        feature_importance = defaultdict(int)
        
        for i, tree in enumerate(trees):
            split_indices = tree.get('split_indices', [])
            split_conditions = tree.get('split_conditions', [])
            
            # Count feature usage
            for feature_idx in split_indices:
                feature_importance[feature_idx] += 1
            
            # Collect thresholds for each feature
            for feat_idx, threshold in zip(split_indices, split_conditions):
                feature_thresholds[feat_idx].append(threshold)
        
        # Calculate feature ranges
        feature_ranges = {}
        for feat_idx, thresholds in feature_thresholds.items():
            if thresholds:
                feature_ranges[feat_idx] = {
                    'min': min(thresholds),
                    'max': max(thresholds),
                    'mean': sum(thresholds) / len(thresholds),
                    'std': np.std(thresholds),
                    'percentiles': {
                        '25': np.percentile(thresholds, 25),
                        '50': np.percentile(thresholds, 50),
                        '75': np.percentile(thresholds, 75)
                    }
                }
        
        # Generate report
        with open(self.output_file, 'w') as f:
            f.write("=== XGBOOST MODEL DECISION RULES ===\n\n")
            
            f.write("Feature Importance (by usage frequency):\n")
            f.write("-" * 50 + "\n")
            for feat_idx, count in sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True):
                f.write(f"Feature {feat_idx}: Used {count} times\n")
            
            f.write("\nFeature Decision Ranges:\n")
            f.write("-" * 50 + "\n")
            for feat_idx, ranges in feature_ranges.items():
                f.write(f"\nFeature {feat_idx}:\n")
                f.write(f"  Range: {ranges['min']:.4f} to {ranges['max']:.4f}\n")
                f.write(f"  Mean: {ranges['mean']:.4f}\n")
                f.write(f"  Std: {ranges['std']:.4f}\n")
                f.write("  Percentiles:\n")
                f.write(f"    25th: {ranges['percentiles']['25']:.4f}\n")
                f.write(f"    50th: {ranges['percentiles']['50']:.4f}\n")
                f.write(f"    75th: {ranges['percentiles']['75']:.4f}\n")
            
            f.write("\nModel Structure:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Trees: {len(trees)}\n")
            f.write(f"Features Used: {len(feature_importance)}\n")
            f.write(f"Base Score: {learner.get('learner_model_param', {}).get('base_score', 'N/A')}\n")

def main():
    model_path = r"C:\Users\alsal\Projects\Hussam\crypto_model\model_results\20241118_222036\spike_model.json"
    extractor = XGBoostRuleExtractor(model_path)
    extractor.extract_rules()
    print(f"Rules extracted to {extractor.output_file}")

if __name__ == "__main__":
    main()