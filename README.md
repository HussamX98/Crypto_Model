# Crypto Trading Signal Model (Memecoin Pump Detector)

Built with Claude's Sonnet 4.5 (ancient history) + XGBoost — identifies patterns in memecoins that historically led to 10x–100x moves.

## Overview
This project trains a machine learning model to detect early signals of sharp upward price movements in memecoins.  
I used historical on-chain + price data, engineered features based on what I observed in past pumps, trained an **XGBoost classifier**, extracted interpretable rules, and backtested the strategy.

**Key outcome**: The model learned real, repeatable patterns that most sharp pumps have in common (volume spikes, holder growth, liquidity changes, etc.).

## Tech Stack
- **Language**: Python 3
- **Model**: XGBoost (gradient boosting)
- **Data**: pandas, numpy
- **Analysis**: scikit-learn, SHAP (for rule extraction)
- **Visualization**: matplotlib / seaborn (charts in backtest_results & model_results)
- **Others**: joblib (model saving), datetime, etc.

## Project Structure

Crypto_Model/
├── data/                    ← raw + processed memecoin datasets
├── scripts/                 ← training, feature engineering, backtesting scripts
├── backtest_results/        ← equity curves, trade logs, performance metrics
├── model_results/           ← trained model, feature importance, confusion matrix
├── output/                  ← generated reports
├── model_rules.txt          ← human-readable IF-THEN rules the model learned
├── xgboost_model_analysis.json ← full model metrics + feature importances
├── model_analysis_output.txt ← detailed performance report
└── model_analysis_report.txt

## What the Model Looks For (Example Rules)
(See `model_rules.txt` for the full extracted rule set)

Typical high-confidence conditions the model flags:
- Extreme relative volume spike + low market cap
- Rapid increase in holder count within 5 minutes.
- Price breaking recent high with increasing buy pressure
- Buy/Sell imbalance. 

## Results Highlights
- Backtested on historical memecoin launches
- Strong precision on 10x+ moves (exact numbers in `backtest_results/` and reports)
- Feature importance chart
- Still needs more backtesting and refining execution.


## How It Was Built
1. Collected memecoin data (price, volume, holder count, liquidity, buy/sell imbalances, and other on-chain metrics)
2. Labeled coins that showed over 10x increase in price within 5 minutes.
3. Identified 10+ correlations that are common in 75% of these 10x plus surge occurrences. 
4. Trained XGBoost + hyperparameter tuning
5. Used SHAP / rule extraction to make the black-box model interpretable
6. Backtested with realistic slippage & fees

