# Feature Analysis Report

    ## Overview
    - Total Spikes Analyzed: 5651
    - Number of Features: 101
    - Spike Range: 5.00x to 14793.00x

    ## Key Indicators
    Most Predictive Features (correlation with price increase):
    future_price_increase    1.000000
price_change_15m         0.769460
momentum_14m             0.769340
price_change_10m         0.769249
momentum_5m              0.609310
trend_acceleration       0.608907
momentum_7m              0.499814
range_10m                0.301414
range_15m                0.291235
volatility_15m           0.216230

    ## Feature Patterns by Spike Magnitude
                        rsi_14  volatility_15m  momentum_15m  trend_strength
spike_magnitude                                                         
Very Low         75.639772        0.249798           0.0        0.044782
Low              75.640933        0.264903           0.0        0.045792
Medium           73.543978        0.291265           0.0        0.049892
High             74.190686        0.293278           0.0        0.050645
Very High        74.454314        0.370208           0.0        0.066334

    ## Statistical Summary
    - Median Spike Size: 7.46x
    - Average RSI before spike: 74.69
    - Average Volatility: 0.2939
    