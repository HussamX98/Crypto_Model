# Feature Analysis Report

## Run Summary

- Analysis Date: 2024-10-25 13:28:03
- Total Spikes Analyzed: 467
- Unique Tokens: 15
- Selected Features: 67

## Feature Selection Results

- Total Features Extracted: 204
- Features Selected: 67
### Selected Features:
- price_change_10m
- volume_momentum_10m
- volume_momentum_5m
- trade_interval_std_10m
- trend_consistency_15m
- trend_consistency_10m
- volume_acceleration_5m
- avg_trade_interval_5m
- price_change_5m
- rsi_14
- rsi_14_slope
- trade_interval_std_3m
- holder_count
- avg_trade_interval_3m
- trend_consistency_3m
- trend_consistency_5m
- volatility_3m
- buy_pressure_volatility_15m
- price_volume_correlation_3m
- liquidity
- volatility_adjusted_momentum_15m
- order_imbalance_3m
- max_trade_size_10m
- rsi_trend_strength_7
- buy_pressure_volatility_10m
- volume_total
- volatility_adjusted_momentum_10m
- trend_strength_15m
- min_trade_size_10m
- min_trade_size_3m
- buy_pressure_volatility_5m
- buy_sell_imbalance_5m
- trend_strength_5m
- avg_trade_size_3m
- volatility_adjusted_momentum_5m
- avg_trade_size_5m
- buy_sell_ratio_5m
- buy_sell_imbalance_3m
- trend_strength_3m
- min_trade_size_5m
- price_std
- max_trade_size_5m
- trend_strength_10m
- trade_size_skew
- trade_size_median
- trades_per_holder
- trend_consistency_2m
- price_min
- trade_size_trend_3m
- trade_size_concentration_5m
- volume_1m
- buy_pressure_volatility_3m
- order_imbalance_1m
- trade_size_concentration_3m
- market_count
- price_volume_correlation_10m
- market_impact_1m
- unique_wallets_24h
- volatility_adjusted_return_10m
- price_volume_correlation_5m
- active_ratio
- market_impact_3m
- buy_sell_imbalance_1m
- rsi_trend_strength_14
- trend_strength_2m
- volume_acceleration_10m
- macd

## Feature Patterns

- Highly Correlated Pairs: 152
- Stable Features: 0
- Volatile Features: 47

## Key Statistics for Selected Features

### price_change_10m
- Mean: 51.8925
- Std: 90.3392
- Min: 0.0536
- Max: 216.5456
- Correlation with target: 0.9812

### volume_momentum_10m
- Mean: 130.6388
- Std: 221.3324
- Min: 2.5459
- Max: 660.9495
- Correlation with target: 0.9759

### volume_momentum_5m
- Mean: 268.3139
- Std: 506.1327
- Min: -0.2073
- Max: 1484.5347
- Correlation with target: 0.9756

### trade_interval_std_10m
- Mean: 102996.8347
- Std: 122639.6952
- Min: 23.8158
- Max: 303956.8081
- Correlation with target: 0.8889

### trend_consistency_15m
- Mean: 0.5563
- Std: 0.2014
- Min: 0.1333
- Max: 0.8667
- Correlation with target: -0.8372

### trend_consistency_10m
- Mean: 0.6039
- Std: 0.2168
- Min: 0.1000
- Max: 0.9000
- Correlation with target: -0.8338

### volume_acceleration_5m
- Mean: -246.1061
- Std: 658.3265
- Min: -1974.9637
- Max: 100.1567
- Correlation with target: -0.8121

### avg_trade_interval_5m
- Mean: 106734.9823
- Std: 123649.5221
- Min: 4.2500
- Max: 334984.5000
- Correlation with target: 0.7758

### price_change_5m
- Mean: 106.5324
- Std: 218.6451
- Min: 0.0200
- Max: 672.1921
- Correlation with target: 0.7236

### rsi_14
- Mean: 84.4570
- Std: 12.3951
- Min: 57.6514
- Max: 100.0000
- Correlation with target: 0.6870

### rsi_14_slope
- Mean: 5.6704
- Std: 1.2964
- Min: 3.8530
- Max: 9.9253
- Correlation with target: 0.6447

### trade_interval_std_3m
- Mean: 107661.9360
- Std: 197761.7926
- Min: 3.5355
- Max: 623492.8185
- Correlation with target: 0.6406

### holder_count
- Mean: 788.1692
- Std: 644.7565
- Min: 19.0000
- Max: 2867.0000
- Correlation with target: -0.6226

### avg_trade_interval_3m
- Mean: 96877.1403
- Std: 154774.7525
- Min: 4.0000
- Max: 488566.0000
- Correlation with target: 0.5700

### trend_consistency_3m
- Mean: 0.8337
- Std: 0.1697
- Min: 0.3333
- Max: 1.0000
- Correlation with target: -0.5480

### trend_consistency_5m
- Mean: 0.6373
- Std: 0.1675
- Min: 0.2000
- Max: 0.8000
- Correlation with target: -0.5302

### volatility_3m
- Mean: 115.3205
- Std: 294.6650
- Min: 0.0001
- Max: 919.4119
- Correlation with target: 0.5150

### buy_pressure_volatility_15m
- Mean: 7.3726
- Std: 8.7721
- Min: 0.0000
- Max: 32.5303
- Correlation with target: -0.4654

### price_volume_correlation_3m
- Mean: -0.1713
- Std: 0.7203
- Min: -0.9993
- Max: 0.9999
- Correlation with target: 0.4490

### liquidity
- Mean: 21720.1628
- Std: 27026.0571
- Min: 0.0000
- Max: 135122.4071
- Correlation with target: -0.4430

### volatility_adjusted_momentum_15m
- Mean: 0.3869
- Std: 0.1533
- Min: 0.0671
- Max: 0.7741
- Correlation with target: -0.4485

### order_imbalance_3m
- Mean: -0.3311
- Std: 0.7571
- Min: -1.0000
- Max: 0.9756
- Correlation with target: 0.4421

### max_trade_size_10m
- Mean: 7862398.5089
- Std: 10119656.9925
- Min: 56128.1076
- Max: 43215953.3830
- Correlation with target: -0.4293

### rsi_trend_strength_7
- Mean: 3.8263
- Std: 5.1507
- Min: 0.0000
- Max: 51.1363
- Correlation with target: -0.3933

### buy_pressure_volatility_10m
- Mean: 9.1201
- Std: 12.8797
- Min: 0.0000
- Max: 59.3180
- Correlation with target: -0.3932

### volume_total
- Mean: 20822105.5449
- Std: 30419402.1252
- Min: 84698.8964
- Max: 157325744.4804
- Correlation with target: -0.3780

### volatility_adjusted_momentum_10m
- Mean: 0.5657
- Std: 0.3576
- Min: 0.0973
- Max: 1.3259
- Correlation with target: -0.3777

### trend_strength_15m
- Mean: 0.3994
- Std: 0.1926
- Min: 0.1243
- Max: 0.9588
- Correlation with target: -0.3642

### min_trade_size_10m
- Mean: 49268.5953
- Std: 77555.4082
- Min: 0.1245
- Max: 289524.9104
- Correlation with target: -0.3542

### min_trade_size_3m
- Mean: 879218.0257
- Std: 1378608.6568
- Min: 21.7428
- Max: 4005033.0000
- Correlation with target: -0.3536

### buy_pressure_volatility_5m
- Mean: 21.3054
- Std: 33.2713
- Min: 0.0000
- Max: 162.4594
- Correlation with target: -0.3521

### buy_sell_imbalance_5m
- Mean: -0.4372
- Std: 0.3657
- Min: -1.0000
- Max: 0.2696
- Correlation with target: -0.3471

### trend_strength_5m
- Mean: 1.3572
- Std: 1.3732
- Min: 0.1240
- Max: 6.6888
- Correlation with target: -0.3421

### avg_trade_size_3m
- Mean: 3362782.0445
- Std: 5504883.9630
- Min: 10657.9315
- Max: 15334305.3548
- Correlation with target: -0.3377

### volatility_adjusted_momentum_5m
- Mean: 1.0577
- Std: 1.0097
- Min: 0.0936
- Max: 4.0732
- Correlation with target: -0.3316

### avg_trade_size_5m
- Mean: 2366665.4780
- Std: 4003853.6073
- Min: 10075.1601
- Max: 21944571.7835
- Correlation with target: -0.3272

### buy_sell_ratio_5m
- Mean: 0.5007
- Std: 0.4530
- Min: 0.0000
- Max: 1.7383
- Correlation with target: -0.3263

### buy_sell_imbalance_3m
- Mean: -0.4602
- Std: 0.4846
- Min: -1.0000
- Max: 0.9934
- Correlation with target: -0.3229

### trend_strength_3m
- Mean: 1.7642
- Std: 2.0181
- Min: 0.1379
- Max: 8.9421
- Correlation with target: -0.3226

### min_trade_size_5m
- Mean: 228636.3262
- Std: 393520.8476
- Min: 0.6883
- Max: 1417215.8539
- Correlation with target: -0.3224

### price_std
- Mean: 0.0120
- Std: 0.0218
- Min: 0.0000
- Max: 0.0686
- Correlation with target: 0.3179

### max_trade_size_5m
- Mean: 5978884.3090
- Std: 10476045.8373
- Min: 13582.2153
- Max: 43215953.3830
- Correlation with target: -0.3150

### trend_strength_10m
- Mean: 0.7220
- Std: 0.6901
- Min: 0.1191
- Max: 2.9828
- Correlation with target: -0.3071

### trade_size_skew
- Mean: 1.6432
- Std: 0.8257
- Min: 0.5631
- Max: 3.0611
- Correlation with target: -0.2797

### trade_size_median
- Mean: 783555.6701
- Std: 1565792.4353
- Min: 30.6705
- Max: 12225865.7835
- Correlation with target: -0.2782

### trades_per_holder
- Mean: 0.0314
- Std: 0.0703
- Min: 0.0000
- Max: 0.5155
- Correlation with target: -0.2459

### trend_consistency_2m
- Mean: 0.9154
- Std: 0.1876
- Min: 0.5000
- Max: 1.0000
- Correlation with target: 0.2448

### price_min
- Mean: 0.0310
- Std: 0.0747
- Min: 0.0000
- Max: 0.2971
- Correlation with target: -0.2319

### trade_size_trend_3m
- Mean: 11.0099
- Std: 18.5438
- Min: -0.9706
- Max: 81.3839
- Correlation with target: -0.2304

### trade_size_concentration_5m
- Mean: 2.9411
- Std: 0.7489
- Min: 1.8394
- Max: 4.2934
- Correlation with target: -0.2265

### volume_1m
- Mean: 3401649.5395
- Std: 8408048.4438
- Min: 239.4228
- Max: 31513033.6975
- Correlation with target: -0.2253

### buy_pressure_volatility_3m
- Mean: 13.5773
- Std: 33.9859
- Min: 0.0000
- Max: 130.1847
- Correlation with target: -0.2204

### order_imbalance_1m
- Mean: -0.7559
- Std: 0.6554
- Min: -1.0000
- Max: 1.0000
- Correlation with target: -0.2073

### trade_size_concentration_3m
- Mean: 2.3544
- Std: 0.5137
- Min: 1.3359
- Max: 2.9634
- Correlation with target: -0.2070

### market_count
- Mean: 1.3105
- Std: 0.8389
- Min: 1.0000
- Max: 5.0000
- Correlation with target: -0.2050

### price_volume_correlation_10m
- Mean: -0.0593
- Std: 0.2516
- Min: -0.8710
- Max: 0.7896
- Correlation with target: -0.2027

### market_impact_1m
- Mean: 86.2903
- Std: 178.9587
- Min: 0.0241
- Max: 718.6030
- Correlation with target: -0.0323

### unique_wallets_24h
- Mean: 26.0921
- Std: 68.7057
- Min: 1.0000
- Max: 426.0000
- Correlation with target: -0.1962

### volatility_adjusted_return_10m
- Mean: 8.5018
- Std: 9.4142
- Min: 0.2789
- Max: 40.0233
- Correlation with target: -0.1942

### price_volume_correlation_5m
- Mean: -0.1438
- Std: 0.4389
- Min: -0.9109
- Max: 0.9545
- Correlation with target: -0.1795

### active_ratio
- Mean: 0.0264
- Std: 0.0265
- Min: 0.0008
- Max: 0.1486
- Correlation with target: 0.1599

### market_impact_3m
- Mean: 293.3114
- Std: 879.2402
- Min: 1.6273
- Max: 6083.6697
- Correlation with target: -0.1486

### buy_sell_imbalance_1m
- Mean: -0.4261
- Std: 0.9056
- Min: -1.0000
- Max: 1.0000
- Correlation with target: 0.1208

### rsi_trend_strength_14
- Mean: 17.3559
- Std: 83.7476
- Min: 0.0000
- Max: 739.1134
- Correlation with target: -0.1104

### trend_strength_2m
- Mean: 3.8570
- Std: 16.1570
- Min: 0.1173
- Max: 257.6799
- Correlation with target: -0.1078

### volume_acceleration_10m
- Mean: -0.4944
- Std: 7.7675
- Min: -38.2150
- Max: 34.3747
- Correlation with target: -0.1004

### macd
- Mean: nan
- Std: nan
- Min: nan
- Max: nan
- Correlation with target: nan
