# Feature Analysis Report

## Run Summary

- Analysis Date: 2024-10-27 12:30:44
- Total Spikes Analyzed: 467
- Unique Tokens: 15
- Selected Features: 130

## Feature Selection Results

- Total Features Extracted: 147
- Features Selected: 130
### Selected Features:
- price_mean
- price_std
- price_min
- price_max
- price_last
- rsi_7
- rsi_7_slope
- rsi_14
- rsi_14_slope
- macd
- macd_signal
- macd_divergence
- price_change_3m
- volatility_3m
- momentum_3m
- momentum_std_3m
- acceleration_3m
- trend_strength_3m
- trend_consistency_3m
- price_change_5m
- volatility_5m
- momentum_5m
- momentum_std_5m
- acceleration_5m
- trend_strength_5m
- trend_consistency_5m
- price_change_10m
- volatility_10m
- momentum_10m
- momentum_std_10m
- acceleration_10m
- trend_strength_10m
- trend_consistency_10m
- price_change_15m
- volatility_15m
- momentum_15m
- momentum_std_15m
- acceleration_15m
- trend_strength_15m
- trend_consistency_15m
- volume_total
- volume_mean
- volume_std
- volume_skew
- volume_kurtosis
- avg_trade_interval
- trade_interval_std
- whale_trade_count
- whale_volume_ratio
- volume_1m
- trades_count_1m
- avg_trade_size_1m
- max_trade_size_1m
- buy_sell_ratio_1m
- buy_sell_imbalance_1m
- large_trade_ratio_1m
- volume_3m
- trades_count_3m
- avg_trade_size_3m
- max_trade_size_3m
- buy_sell_ratio_3m
- buy_sell_imbalance_3m
- large_trade_ratio_3m
- volume_momentum_3m
- volume_momentum_std_3m
- volume_acceleration_3m
- volume_5m
- trades_count_5m
- avg_trade_size_5m
- max_trade_size_5m
- buy_sell_ratio_5m
- buy_sell_imbalance_5m
- large_trade_ratio_5m
- volume_momentum_5m
- volume_momentum_std_5m
- volume_acceleration_5m
- volume_10m
- trades_count_10m
- avg_trade_size_10m
- max_trade_size_10m
- buy_sell_ratio_10m
- buy_sell_imbalance_10m
- large_trade_ratio_10m
- volume_momentum_10m
- volume_momentum_std_10m
- volume_acceleration_10m
- volume_15m
- trades_count_15m
- avg_trade_size_15m
- max_trade_size_15m
- buy_sell_ratio_15m
- buy_sell_imbalance_15m
- large_trade_ratio_15m
- volume_momentum_15m
- volume_momentum_std_15m
- volume_acceleration_15m
- volume_trend_slope
- volume_trend_r2
- volume_trend_strength
- volume_acceleration
- trade_interval_skew
- cluster_count
- clustering_ratio
- liquidity
- market_count
- holder_count
- unique_wallets_24h
- active_ratio
- average_holding
- trades_24h
- trades_per_holder
- volume_per_holder
- trend_strength_2m
- trend_consistency_2m
- momentum_strength_2m
- momentum_strength_3m
- volatility_acceleration_1m
- volatility_acceleration_3m
- volatility_acceleration_5m
- volatility_acceleration_10m
- volatility_acceleration_15m
- volatility_adjusted_momentum_3m
- volatility_adjusted_momentum_5m
- volatility_adjusted_momentum_10m
- volatility_adjusted_momentum_15m
- rsi_trend_strength_7
- rsi_trend_strength_14
- rsi_trend_strength_21
- data_points_count
- trades_count

## Feature Patterns

- Highly Correlated Pairs: 716
- Stable Features: 8
- Volatile Features: 96

## Key Statistics for Selected Features

### price_mean
- mean: 0.0380
- std: 0.0858
- min: 0.0000
- max: 0.3482
- correlation_with_target: -0.2058

### price_std
- mean: 0.0120
- std: 0.0218
- min: 0.0000
- max: 0.0686
- correlation_with_target: 0.3175

### price_min
- mean: 0.0310
- std: 0.0747
- min: 0.0000
- max: 0.2971
- correlation_with_target: -0.2324

### price_max
- mean: 0.0757
- std: 0.1248
- min: 0.0000
- max: 0.4415
- correlation_with_target: 0.0969

### price_last
- mean: 0.0757
- std: 0.1248
- min: 0.0000
- max: 0.4415
- correlation_with_target: 0.0969

### rsi_7
- mean: 87.4503
- std: 11.4018
- min: 61.2365
- max: 100.0000
- correlation_with_target: 0.6070

### rsi_7_slope
- mean: 3.6432
- std: 3.8082
- min: -4.0276
- max: 14.7305
- correlation_with_target: 0.1157

### rsi_14
- mean: 84.4570
- std: 12.3951
- min: 57.6514
- max: 100.0000
- correlation_with_target: 0.6882

### rsi_14_slope
- mean: 7.1521
- std: 12.1204
- min: -2.1490
- max: 49.1955
- correlation_with_target: 0.3720

### macd
- mean: nan
- std: nan
- min: nan
- max: nan
- correlation_with_target: nan

### macd_signal
- mean: nan
- std: nan
- min: nan
- max: nan
- correlation_with_target: nan

### macd_divergence
- mean: nan
- std: nan
- min: nan
- max: nan
- correlation_with_target: nan

### price_change_3m
- mean: 219.7756
- std: 522.4224
- min: 0.0215
- max: 1634.9240
- correlation_with_target: 0.5772

### volatility_3m
- mean: 115.3205
- std: 294.6650
- min: 0.0001
- max: 919.4119
- correlation_with_target: 0.5148

### momentum_3m
- mean: 81.8149
- std: 208.3882
- min: 0.0107
- max: 650.3793
- correlation_with_target: 0.5170

### momentum_std_3m
- mean: 115.3205
- std: 294.6650
- min: 0.0001
- max: 919.4119
- correlation_with_target: 0.5148

### acceleration_3m
- mean: 143.6565
- std: 423.8256
- min: -78.0031
- max: 1300.2447
- correlation_with_target: 0.4073

### trend_strength_3m
- mean: 1.7642
- std: 2.0181
- min: 0.1379
- max: 8.9421
- correlation_with_target: -0.3231

### trend_consistency_3m
- mean: 0.8337
- std: 0.1697
- min: 0.3333
- max: 1.0000
- correlation_with_target: -0.5476

### price_change_5m
- mean: 106.5324
- std: 218.6451
- min: 0.0200
- max: 672.1921
- correlation_with_target: 0.7234

### volatility_5m
- mean: 81.6683
- std: 208.3830
- min: 0.0105
- max: 650.2819
- correlation_with_target: 0.5155

### momentum_5m
- mean: 40.9234
- std: 104.1470
- min: 0.0128
- max: 325.0790
- correlation_with_target: 0.5170

### momentum_std_5m
- mean: 81.6683
- std: 208.3830
- min: 0.0105
- max: 650.2819
- correlation_with_target: 0.5155

### acceleration_5m
- mean: 51.1937
- std: 139.8119
- min: -0.2629
- max: 433.4272
- correlation_with_target: 0.4626

### trend_strength_5m
- mean: 1.3572
- std: 1.3732
- min: 0.1240
- max: 6.6888
- correlation_with_target: -0.3424

### trend_consistency_5m
- mean: 0.7966
- std: 0.2094
- min: 0.2500
- max: 1.0000
- correlation_with_target: -0.5298

### price_change_10m
- mean: 51.8925
- std: 90.3392
- min: 0.0536
- max: 216.5456
- correlation_with_target: 0.9812

### volatility_10m
- mean: 54.4885
- std: 138.9156
- min: 0.0221
- max: 433.5422
- correlation_with_target: 0.5157

### momentum_10m
- mean: 18.1963
- std: 46.2512
- min: 0.0100
- max: 144.3896
- correlation_with_target: 0.5169

### momentum_std_10m
- mean: 54.4885
- std: 138.9156
- min: 0.0221
- max: 433.5422
- correlation_with_target: 0.5157

### acceleration_10m
- mean: 19.2072
- std: 52.4507
- min: -0.0277
- max: 162.6028
- correlation_with_target: 0.4626

### trend_strength_10m
- mean: 0.7220
- std: 0.6901
- min: 0.1191
- max: 2.9828
- correlation_with_target: -0.3072

### trend_consistency_10m
- mean: 0.6709
- std: 0.2408
- min: 0.1111
- max: 1.0000
- correlation_with_target: -0.8337

### price_change_15m
- mean: 45.2050
- std: 80.1010
- min: 0.1222
- max: 217.9383
- correlation_with_target: 1.0000

### volatility_15m
- mean: 43.8044
- std: 111.4702
- min: 0.0400
- max: 347.5991
- correlation_with_target: 0.5157

### momentum_15m
- mean: 11.7257
- std: 29.7527
- min: 0.0118
- max: 92.8053
- correlation_with_target: 0.5168

### momentum_std_15m
- mean: 43.8044
- std: 111.4702
- min: 0.0400
- max: 347.5991
- correlation_with_target: 0.5157

### acceleration_15m
- mean: 11.8425
- std: 32.2994
- min: -0.0328
- max: 100.0389
- correlation_with_target: 0.4626

### trend_strength_15m
- mean: 0.3983
- std: 0.1915
- min: 0.1243
- max: 0.9588
- correlation_with_target: -0.3641

### trend_consistency_15m
- mean: 0.5955
- std: 0.2157
- min: 0.1429
- max: 0.9286
- correlation_with_target: -0.8374

### volume_total
- mean: 20822105.5449
- std: 30419402.1252
- min: 84698.8964
- max: 157325744.4804
- correlation_with_target: -0.3788

### volume_mean
- mean: 2082210.5545
- std: 3041940.2125
- min: 8469.8896
- max: 15732574.4480
- correlation_with_target: -0.3788

### volume_std
- mean: 2832539.1468
- std: 3762931.7174
- min: 18939.6287
- max: 16987609.1093
- correlation_with_target: -0.4162

### volume_skew
- mean: 1.6432
- std: 0.8257
- min: 0.5631
- max: 3.0611
- correlation_with_target: -0.2793

### volume_kurtosis
- mean: 2.4593
- std: 3.7482
- min: -2.0109
- max: 9.5085
- correlation_with_target: -0.2732

### avg_trade_interval
- mean: -64920.0645
- std: 71589.6537
- min: -183783.6667
- max: -10.2222
- correlation_with_target: -0.8645

### trade_interval_std
- mean: 102996.8347
- std: 122639.6952
- min: 23.8158
- max: 303956.8081
- correlation_with_target: 0.8889

### whale_trade_count
- mean: 0.9893
- std: 0.1030
- min: 0.0000
- max: 1.0000
- correlation_with_target: 0.0587

### whale_volume_ratio
- mean: 0.4293
- std: 0.1617
- min: 0.0000
- max: 0.7789
- correlation_with_target: -0.0194

### volume_1m
- mean: 3401649.5395
- std: 8408048.4438
- min: 239.4228
- max: 31513033.6975
- correlation_with_target: -0.2258

### trades_count_1m
- mean: 1.0000
- std: 0.0000
- min: 1.0000
- max: 1.0000
- correlation_with_target: nan

### avg_trade_size_1m
- mean: 3401649.5395
- std: 8408048.4438
- min: 239.4228
- max: 31513033.6975
- correlation_with_target: -0.2258

### max_trade_size_1m
- mean: 3401649.5395
- std: 8408048.4438
- min: 239.4228
- max: 31513033.6975
- correlation_with_target: -0.2258

### buy_sell_ratio_1m
- mean: inf
- std: nan
- min: 0.0000
- max: inf
- correlation_with_target: nan

### buy_sell_imbalance_1m
- mean: -0.4261
- std: 0.9056
- min: -1.0000
- max: 1.0000
- correlation_with_target: 0.1202

### large_trade_ratio_1m
- mean: 1.0000
- std: 0.0000
- min: 1.0000
- max: 1.0000
- correlation_with_target: nan

### volume_3m
- mean: 10088346.1334
- std: 16514651.8889
- min: 31973.7944
- max: 46002916.0643
- correlation_with_target: -0.3386

### trades_count_3m
- mean: 3.0000
- std: 0.0000
- min: 3.0000
- max: 3.0000
- correlation_with_target: nan

### avg_trade_size_3m
- mean: 3362782.0445
- std: 5504883.9630
- min: 10657.9315
- max: 15334305.3548
- correlation_with_target: -0.3386

### max_trade_size_3m
- mean: 5897131.8091
- std: 10514695.8243
- min: 13577.0018
- max: 43215953.3830
- correlation_with_target: -0.3104

### buy_sell_ratio_3m
- mean: 7.6203
- std: 45.9140
- min: 0.0000
- max: 302.8975
- correlation_with_target: -0.0907

### buy_sell_imbalance_3m
- mean: -0.4602
- std: 0.4846
- min: -1.0000
- max: 0.9934
- correlation_with_target: -0.3231

### large_trade_ratio_3m
- mean: 0.3390
- std: 0.0433
- min: 0.3333
- max: 0.6667
- correlation_with_target: -0.0716

### volume_momentum_3m
- mean: 157.3338
- std: 414.9278
- min: -0.6548
- max: 1289.9810
- correlation_with_target: 0.4551

### volume_momentum_std_3m
- mean: 223.0717
- std: 586.8417
- min: 0.0837
- max: 1825.0122
- correlation_with_target: 0.4556

### volume_acceleration_3m
- mean: -292.9796
- std: 838.1412
- min: -2580.9570
- max: 300.8975
- correlation_with_target: -0.4653

### volume_5m
- mean: 11833327.3901
- std: 20019268.0367
- min: 50375.8005
- max: 109722858.9176
- correlation_with_target: -0.3280

### trades_count_5m
- mean: 5.0000
- std: 0.0000
- min: 5.0000
- max: 5.0000
- correlation_with_target: nan

### avg_trade_size_5m
- mean: 2366665.4780
- std: 4003853.6073
- min: 10075.1601
- max: 21944571.7835
- correlation_with_target: -0.3280

### max_trade_size_5m
- mean: 5978884.3090
- std: 10476045.8373
- min: 13582.2153
- max: 43215953.3830
- correlation_with_target: -0.3156

### buy_sell_ratio_5m
- mean: 0.5007
- std: 0.4530
- min: 0.0000
- max: 1.7383
- correlation_with_target: -0.3279

### buy_sell_imbalance_5m
- mean: -0.4372
- std: 0.3657
- min: -1.0000
- max: 0.2696
- correlation_with_target: -0.3495

### large_trade_ratio_5m
- mean: 0.2021
- std: 0.0206
- min: 0.2000
- max: 0.4000
- correlation_with_target: -0.0587

### volume_momentum_5m
- mean: 268.3139
- std: 506.1327
- min: -0.2073
- max: 1484.5347
- correlation_with_target: 0.9756

### volume_momentum_std_5m
- mean: 534.6214
- std: 1010.0207
- min: 0.5283
- max: 2960.2450
- correlation_with_target: 0.9759

### volume_acceleration_5m
- mean: -246.1061
- std: 658.3265
- min: -1974.9637
- max: 100.1567
- correlation_with_target: -0.8121

### volume_10m
- mean: 20822105.5449
- std: 30419402.1252
- min: 84698.8964
- max: 157325744.4804
- correlation_with_target: -0.3788

### trades_count_10m
- mean: 10.0000
- std: 0.0000
- min: 10.0000
- max: 10.0000
- correlation_with_target: nan

### avg_trade_size_10m
- mean: 2082210.5545
- std: 3041940.2125
- min: 8469.8896
- max: 15732574.4480
- correlation_with_target: -0.3788

### max_trade_size_10m
- mean: 7862398.5089
- std: 10119656.9925
- min: 56128.1076
- max: 43215953.3830
- correlation_with_target: -0.4301

### buy_sell_ratio_10m
- mean: 0.8516
- std: 0.6160
- min: 0.0000
- max: 1.9652
- correlation_with_target: -0.0792

### buy_sell_imbalance_10m
- mean: -0.2137
- std: 0.4242
- min: -1.0000
- max: 0.3255
- correlation_with_target: 0.0819

### large_trade_ratio_10m
- mean: 0.1011
- std: 0.0103
- min: 0.1000
- max: 0.2000
- correlation_with_target: -0.0587

### volume_momentum_10m
- mean: 130.6388
- std: 221.3324
- min: 2.5459
- max: 660.9495
- correlation_with_target: 0.9759

### volume_momentum_std_10m
- mean: 380.1741
- std: 661.1413
- min: 6.9864
- max: 1973.9861
- correlation_with_target: 0.9730

### volume_acceleration_10m
- mean: -0.4944
- std: 7.7675
- min: -38.2150
- max: 34.3747
- correlation_with_target: -0.1003

### volume_15m
- mean: 20822105.5449
- std: 30419402.1252
- min: 84698.8964
- max: 157325744.4804
- correlation_with_target: -0.3788

### trades_count_15m
- mean: 10.0000
- std: 0.0000
- min: 10.0000
- max: 10.0000
- correlation_with_target: nan

### avg_trade_size_15m
- mean: 2082210.5545
- std: 3041940.2125
- min: 8469.8896
- max: 15732574.4480
- correlation_with_target: -0.3788

### max_trade_size_15m
- mean: 7862398.5089
- std: 10119656.9925
- min: 56128.1076
- max: 43215953.3830
- correlation_with_target: -0.4301

### buy_sell_ratio_15m
- mean: 0.8516
- std: 0.6160
- min: 0.0000
- max: 1.9652
- correlation_with_target: -0.0792

### buy_sell_imbalance_15m
- mean: -0.2137
- std: 0.4242
- min: -1.0000
- max: 0.3255
- correlation_with_target: 0.0819

### large_trade_ratio_15m
- mean: 0.1011
- std: 0.0103
- min: 0.1000
- max: 0.2000
- correlation_with_target: -0.0587

### volume_momentum_15m
- mean: 130.6388
- std: 221.3324
- min: 2.5459
- max: 660.9495
- correlation_with_target: 0.9759

### volume_momentum_std_15m
- mean: 380.1741
- std: 661.1413
- min: 6.9864
- max: 1973.9861
- correlation_with_target: 0.9730

### volume_acceleration_15m
- mean: -0.4944
- std: 7.7675
- min: -38.2150
- max: 34.3747
- correlation_with_target: -0.1003

### volume_trend_slope
- mean: 235583.2493
- std: 536795.9729
- min: -387290.8134
- max: 1455429.2164
- correlation_with_target: -0.2456

### volume_trend_r2
- mean: 0.2150
- std: 0.2005
- min: 0.0046
- max: 0.5550
- correlation_with_target: 0.0269

### volume_trend_strength
- mean: 0.1339
- std: 0.0745
- min: 0.0225
- max: 0.2461
- correlation_with_target: 0.1193

### volume_acceleration
- mean: -0.4944
- std: 7.7675
- min: -38.2150
- max: 34.3747
- correlation_with_target: -0.1003

### trade_interval_skew
- mean: -1.5943
- std: 0.5544
- min: -2.8733
- max: -0.0683
- correlation_with_target: -0.2670

### cluster_count
- mean: 1.3212
- std: 0.4674
- min: 1.0000
- max: 2.0000
- correlation_with_target: -0.3780

### clustering_ratio
- mean: 0.1321
- std: 0.0467
- min: 0.1000
- max: 0.2000
- correlation_with_target: -0.3780

### liquidity
- mean: 21720.1628
- std: 27026.0571
- min: 0.0000
- max: 135122.4071
- correlation_with_target: -0.4436

### market_count
- mean: 1.3105
- std: 0.8389
- min: 1.0000
- max: 5.0000
- correlation_with_target: -0.2043

### holder_count
- mean: 788.1692
- std: 644.7565
- min: 19.0000
- max: 2867.0000
- correlation_with_target: -0.6238

### unique_wallets_24h
- mean: 26.0921
- std: 68.7057
- min: 1.0000
- max: 426.0000
- correlation_with_target: -0.1953

### active_ratio
- mean: 0.0264
- std: 0.0265
- min: 0.0008
- max: 0.1486
- correlation_with_target: 0.1607

### average_holding
- mean: 0.0000
- std: 0.0000
- min: 0.0000
- max: 0.0000
- correlation_with_target: nan

### trades_24h
- mean: 53.0535
- std: 196.5451
- min: 0.0000
- max: 1478.0000
- correlation_with_target: -0.1467

### trades_per_holder
- mean: 0.0314
- std: 0.0703
- min: 0.0000
- max: 0.5155
- correlation_with_target: -0.2449

### volume_per_holder
- mean: 0.0000
- std: 0.0000
- min: 0.0000
- max: 0.0000
- correlation_with_target: nan

### trend_strength_2m
- mean: 3.8570
- std: 16.1570
- min: 0.1173
- max: 257.6799
- correlation_with_target: -0.1080

### trend_consistency_2m
- mean: 0.9154
- std: 0.1876
- min: 0.5000
- max: 1.0000
- correlation_with_target: 0.2454

### momentum_strength_2m
- mean: 3.8570
- std: 16.1570
- min: 0.1173
- max: 257.6799
- correlation_with_target: -0.1080

### momentum_strength_3m
- mean: 1.7642
- std: 2.0181
- min: 0.1379
- max: 8.9421
- correlation_with_target: -0.3231

### volatility_acceleration_1m
- mean: nan
- std: nan
- min: nan
- max: nan
- correlation_with_target: nan

### volatility_acceleration_3m
- mean: 191.3465
- std: 468.4038
- min: 0.1260
- max: 1442.4257
- correlation_with_target: 0.4353

### volatility_acceleration_5m
- mean: 171.4017
- std: 461.0393
- min: 0.2422
- max: 1430.1502
- correlation_with_target: 0.4603

### volatility_acceleration_10m
- mean: 140.6875
- std: 381.7241
- min: 0.6099
- max: 1183.0564
- correlation_with_target: 0.4616

### volatility_acceleration_15m
- mean: 130.2383
- std: 352.2440
- min: 0.9826
- max: 1091.7244
- correlation_with_target: 0.4619

### volatility_adjusted_momentum_3m
- mean: 1.7642
- std: 2.0181
- min: 0.1379
- max: 8.9421
- correlation_with_target: -0.3231

### volatility_adjusted_momentum_5m
- mean: 1.0577
- std: 1.0097
- min: 0.0936
- max: 4.0732
- correlation_with_target: -0.3316

### volatility_adjusted_momentum_10m
- mean: 0.5657
- std: 0.3576
- min: 0.0973
- max: 1.3259
- correlation_with_target: -0.3772

### volatility_adjusted_momentum_15m
- mean: 0.3869
- std: 0.1533
- min: 0.0671
- max: 0.7741
- correlation_with_target: -0.4485

### rsi_trend_strength_7
- mean: 3.8263
- std: 5.1507
- min: 0.0000
- max: 51.1363
- correlation_with_target: -0.3927

### rsi_trend_strength_14
- mean: 17.3559
- std: 83.7476
- min: 0.0000
- max: 739.1134
- correlation_with_target: -0.1106

### rsi_trend_strength_21
- mean: 0.0000
- std: 0.0000
- min: 0.0000
- max: 0.0000
- correlation_with_target: nan

### data_points_count
- mean: 15.9957
- std: 0.0925
- min: 14.0000
- max: 16.0000
- correlation_with_target: nan

### trades_count
- mean: 10.0000
- std: 0.0000
- min: 10.0000
- max: 10.0000
- correlation_with_target: nan

