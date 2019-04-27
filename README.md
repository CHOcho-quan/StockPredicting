# StockPredicting
Stock midprice predicting using multiple methods

Currently the data is cleaned by meaning the data inside 1 second. Using 10 x 108 indicators to predict the price of after 10 seconds. The result is measured by MSE loss.

| Method \ MSE Loss | with PCA | without PCA |
| ----------------- | -------- | ----------- |
| LSTM Network      |          | 2.29        |
| GBDT              |          | 2.19        |
| Random Forest     | 2.25     | 2.29        |



