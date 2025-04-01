"""MACD Strategy"""

import pandas as pd
import numpy as np

csv_filename = "crypto_prices_30_days.csv"
df = pd.read_csv(csv_filename)

df["Date"] = pd.to_datetime(df["Date"], unit="s")

df.sort_values(by=["Coin", "Date"], inplace=True)

strategy_results = {}

short_window = 12
long_window = 26
signal_window = 9
initial_capital = 1000

for coin in df["Coin"].unique():
    coin_df = df[df["Coin"] == coin].copy()

    coin_df["EMA_12"] = coin_df["Closing Price"].ewm(span=short_window, adjust=False).mean()
    coin_df["EMA_26"] = coin_df["Closing Price"].ewm(span=long_window, adjust=False).mean()
    coin_df["MACD"] = coin_df["EMA_12"] - coin_df["EMA_26"]
    coin_df["Signal Line"] = coin_df["MACD"].ewm(span=signal_window, adjust=False).mean()

    coin_df["Signal"] = 0
    coin_df.loc[coin_df["MACD"] > coin_df["Signal Line"], "Signal"] = 1  # Buy
    coin_df.loc[coin_df["MACD"] < coin_df["Signal Line"], "Signal"] = -1  # Sell

    coin_df["Daily Return"] = coin_df["Closing Price"].pct_change()
    coin_df["Strategy Return"] = coin_df["Signal"].shift(1) * coin_df["Daily Return"]

    coin_df["Cumulative Strategy Return"] = (1 + coin_df["Strategy Return"]).cumprod()

    cumulative_return = coin_df["Cumulative Strategy Return"].iloc[-1] - 1
    num_days = (coin_df["Date"].iloc[-1] - coin_df["Date"].iloc[0]).days
    annualized_return = (1 + cumulative_return) ** (252 / num_days) - 1
    sharpe_ratio = (coin_df["Strategy Return"].mean() / coin_df["Strategy Return"].std()) * np.sqrt(252)

    rolling_max = coin_df["Cumulative Strategy Return"].cummax()
    drawdown = (coin_df["Cumulative Strategy Return"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    strategy_results[coin] = {
        "Cumulative Return (%)": cumulative_return * 100,
        "Annualized Return (%)": annualized_return * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown * 100
    }

print("\nMACD Strategy Performance Metrics:\n")
for coin, metrics in strategy_results.items():
    print(f"Coin: {coin}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    print()