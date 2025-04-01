"""ZMR + SMA Baseline"""

import pandas as pd
import numpy as np

# Load data from CSV
csv_filename = "crypto_prices_30_days.csv"
df = pd.read_csv(csv_filename)

# Convert timestamp to datetime format
df["Date"] = pd.to_datetime(df["Date"], unit="s")

# Sort data by date for each coin
df.sort_values(by=["Coin", "Date"], inplace=True)

# Strategy Parameters
sma_period = 50  # SMA window
zmr_threshold = 1.5  # Z-score threshold
initial_capital = 10000  # Assume $10,000 investment per coin
trading_days_per_year = 252  # Approximate number of trading days in a year

strategy_results = {}

for coin in df["Coin"].unique():
    coin_df = df[df["Coin"] == coin].copy()

    # SMA Strategy
    coin_df["SMA"] = coin_df["Closing Price"].rolling(window=sma_period).mean()
    coin_df["SMA Signal"] = 0
    coin_df.loc[coin_df["Closing Price"] > coin_df["SMA"], "SMA Signal"] = 1  # Buy when price is above SMA
    coin_df.loc[coin_df["Closing Price"] < coin_df["SMA"], "SMA Signal"] = -1  # Sell when price is below SMA

    # ZMR (Zero Mean Reversion) Strategy
    coin_df["Mean"] = coin_df["Closing Price"].rolling(window=sma_period).mean()
    coin_df["Std Dev"] = coin_df["Closing Price"].rolling(window=sma_period).std()
    coin_df["Z-score"] = (coin_df["Closing Price"] - coin_df["Mean"]) / coin_df["Std Dev"]

    coin_df["ZMR Signal"] = 0
    coin_df.loc[coin_df["Z-score"] < -zmr_threshold, "ZMR Signal"] = 1  # Buy when price is far below mean
    coin_df.loc[coin_df["Z-score"] > zmr_threshold, "ZMR Signal"] = -1  # Sell when price is far above mean

    # Strategy Returns
    coin_df["Daily Return"] = coin_df["Closing Price"].pct_change()

    # SMA Strategy Returns
    coin_df["SMA Strategy Return"] = coin_df["SMA Signal"].shift(1) * coin_df["Daily Return"]
    coin_df["Cumulative SMA Return"] = (1 + coin_df["SMA Strategy Return"]).cumprod()

    # ZMR Strategy Returns
    coin_df["ZMR Strategy Return"] = coin_df["ZMR Signal"].shift(1) * coin_df["Daily Return"]
    coin_df["Cumulative ZMR Return"] = (1 + coin_df["ZMR Strategy Return"]).cumprod()

    # Performance Metrics for SMA Strategy
    cumulative_sma_return = coin_df["Cumulative SMA Return"].iloc[-1] - 1
    sharpe_sma = (coin_df["SMA Strategy Return"].mean() / coin_df["SMA Strategy Return"].std()) * np.sqrt(trading_days_per_year)

    # Maximum Drawdown for SMA
    rolling_max_sma = coin_df["Cumulative SMA Return"].cummax()
    drawdown_sma = (coin_df["Cumulative SMA Return"] - rolling_max_sma) / rolling_max_sma
    max_drawdown_sma = drawdown_sma.min()

    # Annualized Return for SMA
    num_days = len(coin_df)
    annual_return_sma = (1 + cumulative_sma_return) ** (trading_days_per_year / num_days) - 1

    # Performance Metrics for ZMR Strategy
    cumulative_zmr_return = coin_df["Cumulative ZMR Return"].iloc[-1] - 1
    sharpe_zmr = (coin_df["ZMR Strategy Return"].mean() / coin_df["ZMR Strategy Return"].std()) * np.sqrt(trading_days_per_year)

    # Maximum Drawdown for ZMR
    rolling_max_zmr = coin_df["Cumulative ZMR Return"].cummax()
    drawdown_zmr = (coin_df["Cumulative ZMR Return"] - rolling_max_zmr) / rolling_max_zmr
    max_drawdown_zmr = drawdown_zmr.min()

    # Annualized Return for ZMR
    annual_return_zmr = (1 + cumulative_zmr_return) ** (trading_days_per_year / num_days) - 1

    # Store results separately for SMA and ZMR
    strategy_results[coin] = {
        "SMA Cumulative Return (%)": cumulative_sma_return * 100,
        "SMA Sharpe Ratio": sharpe_sma,
        "SMA Maximum Drawdown (%)": max_drawdown_sma * 100,
        "SMA Annual Return (%)": annual_return_sma * 100,
        "ZMR Cumulative Return (%)": cumulative_zmr_return * 100,
        "ZMR Sharpe Ratio": sharpe_zmr,
        "ZMR Maximum Drawdown (%)": max_drawdown_zmr * 100,
        "ZMR Annual Return (%)": annual_return_zmr * 100
    }

# Print results
print("\nSMA and ZMR Strategy Performance Metrics:\n")
for coin, metrics in strategy_results.items():
    print(f"Coin: {coin}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    print()