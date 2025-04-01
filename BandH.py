"""Simple Buy and Hold Strategy"""

import pandas as pd
import numpy as np

csv_filename = "crypto_prices_30_days.csv"
df = pd.read_csv(csv_filename)

df["Date"] = pd.to_datetime(df["Date"], unit="s")

df.sort_values(by=["Coin", "Date"], inplace=True)

df_pivot = df.pivot(index="Date", columns="Coin", values="Closing Price")

returns = df_pivot.pct_change()

initial_prices = df_pivot.iloc[0]
final_prices = df_pivot.iloc[-1]

cumulative_returns = (final_prices - initial_prices) / initial_prices

num_days = (df_pivot.index[-1] - df_pivot.index[0]).days
annualized_return = (1 + cumulative_returns) ** (252 / num_days) - 1

risk_free_rate = 0.00
mean_daily_return = returns.mean()
std_daily_return = returns.std()
sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252)

cumulative_returns_series = (1 + returns).cumprod()
rolling_max = cumulative_returns_series.cummax()
drawdown = (cumulative_returns_series - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("Buy and Hold Strategy Performance Metrics:\n")
print("Cumulative Return (%):")
print(cumulative_returns * 100, "\n")

print("Annualized Return (%):")
print(annualized_return * 100, "\n")

print("Sharpe Ratio:")
print(sharpe_ratio, "\n")

print("Maximum Drawdown (%):")
print(max_drawdown * 100)