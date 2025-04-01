"""KDJ + RSI Strategy"""

import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("crypto_prices_30_days.csv")

def calculate_rsi(df, period=14):
    delta = df['Closing Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi  # Use .loc to avoid slice warning
    return df

# Function to calculate KDJ
def calculate_kdj(df, period=14):
    low_min = df['Lowest Price'].rolling(window=period).min()
    high_max = df['Highest Price'].rolling(window=period).max()
    rsv = (df['Closing Price'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()  # Use .loc to avoid slice warning
    df['D'] = df['K'].ewm(com=2).mean()  # Use .loc to avoid slice warning
    df['J'] = 3 * df['K'] - 2 * df['D']  # Use .loc to avoid slice warning
    return df

# Function to compute trading signals
def trading_signals(df):
    buy_signals = []
    sell_signals = []
    for i in range(len(df)):
        buy_signals.append(1 if df['K'].iloc[i] < 20 and df['RSI'].iloc[i] < 30 else 0)
        sell_signals.append(-1 if df['K'].iloc[i] > 80 and df['RSI'].iloc[i] > 70 else 0)

    df['Buy Signal'] = buy_signals
    df['Sell Signal'] = sell_signals

# Function to calculate performance metrics
def calculate_performance_metrics(df):
    df['Market Return'] = df['Closing Price'].pct_change()
    df['Strategy Return'] = df['Market Return'] * (df['Buy Signal'].shift(1) + df['Sell Signal'].shift(1))

    cumulative_return = (1 + df['Strategy Return']).cumprod() - 1
    annual_return = cumulative_return.iloc[-1] / len(df) * 365
    sharpe_ratio = (df['Strategy Return'].mean() / df['Strategy Return'].std()) * np.sqrt(252)  # Assuming trading days in a year
    max_drawdown = (cumulative_return.cummax() - cumulative_return).max()

    return cumulative_return, annual_return, sharpe_ratio, max_drawdown

# Process data for each coin
performance_results = {}
for coin in data['Coin'].unique():
    coin_data = data[data['Coin'] == coin].copy()  # Create a copy to avoid SettingWithCopyWarning
    coin_data = calculate_rsi(coin_data)
    coin_data = calculate_kdj(coin_data)
    trading_signals(coin_data)

    # Calculate performance metrics
    metrics = calculate_performance_metrics(coin_data)
    performance_results[coin] = {
        'Cumulative Return': metrics[0],
        'Annual Return': metrics[1],
        'Sharpe Ratio': metrics[2],
        'Max Drawdown': metrics[3]
    }


for coin, results in performance_results.items():
    print(f"\nPerformance metrics for {coin}:")
    for metric, value in results.items():
        if isinstance(value, pd.Series):
            value = value.iloc[-1]  # Get the last value in the Series

        # Format output
        if metric in ['Cumulative Return', 'Annual Return', 'Max Drawdown']:
            print(f"{metric}: {value * 100:.2f}%")  # Multiply by 100 for percentage
        else:
            print(f"{metric}: {value:.4f}")  # Sharpe Ratio prints as is
