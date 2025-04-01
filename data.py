"""Dataset Generation using Coin API
"""

import requests
import csv

coins = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
base_url = "https://data-api.coindesk.com/index/cc/v1/historical/days"

csv_filename = "crypto_prices_30_days.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(["Date", "Coin", "Opening Price", "Highest Price", "Lowest Price", "Closing Price", "Trading Volume"])

    for coin in coins:
        params = {
            "market": "cadli",
            "instrument": f"{coin}-USD",
            "limit": 30,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON"
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()

            if "Data" in data:
                for entry in data["Data"]:
                    date = entry["TIMESTAMP"]
                    open_price = entry["OPEN"]
                    high_price = entry["HIGH"]
                    low_price = entry["LOW"]
                    close_price = entry["CLOSE"]
                    volume = entry["VOLUME"]

                    writer.writerow([date, coin, open_price, high_price, low_price, close_price, volume])

                print(f"Successfully fetched data for {coin}.")
            else:
                print(f"No data found for {coin}.")
        else:
            print(f"Failed to fetch data for {coin}. HTTP Status Code: {response.status_code}")

print(f"Data saved to {csv_filename}")