import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("POLYGON_API")

# Define the API endpoint
url = "https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/prev?adjusted=true&apiKey=aa941qnhzWcD2z3DAaPHlh3djfKyG1Tj"

# Set up parameters
params = {
    "adjusted": "true",
    "apiKey": api_key
}

# Make the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Convert response to JSON
    result = data["results"][0]
    day_open = result["o"]
    day_high = result["h"]
    day_low = result["l"]
    day_close = result["c"]
    trading_volume = result["v"]

    # Printing values
    print(f"Opening Price: {day_open}")
    print(f"Highest Price: {day_high}")
    print(f"Lowest Price: {day_low}")
    print(f"Closing Price: {day_close}")
    print(f"Trading Volume: {trading_volume}")
else:
    print(f"Error {response.status_code}: {response.text}")
