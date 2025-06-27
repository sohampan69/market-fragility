import requests
import pandas as pd
import os

def fetch_alpha_vantage_data(symbol, api_key):
    # Check if the API key is provided
    if not api_key:
        print("API key is required.")
        return None

    # Construct the API URL
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}"
    )

    # Send the GET request to the API
    response = requests.get(url)
    data = response.json()

    # Check if the response contains the expected data
    if "Time Series (Daily)" not in data:
        print("Failed to fetch data. Reason:", data.get("Note") or data)
        return None

    # Process the raw data into a DataFrame
    raw = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(raw, orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    save_directory = r"E:\HFT\hft_fragility_system\data"

    # Ensure the data directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Ask user for custom date range
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # Filter DataFrame
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    # Save the DataFrame to a CSV file
    filename = os.path.join(save_directory, f"{symbol}_daily_alpha.csv")
    df.to_csv(filename)
    print(f"Data saved to {filename}")
    return df

if __name__ == "__main__":
    # Get user input for stock symbol and API key
    symbol = input("Enter the stock symbol (e.g., TCS.BSE or RELIANCE.BSE): ").upper()
    api_key = 'J3Z6VNICGJYOXTOD'

    # Fetch the data and display the first few rows
    df = fetch_alpha_vantage_data(symbol, api_key)
    if df is not None:
        print(df.head())
