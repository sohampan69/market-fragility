import requests
import pandas as pd
import os
from features import generate_features  # Your custom feature builder

def fetch_and_generate_features(symbol, api_key):
    if not api_key:
        print("API key is required.")
        return None

    # 1. Alpha Vantage URL for full daily data
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}"
    )

    # 2. Fetch
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print("Failed to fetch data. Reason:", data.get("Note") or data)
        return None

    # 3. Parse into DataFrame
    raw = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(raw, orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # 4. Ask for date range
    print(f"\n🕰 Available data from: {df.index.min().date()} to {df.index.max().date()}")
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    df = df[(df.index >= start_date) & (df.index <= end_date)]

    # 5. File paths
    safe_symbol = symbol.replace(".", "_")
    os.makedirs("data", exist_ok=True)
    raw_path = f"data/{safe_symbol}_{start_date}_to_{end_date}_raw.csv"
    features_path = f"data/features_{safe_symbol}_{start_date}_to_{end_date}.csv"

    # 6. Save raw data
    #df.to_csv(raw_path)
    #print(f"Raw data saved to: {raw_path}")

    # 7. Generate features
    feature_df = generate_features(df)

    # 8. Save features
    feature_df.to_csv(features_path)
    print(f"Features saved to: {features_path}")

    return feature_df

# ───────────────────────────────
if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g. RELIANCE.BSE): ").upper()
    api_key = '5GOMSQ2O4I9S6YIL'  # Keep it safe in production
    fetch_and_generate_features(symbol, api_key)
