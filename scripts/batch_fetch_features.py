import time
from features import generate_features
import pandas as pd
import requests
import os

symbols = ["PIDILITIND.BSE" , "HINDALCO.BSE" , "NATIONALUM.BSE" , "GSMFOILS.BSE" , "BPCL.BSE"

]

api_key = 'SDIQBWBXP3O9KQ9W'

def fetch_and_save(symbol):
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}"
    )

    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print(f"❌ Failed for {symbol}. Reason: {data.get('Note') or data}")
        return

    raw = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(raw, orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Generate features
    feature_df = generate_features(df)

    # Save file
    os.makedirs("data", exist_ok=True)
    safe_symbol = symbol.replace(".", "_")
    feature_df.to_csv(f"data/features_{safe_symbol}.csv")
    print(f"✅ {symbol} → data/features_{safe_symbol}.csv")

# ─────────────────────────────────────
# Run batch fetch
# ─────────────────────────────────────
if __name__ == "__main__":
    for symbol in symbols:
        fetch_and_save(symbol)
        time.sleep(10)  # Respect Alpha Vantage free tier limit
