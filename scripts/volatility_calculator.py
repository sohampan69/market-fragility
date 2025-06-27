import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_volatility(csv_path, window=14):
    # Load CSV data
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Calculate log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Calculate rolling volatility (standard deviation of log returns)
    df['volatility'] = df['log_return'].rolling(window=window).std()

    # Plotting the Close price and Volatility
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Close Price')
    plt.title('Stock Close Price')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['volatility'], label=f'{window}-Day Rolling Volatility', color='orange')
    plt.title('Volatility')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ✅ Save the DataFrame with volatility column
    out_path = csv_path.replace(".csv", "_volatility.csv")
    df.to_csv(out_path)
    print(f"✅ Volatility data saved to: {out_path}")

    return df

# Run from here
if __name__ == "__main__":
    symbol = input("Enter symbol used in filename (e.g., RELIANCE.BSE): ").upper()
    csv_path = f"E:\HFT\hft_fragility_system\data/{symbol}_daily_alpha.csv"
    calculate_volatility(csv_path)
