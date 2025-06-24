import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_fragility(csv_path, window=14):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # STEP 1: Ensure log_return & volatility exist
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    if 'volatility' not in df.columns:
        df['volatility'] = df['log_return'].rolling(window=window).std()

    # STEP 2: Define "non-fundamental" shocks → extreme returns
    threshold = df['log_return'].std() * 2  # 2-sigma rule
    df['nonfund_shock'] = np.where(abs(df['log_return']) > threshold, 1, 0)

    # STEP 3: Calculate non-fundamental volatility
    df['nonfund_vol'] = df['log_return'] * df['nonfund_shock']
    df['rolling_nonfund_vol'] = df['nonfund_vol'].rolling(window=window).std()

    # STEP 4: Fragility Ratio = nonfund vol / total vol
    df['fragility_ratio'] = df['rolling_nonfund_vol'] / df['volatility']

    # STEP 5: Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(df['volatility'], label='Rolling Volatility', color='blue')
    plt.plot(df['rolling_nonfund_vol'], label='Non-Fundamental Volatility', color='red')
    plt.title('Volatility vs Non-Fundamental Volatility')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['fragility_ratio'], label='Fragility Ratio', color='purple')
    plt.axhline(1, color='gray', linestyle='--', linewidth=1)
    plt.title('Fragility Ratio Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save fragility-enhanced data
    out_path = csv_path.replace(".csv", "_fragility.csv")
    df.to_csv(out_path)
    print(f"✅ Fragility data saved to: {out_path}")

    return df

if __name__ == "__main__":
    symbol = input("Enter symbol used in filename (e.g., RELIANCE.BSE): ").upper()
    csv_path = f"E:\HFT\hft_fragility_system\data/{symbol}_daily_alpha_volatility.csv"
    calculate_fragility(csv_path)
