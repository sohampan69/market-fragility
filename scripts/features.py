import pandas as pd
import numpy as np

def generate_features(df,
                      rsi_window=14, sma_window=20, ema_window=20, bb_window=20, bb_std=2,
                      macd_fast=12, macd_slow=26, macd_signal=9, vol_window=14):
    df = df.copy()

    # ───── Technical Indicators ─────
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=vol_window).std()
    # Shock and nonfundamental volatility
    threshold = df['log_return'].std()
    df['shock'] = np.where(abs(df['log_return']) > threshold, 1, 0)
    df['non-fund_vol'] = df['log_return'] * df['shock']
    df['rolling_non-fund_vol'] = df['non-fund_vol'].rolling(window=vol_window).std()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['SMA'] = df['Close'].rolling(window=sma_window).mean()
    df['EMA'] = df['Close'].ewm(span=ema_window, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
    df['BB_Std'] = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Middle'] + bb_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - bb_std * df['BB_Std']
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Squeeze'] = np.where(df['BB_Width'] < 0.015 * df['Close'], 1, 0)

    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()

    # ───── Fragility & Volatility Zones ─────
    df['fragility_ratio'] = df['rolling_non-fund_vol'] / (df['volatility'] + 1e-8)

    df['Fragile_Zone'] = np.where(df['fragility_ratio'] > 1.2, 1, 0)
    df['Low_Vol_Zone'] = np.where(df['volatility'] < 0.015, 1, 0)
    df['shock'] = np.where(abs(df['log_return']) > df['log_return'].std(), 1, 0)
    df['non-fund_vol'] = df['log_return'] * df['shock']
    df['rolling_non-fund_vol'] = df['non-fund_vol'].rolling(window=14).std()
    df['fragility_ratio'] = df['rolling_non-fund_vol'] / (df['volatility'] + 1e-8)

    # ───── Signal Strength & Labeling ─────
    df['Signal_Strength'] = 0
    df['Label'] = 'HOLD'

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        signals = []

        # RSI
        if row['RSI'] < 25:
            signals.append('RSI_LONG')
        elif row['RSI'] > 70:
            signals.append('RSI_SHORT')

        # Bollinger Band Logic
        if row['Close'] < row['BB_Lower']:
            signals.append('BB_LONG')
        elif row['Close'] > row['BB_Upper']:
            signals.append('BB_SHORT')

        # MACD cross
        macd_cross_up = prev['MACD'] < prev['MACD_Signal'] and row['MACD'] > row['MACD_Signal']
        macd_cross_down = prev['MACD'] > prev['MACD_Signal'] and row['MACD'] < row['MACD_Signal']

        if macd_cross_up and row['MACD'] < 0:
            signals.append('MACD_LONG')
        elif macd_cross_down and row['MACD'] > 0:
            signals.append('MACD_SHORT')

        # Trend Filter
        if row['Close'] > row['EMA200']:
            signals.append('BULLISH')
        elif row['Close'] < row['EMA200']:
            signals.append('BEARISH')

        # Squeeze Detection
        bb_tight = row['BB_Width'] < 0.02 * row['Close']
        sideways = bb_tight and (25 <= row['RSI'] <= 70)

        # Compute Score
        signal_score = sum([
            'RSI_LONG' in signals or 'RSI_SHORT' in signals,
            'BB_LONG' in signals or 'BB_SHORT' in signals,
            'MACD_LONG' in signals or 'MACD_SHORT' in signals
        ])

        penalty = row['Fragile_Zone'] + row['Low_Vol_Zone']
        final_score = max(signal_score - penalty, 0)

        df.at[df.index[i], 'Signal_Strength'] = final_score

        # Labeling
        long_signals = {'RSI_LONG', 'BB_LONG', 'MACD_LONG', 'BULLISH'}
        short_signals = {'RSI_SHORT', 'BB_SHORT', 'MACD_SHORT', 'BEARISH'}

        if len(long_signals.intersection(signals)) >= 2:
            df.at[df.index[i], 'Label'] = 'BUY'
        elif len(short_signals.intersection(signals)) >= 2:
            df.at[df.index[i], 'Label'] = 'SELL'
        elif sideways:
            df.at[df.index[i], 'Label'] = 'HOLD'

    return df.dropna()
