import pandas as pd
import numpy as np

def generate_features(df,
                      rsi_window=14, sma_window=20, ema_window=20, bb_window=20, bb_std=2,
                      macd_fast=12, macd_slow=26, macd_signal=9, vol_window=14,
                      future_window=5, return_threshold=0.006):
    df = df.copy()

    # ───── Basic Returns & Volatility ─────
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=vol_window).std()

    # ───── RSI ─────
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ───── SMA & EMA ─────
    df['SMA'] = df['Close'].rolling(window=sma_window).mean()
    df['EMA'] = df['Close'].ewm(span=ema_window, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # ───── Bollinger Bands ─────
    df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
    df['BB_Std'] = df['Close'].rolling(window=bb_window).std()
    df['BB_Upper'] = df['BB_Middle'] + bb_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - bb_std * df['BB_Std']
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Squeeze'] = np.where(df['BB_Width'] < 0.015 * df['Close'], 1, 0)

    # ───── MACD ─────
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()

    # ───── Fragility & Shock Vol ─────
    df['shock'] = np.where(abs(df['log_return']) > df['log_return'].std(), 1, 0)
    df['non-fund_vol'] = df['log_return'] * df['shock']
    df['rolling_non-fund_vol'] = df['non-fund_vol'].rolling(window=vol_window).std()
    df['fragility_ratio'] = df['rolling_non-fund_vol'] / (df['volatility'] + 1e-8)
    df['Fragile_Zone'] = np.where(df['fragility_ratio'] > 1.2, 1, 0)
    df['Low_Vol_Zone'] = np.where(df['volatility'] < 0.015, 1, 0)

    # ───── Signal Strength ─────
    df['Signal_Strength'] = 0
    df['Final_Score'] = 0
    df['Label'] = 'HOLD'

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        signals = []

        # RSI + BB
        if row['RSI'] < 25 and row['Close'] < row['BB_Lower']:
            signals.append('RSI_BB_LONG')
        elif row['RSI'] > 75 and row['Close'] > row['BB_Upper']:
            signals.append('RSI_BB_SHORT')

        # BB squeeze (sideways)
        sideways = row['BB_Width'] < 0.02 * row['Close'] and (25 <= row['RSI'] <= 70)

        # MACD + EMA200 confirmation
        macd_cross_up = prev['MACD'] < prev['MACD_Signal'] and row['MACD'] > row['MACD_Signal']
        macd_cross_down = prev['MACD'] > prev['MACD_Signal'] and row['MACD'] < row['MACD_Signal']

        if macd_cross_up and row['MACD'] < 0 and row['Close'] > row['EMA200']:
            signals.append('MACD_CONFIRMED_LONG')
        elif macd_cross_down and row['MACD'] > 0 and row['Close'] < row['EMA200']:
            signals.append('MACD_CONFIRMED_SHORT')

        # Risk dampening
        risk_penalty = row['Fragile_Zone'] + row['Low_Vol_Zone']

        score = 0
        if 'RSI_BB_LONG' in signals or 'RSI_BB_SHORT' in signals:
            score += 1
        if 'MACD_CONFIRMED_LONG' in signals or 'MACD_CONFIRMED_SHORT' in signals:
            score += 1

        final_score = max(score - risk_penalty, 0)

        df.at[df.index[i], 'Signal_Strength'] = score
        df.at[df.index[i], 'Final_Score'] = final_score

        # Assign model-friendly label (used in ML)
        if final_score >= 2:
            if 'RSI_BB_LONG' in signals or 'MACD_CONFIRMED_LONG' in signals:
                df.at[df.index[i], 'Label'] = 'BUY'
            elif 'RSI_BB_SHORT' in signals or 'MACD_CONFIRMED_SHORT' in signals:
                df.at[df.index[i], 'Label'] = 'SELL'
        elif sideways:
            df.at[df.index[i], 'Label'] = 'HOLD'

    # ───── Future Return Label Override ─────
    for i in range(len(df) - future_window):
        current_price = df['Close'].iloc[i]
        future_prices = df['Close'].iloc[i + 1:i + 1 + future_window]
        max_future = future_prices.max()
        min_future = future_prices.min()

        if (max_future - current_price) / current_price >= return_threshold:
            df.at[df.index[i], 'Label'] = 'BUY'
        elif (current_price - min_future) / current_price >= return_threshold:
            df.at[df.index[i], 'Label'] = 'SELL'
        # else keep existing label (might be HOLD or from signal combo)

    return df.dropna()
