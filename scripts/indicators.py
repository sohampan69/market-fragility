import pandas as pd


def add_bollinger_bands(df, window=20, num_std=2):
    """
    Adds Bollinger Bands columns to the DataFrame.
    """
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (num_std * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (num_std * df['BB_Std'])
    return df


def add_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Adds MACD and Signal line to the DataFrame.
    """
    df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df
