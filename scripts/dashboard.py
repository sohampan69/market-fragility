import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
# STREAMLIT CONFIG
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Quant Terminal")
st.title("Welcome to the Terminal..")

# Sidebar inputs
symbol = st.sidebar.text_input("Enter Symbol (e.g. RELIANCE.BSE)")
api_key = '5GOMSQ2O4I9S6YIL'
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Rolling window parameters
vol_window = st.sidebar.slider("Volatility Window", 5, 30, 14)
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
sma_ema_window = st.sidebar.slider("SMA/EMA Window", 5, 100, 20)
threshold_mult = st.sidebar.slider("Shock Threshold (std dev)", 1.0, 3.0, 2.0, step=0.1)

# ─────────────────────────────────────────────
# FETCH FUNCTION
# ─────────────────────────────────────────────
@st.cache_data
def fetch_data(symbol, api_key):
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}"
    )
    r = requests.get(url)
    data = r.json()
    if "Time Series (Daily)" not in data:
        return None
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if st.button("Run Analysis"):
    with st.spinner("Crunching numbers..."):
        df = fetch_data(symbol, api_key)

        if df is None:
            st.error("Failed to fetch data. Check symbol or API key.")
        else:
            # Filter date range
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

            # Log return and volatility
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['volatility'] = df['log_return'].rolling(window=vol_window).std()

            # Shock & fragility logic
            threshold = df['log_return'].std() * threshold_mult
            df['shock'] = np.where(abs(df['log_return']) > threshold, 1, 0)
            df['nonfund_vol'] = df['log_return'] * df['shock']
            df['rolling_nonfund_vol'] = df['nonfund_vol'].rolling(window=vol_window).std()
            df['fragility_ratio'] = df['rolling_nonfund_vol'] / df['volatility']

            # SMA & EMA (user-controlled window)
            df['SMA'] = df['Close'].rolling(window=sma_ema_window).mean()
            df['EMA'] = df['Close'].ewm(span=sma_ema_window, adjust=False).mean()

            # RSI (user-controlled window)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=rsi_window).mean()
            avg_loss = loss.rolling(window=rsi_window).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # ───────────────────────────────────────
            # PLOTLY 3-SUBPLOT CHART
            # ───────────────────────────────────────
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=(
                                    f"{symbol} Price + SMA/EMA",
                                    f"Volatility vs Shock Volatility (Window={vol_window})",
                                    f"RSI ({rsi_window}-day)"
                                ),
                                vertical_spacing=0.08)

            # Row 1: Price + SMA/EMA
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='white')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], name=f"SMA-{sma_ema_window}", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], name=f"EMA-{sma_ema_window}", line=dict(color='violet')), row=1, col=1)

            # Row 2: Volatility vs Shock Volatility
            fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], name="Volatility", line=dict(color='yellow')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['rolling_nonfund_vol'], name="Shock Volatility", line=dict(color='red')), row=2, col=1)

            # Row 3: RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='lime')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), name="Overbought (70)", line=dict(color='red', dash='dash')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), name="Oversold (30)", line=dict(color='blue', dash='dash')), row=3, col=1)

            fig.update_layout(
                height=950,
                template='plotly_dark',
                title_text=f"Quant Dashboard for {symbol}",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # ───────────────────────────────────────
            # DOWNLOAD CSV
            # ───────────────────────────────────────
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer)
            st.download_button(
                label="Download Full Analysis CSV",
                data=csv_buffer.getvalue(),
                file_name=f"fragility_{symbol.replace('.', '_')}.csv",
                mime="text/csv"
            )
