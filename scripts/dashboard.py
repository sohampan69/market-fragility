import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Streamlit Terminal Theme Settings ----
st.set_page_config(layout="wide", page_title="QUANT TERMINAL")

# ---- Force Sidebar to Stay Open ----

# FORCE SIDEBAR TO OPEN ON LOAD
st.markdown(
    """
    <script>
        window.addEventListener('load', function() {
            const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            const toggler = window.parent.document.querySelector('button[title="Hide sidebar"]');

            if (sidebar && toggler && sidebar.offsetWidth === 0) {
                toggler.click();
            }
        });
    </script>
    """,
    unsafe_allow_html=True
)



st.markdown("""
    <style>
        .reportview-container {
            background-color: #0d1117;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #161b22;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00FF41;
        }
        .stButton>button {
            background-color: #00FF41;
            color: black;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#00FF41;'>💻 QUANT TERMINAL - Fragility + Indicators</h1>", unsafe_allow_html=True)

# ---- INPUT SECTION (SIDEBAR) ----
st.sidebar.title("📊 Terminal Controls")
symbol = st.sidebar.text_input("Symbol (e.g., RELIANCE.BSE)", "RELIANCE.BSE")
api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
vol_window = st.sidebar.slider("Volatility Window (days)", 5, 30, 14)
thresh_mult = st.sidebar.slider("Shock Threshold (std dev)", 1.0, 3.0, 2.0, step=0.1)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
sma_period = st.sidebar.slider("SMA Period", 5, 100, 20)
ema_period = st.sidebar.slider("EMA Period", 5, 100, 20)

# ---- DATA FETCH ----
@st.cache_data
def fetch_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if "Time Series (Daily)" not in data:
        return None
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# ---- RSI CALCULATION ----
def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---- RUN ANALYSIS ----
if st.sidebar.button("Run Analysis"):
    with st.spinner("Processing live data..."):
        df = fetch_data(symbol, api_key)

        if df is None:
            st.error("❌ API failed. Check symbol or key.")
        else:
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['volatility'] = df['log_return'].rolling(window=vol_window).std()
            threshold = df['log_return'].std() * thresh_mult
            df['shock'] = np.where(abs(df['log_return']) > threshold, 1, 0)
            df['nonfund_vol'] = df['log_return'] * df['shock']
            df['rolling_nonfund_vol'] = df['nonfund_vol'].rolling(window=vol_window).std()
            df['fragility_ratio'] = df['rolling_nonfund_vol'] / df['volatility']

            df['RSI'] = calculate_rsi(df['Close'], rsi_period)
            df['SMA'] = df['Close'].rolling(window=sma_period).mean()
            df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

            df['Signal'] = np.where((df['RSI'] < 30) & (df['Close'] > df['SMA']), 'BUY',
                              np.where((df['RSI'] > 70) & (df['Close'] < df['SMA']), 'SELL', 'HOLD'))

            # ---- CHART ----
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Close Price + SMA/EMA", "Volatility vs Shock Vol", "Fragility Ratio", "RSI"))

            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='#00FF41')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], name="SMA", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], name="EMA", line=dict(color='cyan')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], name="Volatility", line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['rolling_nonfund_vol'], name="Shock Volatility", line=dict(color='red')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['fragility_ratio'], name="Fragility Ratio", line=dict(color='purple')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=[1]*len(df), name="Threshold=1", line=dict(dash='dash', color='gray')), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='violet')), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), name="RSI=30", line=dict(dash='dash', color='green')), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), name="RSI=70", line=dict(dash='dash', color='red')), row=4, col=1)

            fig.update_layout(height=1000, paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font_color='white')

            st.plotly_chart(fig, use_container_width=True)

            df.to_csv(f"fragility_{symbol.replace('.', '_')}.csv")
            st.success("✅ Terminal run complete. Data saved locally.")

            # ---- SIGNAL SNAPSHOT ----
            st.markdown("<h3 style='color:#00FF41;'>📌 Trading Signal Summary</h3>", unsafe_allow_html=True)
            last_signal = df['Signal'].iloc[-1]
            st.markdown(f"<h4 style='color:yellow;'>Signal: <span style='color:#00FF41;'>{last_signal}</span></h4>", unsafe_allow_html=True)
