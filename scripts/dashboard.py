import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import joblib

from features import generate_features
from risk_metrics import (
    calculate_percent_return,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio
)

# ────────────────────────────────────────
# Streamlit UI Setup
# ────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Quant Terminal")
st.title("Your Terminal...")

# Sidebar Inputs
symbol = st.sidebar.text_input("Enter Symbol (e.g. RELIANCE.BSE)").upper()
api_key = 'J3Z6VNICGJYOXTOD'
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Hyperparameters
vol_window = st.sidebar.slider("Volatility Window", 5, 30, 14)
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
sma_ema_window = st.sidebar.slider("SMA/EMA Window", 5, 100, 20)
threshold_mult = st.sidebar.slider("Shock Threshold (std dev)", 1.0, 3.0, 2.0, step=0.1)

# ────────────────────────────────────────
# Data Fetch Function
# ────────────────────────────────────────
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

# ────────────────────────────────────────
# MAIN LOGIC
# ────────────────────────────────────────
if st.button("Run Analysis"):
    with st.spinner("Fetching and analyzing data..."):
        df = fetch_data(symbol, api_key)

        if df is None:
            st.error("⚠️ Failed to fetch data. Please check the symbol or API key.")
        else:
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            df = generate_features(df)

            # ────────────────────────────────────────
            # ML Prediction
            # ────────────────────────────────────────
            try:
                st.subheader("🤖 AI Model Prediction")

                model_path = os.path.join(os.path.dirname(__file__), "trained_model_full_v1.joblib")
                model = joblib.load(model_path)

                latest_df = generate_features(df.copy())
                latest_features = latest_df.iloc[-1].to_dict()

                for col in ['Label', 'Close', 'Open', 'High', 'Low', 'Volume', 'log_return']:
                    latest_features.pop(col, None)

                prediction = model.predict_one(latest_features)
                proba = model.predict_proba_one(latest_features)

                confidence = proba.get(prediction, 0) * 100  # confidence in %
                #st.write("latest_features used for prediction: ", latest_features)

                # Apply threshold logic
                if confidence < 40:
                    st.warning(f"⚠️ Signal confidence is too low ({confidence:.2f}%). Defaulting to **HOLD**.")
                    st.info("🟡 Suggested Action: HOLD")
                else:
                    if prediction == "BUY":
                        st.success(f"📈 **BUY** Signal with {confidence:.2f}% confidence")
                    elif prediction == "SELL":
                        st.error(f"📉 **SELL** Signal with {confidence:.2f}% confidence")
                    else:
                        st.info(f"🟡 HOLD Signal with {confidence:.2f}% confidence")

            except Exception as e:
                st.warning(f"⚠️ Could not make ML prediction: {e}")

            # ────────────────────────────────────────
            # Backtest Simulation
            # ────────────────────────────────────────
            df['signal'] = df['Label']
            initial_capital = 100000
            capital = initial_capital
            position = 0
            shares = 0
            trade_log = []

            for i in range(len(df)):
                row = df.iloc[i]
                price = row['Close']
                date = df.index[i]

                if row['Label'] == 'BUY' and position == 0:
                    shares = capital // price
                    entry_price = price
                    entry_date = date
                    position = 1

                elif row['Label'] == 'SELL' and position == 1:
                    exit_price = price
                    pnl = (exit_price - entry_price) * shares
                    capital = shares * exit_price
                    trade_log.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': exit_price,
                        'Shares': shares,
                        'PnL': pnl,
                        'Capital After Trade': capital
                    })
                    position = 0
                    shares = 0

            # ────────────────────────────────────────
            # Show Backtest Result
            # ────────────────────────────────────────
            if trade_log:
                st.subheader("📋 Trade Log")
                trade_df = pd.DataFrame(trade_log)
                st.dataframe(trade_df)

                total_pnl = trade_df['PnL'].sum()
                final_capital = trade_df['Capital After Trade'].iloc[-1]
                equity_series = [initial_capital] + list(trade_df['Capital After Trade'])

                st.metric("💰 Total Profit/Loss", f"₹{total_pnl:.2f}")
                st.metric("📈 Final Capital", f"₹{final_capital:.2f}")

                percent_return = calculate_percent_return(initial_capital, final_capital)
                sharpe = calculate_sharpe_ratio(trade_df['PnL'])
                sortino = calculate_sortino_ratio(trade_df['PnL'])
                max_dd = calculate_max_drawdown(pd.Series(equity_series))
                calmar = calculate_calmar_ratio(trade_df['PnL'].mean(), max_dd)

                col1, col2, col3 = st.columns(3)
                col1.metric("📈 % Return", f"{percent_return:.2f}%")
                col2.metric("📉 Max Drawdown", f"₹{max_dd:.2f}")
                col3.metric("📊 Calmar Ratio", f"{calmar:.2f}")

                col4, col5 = st.columns(2)
                col4.metric("⚖️ Sharpe Ratio", f"{sharpe:.2f}")
                col5.metric("📉 Sortino Ratio", f"{sortino:.2f}")

                # Equity Curve Plot
                st.subheader("📉 Equity Curve")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=list(range(len(equity_series))),
                    y=equity_series,
                    name="Equity",
                    mode="lines+markers",
                    line=dict(color="cyan")
                ))
                fig_eq.update_layout(
                    template="plotly_dark",
                    height=400,
                    xaxis_title="Trade #",
                    yaxis_title="Equity"
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            else:
                st.warning("📉 No completed BUY→SELL trades found in this time window.")

            # ────────────────────────────────────────
            # Price Chart
            # ────────────────────────────────────────
            st.subheader("📊 Price Chart with Signals")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.08,
                                subplot_titles=[f"{symbol} Price Chart", "RSI + MACD + Volatility"])

            # Row 1: Price & Signals
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close", line=dict(color='white')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], name="SMA", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA'], name="EMA", line=dict(color='violet')), row=1, col=1)

            buy_signals = df[df['Label'] == 'BUY']
            sell_signals = df[df['Label'] == 'SELL']

            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers',
                                     name="BUY", marker=dict(symbol='triangle-up', color='green', size=10)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers',
                                     name="SELL", marker=dict(symbol='triangle-down', color='red', size=10)),
                          row=1, col=1)

            # Row 2: RSI + MACD + Volatility
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='lime')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='aqua')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="MACD Signal", line=dict(color='magenta')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], name="Volatility", line=dict(color='yellow')), row=2, col=1)

            fig.update_layout(height=800, template="plotly_dark", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # ────────────────────────────────────────
            # CSV Export
            # ────────────────────────────────────────
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer)
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"fragility_{symbol.replace('.', '_')}.csv",
                mime="text/csv"
            )

