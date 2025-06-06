import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests

# CONFIG
st.set_page_config(layout="wide")
st.title("Scanner Forex Majeures ⭐ (Confluence 5-6 étoiles)")

# Paramètres
API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", "")
MAJOR_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]
INTERVAL = "1h"

# Fonctions
def get_data_twelvedata(symbol: str, interval: str, apikey: str):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={apikey}"
    r = requests.get(url)
    d = r.json()
    if d.get("status") == "ok":
        df = pd.DataFrame(d["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df = df.astype(float).sort_index()
        return df
    else:
        return None

def apply_indicators(df: pd.DataFrame):
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["hma12"] = ta.hma(df["close"], length=12)
    return df

def evaluate_confluence(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # HMA Slope
    hma_slope = 1 if last["hma12"] > prev["hma12"] else -1

    # EMA Trend
    ema_trend = 1 if last["ema20"] > last["ema50"] else -1

    # HMA vs EMA20
    hma_vs_ema = 1 if last["hma12"] > last["ema20"] else -1

    # Confluence Score
    bull = sum([hma_slope == 1, ema_trend == 1, hma_vs_ema == 1])
    bear = sum([hma_slope == -1, ema_trend == -1, hma_vs_ema == -1])
    score = bull if bull >= bear else bear

    signal = "BUY" if bull > bear and score >= 5 else ("SELL" if bear > bull and score >= 5 else "WAIT")
    stars = "⭐" * score if signal != "WAIT" else "WAIT"

    return stars, signal, hma_slope, ema_trend, hma_vs_ema

# Interface
if st.button("Lancer le Scan"):
    if not API_KEY:
        st.error("Clé API Twelve Data manquante. Ajoutez-la dans .streamlit/secrets.toml.")
    else:
        result_rows = []
        progress = st.progress(0)

        for i, pair in enumerate(MAJOR_PAIRS):
            df = get_data_twelvedata(pair, INTERVAL, API_KEY)
            if df is not None and len(df) > 50:
                df = apply_indicators(df)
                stars, signal, slope, trend, hma_pos = evaluate_confluence(df)
                result_rows.append({
                    "Paire": pair,
                    "Signal": signal,
                    "Note": stars,
                    "HMA Slope": "▲" if slope == 1 else "▼",
                    "EMA Trend": "▲" if trend == 1 else "▼",
                    "HMA>EMA20": "✔" if hma_pos == 1 else "✖",
                })
            progress.progress((i + 1) / len(MAJOR_PAIRS))

        if result_rows:
            df_result = pd.DataFrame(result_rows)
            st.dataframe(df_result, use_container_width=True)
        else:
            st.warning("Aucun résultat disponible.")

