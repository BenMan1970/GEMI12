# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import traceback
import requests

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Scanner Confluence Forex (Twelve Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Twelve Data)")
st.markdown("*Utilisation de l'API Twelve Data pour les données de marché H1*")

# --- API CONFIG ---
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
API_KEY = st.secrets.get("TWELVE_DATA_API_KEY")
if not API_KEY:
    st.error("Clé API manquante. Ajoutez-la dans .streamlit/secrets.toml : TWELVE_DATA_API_KEY = '...' ")
    st.stop()

FOREX_PAIRS_TD = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP",
    "XAU/USD", "US30/USD", "NAS100/USD", "SPX/USD"
]
INTERVAL = "1h"
OUTPUT_SIZE = 250

# --- INDICATEURS ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma(s, p):
    if s.empty or len(s) < p: return pd.Series([np.nan]*len(s), index=s.index)
    wma = lambda x: np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1))
    wma1 = s.rolling(p//2).apply(wma, raw=True)
    wma2 = s.rolling(p).apply(wma, raw=True)
    diff = 2*wma1 - wma2
    return diff.rolling(int(np.sqrt(p))).apply(wma, raw=True)

def rsi(src, p):
    d = src.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    rs = rma(g, p) / rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

def adx(h, l, c, p):
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    atr = rma(tr, p)
    up = h.diff(); down = l.shift() - l
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    pdi = 100 * rma(pd.Series(plus, index=h.index), p) / atr.replace(0, 1e-9)
    mdi = 100 * rma(pd.Series(minus, index=h.index), p) / atr.replace(0, 1e-9)
    dx = 100 * abs(pdi - mdi) / (pdi + mdi).replace(0, 1e-9)
    return rma(dx, p)

def heiken_ashi(df):
    ha_close = (df[['Open', 'High', 'Low', 'Close']].sum(axis=1)) / 4
    ha_open = pd.Series(index=ha_close.index, dtype=float)
    ha_open.iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    return ha_open, ha_close

def smoothed_heiken_ashi(df, l1=10, l2=10):
    eo, eh, el, ec = ema(df['Open'], l1), ema(df['High'], l1), ema(df['Low'], l1), ema(df['Close'], l1)
    hao, hac = heiken_ashi(pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}))
    return ema(hao, l2), ema(hac, l2)

def ichimoku_signal(h, l, c, tenkan=9, kijun=26, senkou_b=52):
    if len(h) < senkou_b or len(l) < senkou_b or len(c) < senkou_b:
        return 0
    tenkan_sen = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2
    kijun_sen = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2
    senkou_a = (tenkan_sen + kijun_sen) / 2
    senkou_b_val = (h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2
    ccl = c.iloc[-1]
    cssa = senkou_a.iloc[-1]
    cssb = senkou_b_val.iloc[-1]
    ctn = max(cssa, cssb)
    cbn = min(cssa, cssb)
    if ccl > ctn:
        return 1
    elif ccl < cbn:
        return -1
    return 0

# --- FETCH DATA ---
@st.cache_data(ttl=900)
def get_data(symbol):
    try:
        r = requests.get(TWELVE_DATA_API_URL, params={
            "symbol": symbol,
            "interval": INTERVAL,
            "outputsize": OUTPUT_SIZE,
            "apikey": API_KEY,
            "timezone": "UTC"
        })
        j = r.json()
        if "values" not in j:
            return None
        df = pd.DataFrame(j["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        df = df.astype(float)
        df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
        return df[['Open','High','Low','Close']]
    except Exception:
        return None

