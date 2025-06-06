import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import requests

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

# --- PAIRS ---
FOREX_PAIRS_TD = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP",
    "XAU/USD", "US30/USD", "NAS100/USD", "SPX/USD"
]

# --- INDICATEURS ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

# --- SIGNALS ---
def confluence_stars(val):
    if val == 6: return "⭐⭐⭐⭐⭐⭐"
    elif val == 5: return "⭐⭐⭐⭐⭐"
    elif val == 4: return "⭐⭐⭐⭐"
    elif val == 3: return "⭐⭐⭐"
    elif val == 2: return "⭐⭐"
    elif val == 1: return "⭐"
    else: return "WAIT"

def calculate_signals(df):
    if df is None or len(df) < 60:
        return None
    ohlc4 = df[['Open','High','Low','Close']].mean(axis=1)
    signals = {}
    bull = bear = 0

    hma = df['Close'].rolling(9).mean()  # approximation de HMA
    if hma.iloc[-1] > hma.iloc[-2]: bull += 1; signals['HMA'] = "▲"
    elif hma.iloc[-1] < hma.iloc[-2]: bear += 1; signals['HMA'] = "▼"

    rsi_val = rsi(ohlc4, 10).iloc[-1]
    signals['RSI'] = f"{int(rsi_val)}"
    if rsi_val > 50: bull += 1
    elif rsi_val < 50: bear += 1

    adx_val = adx(df['High'], df['Low'], df['Close'], 14).iloc[-1]
    signals['ADX'] = f"{int(adx_val)}"
    if adx_val >= 20: bull += 1; bear += 1

    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha_close = (df[['Open','High','Low','Close']].sum(axis=1)) / 4
    if ha_close.iloc[-1] > ha_open.iloc[-1]: bull += 1; signals['HA'] = "▲"
    elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear += 1; signals['HA'] = "▼"

    sha = df['Close'].ewm(span=10).mean()
    sha_open = df['Open'].ewm(span=10).mean()
    if sha.iloc[-1] > sha_open.iloc[-1]: bull += 1; signals['SHA'] = "▲"
    elif sha.iloc[-1] < sha_open.iloc[-1]: bear += 1; signals['SHA'] = "▼"

    # Ichimoku simplifié
    tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2
    price = df['Close'].iloc[-1]
    ichi_signal = 1 if price > max(senkou_a.iloc[-1], senkou_b.iloc[-1]) else -1 if price < min(senkou_a.iloc[-1], senkou_b.iloc[-1]) else 0
    if ichi_signal == 1: bull += 1; signals['Ichimoku'] = "▲"
    elif ichi_signal == -1: bear += 1; signals['Ichimoku'] = "▼"
    else: signals['Ichimoku'] = "—"

    confluence = max(bull, bear)
    direction = "HAUSSIER" if bull > bear else "BAISSIER" if bear > bull else "NEUTRE"
    stars = confluence_stars(confluence)

    return {"confluence": confluence, "direction": direction, "stars": stars, "signals": signals}

def rsi(src, p):
    d = src.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    rs = rma(g, p) / rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

def adx(h, l, c, p):
    # This function implements the ADX calculation based on the provided TradingView script.
    # Pine Script logic: adx = 100 * ta.rma(math.abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)
    # The length 'p' is used for both DI Length (dilen) and ADX Smoothing (adxlen).

    # Calculate True Range (TR)
    tr = pd.concat([h - l, abs(h - c.shift()), abs(l - c.shift())], axis=1).max(axis=1)
    rma_tr = rma(tr, p)

    # Calculate Directional Movement (+DM, -DM)
    up = h.diff()
    down = -l.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)

    # Calculate Directional Indicators (+DI, -DI)
    safe_rma_tr = rma_tr.replace(0, 1e-9)
    pdi = 100 * rma(plus_dm, p) / safe_rma_tr
    mdi = 100 * rma(minus_dm, p) / safe_rma_tr

    # Calculate the unscaled Directional Index (term inside the RMA in Pine Script)
    di_sum = (pdi + mdi).replace(0, 1.0) # To avoid division by zero, mimicking (sum == 0 ? 1 : sum)
    dx_unscaled = abs(pdi - mdi) / di_sum

    # Calculate ADX by smoothing the unscaled DX with RMA and then multiplying by 100
    adx_val = 100 * rma(dx_unscaled, p)
    
    return adx_val

# --- INTERFACE UTILISATEUR ---
st.sidebar.header("Paramètres")
min_conf = st.sidebar.slider("Confluence minimale", 0, 6, 3)
show_all = st.sidebar.checkbox("Afficher toutes les paires", value=False)

# This part of the code requires your API key and settings. 
# Make sure to define them before running the script.
# For example:
# API_KEY = "YOUR_API_KEY"
# TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
# INTERVAL = "1h"
# OUTPUT_SIZE = 100

if st.sidebar.button("Lancer le scan"):
    # Check if API_KEY is defined
    if 'API_KEY' not in locals() or not API_KEY:
        st.error("Veuillez définir votre `API_KEY` dans le script.")
    else:
        results = []
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        for i, symbol in enumerate(FOREX_PAIRS_TD):
            status_text.write(f"Scan en cours: {symbol} ({i+1}/{len(FOREX_PAIRS_TD)})")
            df = get_data(symbol)
            time.sleep(1.0) # Respect API rate limits
            res = calculate_signals(df)
            if res:
                if show_all or res['confluence'] >= min_conf:
                    color = 'green' if res['direction'] == 'HAUSSIER' else 'red' if res['direction'] == 'BAISSIER' else 'gray'
                    row = {
                        "Paire": symbol.replace("/", ""),
                        "Confluences": res['stars'],
                        "Direction": f"<span style='color:{color}'>{res['direction']}</span>",
                    }
                    row.update(res['signals'])
                    results.append(row)
            progress_bar.progress((i + 1) / len(FOREX_PAIRS_TD))
        
        status_text.write("Scan terminé !")
        progress_bar.empty()

        if results:
            df_res = pd.DataFrame(results)
            # Ensure specific column order
            cols = ["Paire", "Confluences", "Direction", "HMA", "RSI", "ADX", "HA", "SHA", "Ichimoku"]
            # Filter for columns that actually exist in the dataframe to avoid errors
            existing_cols = [col for col in cols if col in df_res.columns]
            df_res = df_res[existing_cols]

            df_res = df_res.sort_values(by="Confluences", ascending=False)
            st.markdown(df_res.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.download_button("📂 Exporter CSV", data=df_res.to_csv(index=False).encode('utf-8'), file_name="confluences.csv", mime="text/csv")
        else:
            st.warning("Aucun résultat correspondant aux critères.")

st.caption(f"Dernière mise à jour : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
