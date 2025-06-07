import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import requests

# --- CONFIGURATION (LISANT LES SECRETS ET PARAMÈTRES) ---
try:
    # MODIFICATION : On lit directement la variable depuis les secrets.
    API_KEY = st.secrets["TWELVEDATA_API_KEY"]
except (KeyError, FileNotFoundError):
    # Le message d'erreur est maintenant plus précis.
    st.error("Secret 'TWELVEDATA_API_KEY' non trouvé. Veuillez le configurer dans les paramètres de votre application.")
    st.stop()

TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
INTERVAL = "1h" 
OUTPUT_SIZE = 100 

# --- FETCH DATA ---
@st.cache_data(ttl=900)
def get_data(symbol):
    """Récupère les données d'une paire depuis l'API Twelve Data."""
    try:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "outputsize": OUTPUT_SIZE,
            "apikey": API_KEY,
            "timezone": "UTC"
        }
        r = requests.get(TWELVE_DATA_API_URL, params=params)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "error":
            st.error(f"Erreur API pour {symbol}: {j.get('message', 'Format de réponse inconnu.')}")
            return None
        df = pd.DataFrame(j["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        df = df.astype(float)
        df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
        return df[['Open','High','Low','Close']]
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion pour {symbol}: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors de la récupération des données pour {symbol}: {e}")
        return None

# --- PAIRS ---
FOREX_PAIRS_TD = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP", "XAU/USD"
]

# --- INDICATEURS ---
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def rsi(src, p):
    d = src.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    rs = rma(g, p) / rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

def adx(h, l, c, p):
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
    atr = rma(tr, p)
    up = h.diff(); down = -l.diff()
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    pdi = 100 * rma(pd.Series(plus, index=h.index), p) / atr.replace(0, 1e-9)
    mdi = 100 * rma(pd.Series(minus, index=h.index), p) / atr.replace(0, 1e-9)
    dx = 100 * abs(pdi - mdi) / (pdi + mdi).replace(0, 1e-9)
    return rma(dx, p)

# --- SIGNALS ---
def confluence_stars(val):
    return "⭐" * val if 1 <= val <= 6 else "WAIT"

def calculate_signals(df):
    if df is None or len(df) < 60:
        return None
        
    ohlc4 = df[['Open','High','Low','Close']].mean(axis=1)
    signals = {}
    bull = bear = 0

    sma9 = df['Close'].rolling(9).mean()
    if sma9.iloc[-1] > sma9.iloc[-2]: bull += 1; signals['SMA(9)'] = "▲"
    elif sma9.iloc[-1] < sma9.iloc[-2]: bear += 1; signals['SMA(9)'] = "▼"
    else: signals['SMA(9)'] = "—"

    rsi_val = rsi(ohlc4, 10).iloc[-1]
    signals['RSI(10)'] = f"{int(rsi_val)}"
    if rsi_val > 50: bull += 1
    elif rsi_val < 50: bear += 1

    adx_val = adx(df['High'], df['Low'], df['Close'], 14).iloc[-1]
    if adx_val >= 20:
        signals['ADX(14)'] = f"{int(adx_val)} 💪"
        bull += 1
        bear += 1
    else:
        signals['ADX(14)'] = f"{int(adx_val)} 💤"

    ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha_close = (df[['Open','High','Low','Close']].sum(axis=1)) / 4
    if ha_close.iloc[-1] > ha_open.iloc[-1]: bull += 1; signals['Heikin Ashi'] = "▲"
    elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear += 1; signals['Heikin Ashi'] = "▼"
    else: signals['Heikin Ashi'] = "—"
    
    ema10_close = df['Close'].ewm(span=10, adjust=False).mean()
    ema10_open = df['Open'].ewm(span=10, adjust=False).mean()
    if ema10_close.iloc[-1] > ema10_open.iloc[-1]: bull += 1; signals['EMA Cross'] = "▲"
    elif ema10_close.iloc[-1] < ema10_open.iloc[-1]: bear += 1; signals['EMA Cross'] = "▼"
    else: signals['EMA Cross'] = "—"

    tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    price = df['Close'].iloc[-1]
    
    if price > senkou_a.iloc[-1] and price > senkou_b.iloc[-1]:
        bull += 1; signals['Ichimoku'] = "▲"
    elif price < senkou_a.iloc[-1] and price < senkou_b.iloc[-1]:
        bear += 1; signals['Ichimoku'] = "▼"
    else:
        signals['Ichimoku'] = "—"

    confluence = max(bull, bear)
    direction = "HAUSSIER" if bull > bear else "BAISSIER" if bear > bull else "NEUTRE"
    stars = confluence_stars(confluence)

    return {"confluence": confluence, "direction": direction, "stars": stars, "signals": signals}

# --- INTERFACE UTILISATEUR ---
st.set_page_config(layout="wide")
st.title("Scanner de Confluence Forex")

st.sidebar.header("Paramètres")
min_conf = st.sidebar.slider("Confluence minimale", 0, 6, 3)
show_all = st.sidebar.checkbox("Afficher toutes les paires", value=False)

if st.sidebar.button("Lancer le scan"):
    results = []
    total_pairs = len(FOREX_PAIRS_TD)
    progress_bar = st.progress(0, text="Lancement du scan...")

    for i, symbol in enumerate(FOREX_PAIRS_TD):
        progress_text = f"Scan en cours... {symbol} ({i+1}/{total_pairs})"
        progress_bar.progress((i + 1) / total_pairs, text=progress_text)
        
        df = get_data(symbol)
        
        time.sleep(8) 
        
        if df is not None:
            res = calculate_signals(df)
            if res:
                if show_all or res['confluence'] >= min_conf:
                    color = 'green' if res['direction'] == 'HAUSSIER' else 'red' if res['direction'] == 'BAISSIER' else '#888888'
                    row = {
                        "Paire": symbol,
                        "Confluences": res['stars'],
                        "Direction": f"<span style='color:{color}; font-weight:bold;'>{res['direction']}</span>",
                        "confluence_score": res['confluence']
                    }
                    row.update(res['signals'])
                    results.append(row)
    
    progress_bar.empty()

    if results:
        df_res = pd.DataFrame(results).sort_values(by="confluence_score", ascending=False)
        df_display = df_res.drop(columns=['confluence_score']) 
        
        st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📂 Exporter en CSV",
            data=csv,
            file_name="resultats_confluence.csv",
            mime="text/csv",
        )
    else:
        st.warning("Aucun résultat correspondant aux critères. Essayez de baisser la confluence minimale ou de cocher 'Afficher toutes les paires'.")

st.caption(f"Données basées sur l'intervalle {INTERVAL}. Dernière mise à jour : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
