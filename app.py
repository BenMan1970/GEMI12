import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import requests

# --- CONFIGURATION ---
API_KEY = "YOUR_API_KEY_HERE"  # Remplacez par votre clé API
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
INTERVAL = "1h"
OUTPUT_SIZE = 100

# Configuration Streamlit
st.set_page_config(
    page_title="Forex Scanner",
    page_icon="📈",
    layout="wide"
)

# --- PAIRES FOREX ---
FOREX_PAIRS_TD = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP",
    "XAU/USD", "US30/USD", "NAS100/USD", "SPX/USD"
]

# --- FONCTIONS UTILITAIRES ---
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def confluence_stars(val):
    stars_dict = {6: "⭐⭐⭐⭐⭐⭐", 5: "⭐⭐⭐⭐⭐", 4: "⭐⭐⭐⭐", 
                  3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}
    return stars_dict.get(val, "WAIT")

# --- RÉCUPÉRATION DES DONNÉES ---
@st.cache_data(ttl=900)
def get_data(symbol):
    try:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "outputsize": OUTPUT_SIZE,
            "apikey": API_KEY,
            "timezone": "UTC"
        }
        
        response = requests.get(TWELVE_DATA_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "values" not in data:
            return None
            
        df = pd.DataFrame(data["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df.isnull().all().all():
            return None
            
        df.rename(columns={
            "open": "Open", "high": "High", 
            "low": "Low", "close": "Close"
        }, inplace=True)
        
        return df[['Open', 'High', 'Low', 'Close']]
        
    except Exception as e:
        st.error(f"Erreur pour {symbol}: {str(e)}")
        return None

# --- INDICATEURS TECHNIQUES ---
def rsi(src, period):
    delta = src.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

def adx(high, low, close, period):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
    
    atr = rma(tr, period)
    plus_di = 100 * rma(plus_dm, period) / atr.replace(0, 1e-9)
    minus_di = 100 * rma(minus_dm, period) / atr.replace(0, 1e-9)
    
    dx = abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1.0)
    return 100 * rma(dx, period)

# --- CALCUL DES SIGNAUX ---
def calculate_signals(df):
    if df is None or len(df) < 60:
        return None
    
    try:
        ohlc4 = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        signals = {}
        bull_count = bear_count = 0

        # HMA (approximation)
        hma = df['Close'].rolling(window=9, min_periods=1).mean()
        if len(hma) >= 2:
            if hma.iloc[-1] > hma.iloc[-2]:
                bull_count += 1
                signals['HMA'] = "▲"
            elif hma.iloc[-1] < hma.iloc[-2]:
                bear_count += 1
                signals['HMA'] = "▼"
            else:
                signals['HMA'] = "—"

        # RSI
        rsi_values = rsi(ohlc4, 10)
        if not rsi_values.empty:
            rsi_val = rsi_values.iloc[-1]
            signals['RSI'] = f"{int(rsi_val)}" if not pd.isna(rsi_val) else "N/A"
            if rsi_val > 50:
                bull_count += 1
            elif rsi_val < 50:
                bear_count += 1

        # ADX
        adx_values = adx(df['High'], df['Low'], df['Close'], 14)
        if not adx_values.empty:
            adx_val = adx_values.iloc[-1]
            signals['ADX'] = f"{int(adx_val)}" if not pd.isna(adx_val) else "N/A"

        # Heikin Ashi
        ha_close = (df[['Open', 'High', 'Low', 'Close']].sum(axis=1)) / 4
        ha_open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
        
        if len(ha_close) >= 1 and len(ha_open) >= 1:
            if ha_close.iloc[-1] > ha_open.iloc[-1]:
                bull_count += 1
                signals['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                bear_count += 1
                signals['HA'] = "▼"
            else:
                signals['HA'] = "—"

        # Smoothed Heikin Ashi
        sha_close = df['Close'].ewm(span=10).mean()
        sha_open = df['Open'].ewm(span=10).mean()
        
        if len(sha_close) >= 1 and len(sha_open) >= 1:
            if sha_close.iloc[-1] > sha_open.iloc[-1]:
                bull_count += 1
                signals['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                bear_count += 1
                signals['SHA'] = "▼"
            else:
                signals['SHA'] = "—"

        # Ichimoku simplifié
        if len(df) >= 52:
            tenkan = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
            kijun = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
            senkou_a = (tenkan + kijun) / 2
            senkou_b = (df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2
            
            current_price = df['Close'].iloc[-1]
            cloud_top = max(senkou_a.iloc[-1], senkou_b.iloc[-1])
            cloud_bottom = min(senkou_a.iloc[-1], senkou_b.iloc[-1])
            
            if current_price > cloud_top:
                bull_count += 1
                signals['Ichimoku'] = "▲"
            elif current_price < cloud_bottom:
                bear_count += 1
                signals['Ichimoku'] = "▼"
            else:
                signals['Ichimoku'] = "—"
        else:
            signals['Ichimoku'] = "N/A"

        confluence = max(bull_count, bear_count)
        if bull_count > bear_count:
            direction = "HAUSSIER"
        elif bear_count > bull_count:
            direction = "BAISSIER"
        else:
            direction = "NEUTRE"
            
        stars = confluence_stars(confluence)

        return {
            "confluence": confluence,
            "direction": direction,
            "stars": stars,
            "signals": signals,
            "bull_count": bull_count,
            "bear_count": bear_count
        }
        
    except Exception as e:
        st.error(f"Erreur calcul signaux: {str(e)}")
        return None

# --- INTERFACE PRINCIPALE ---
def main():
    st.title("📈 Forex Scanner - Analyse de Confluences")
    st.markdown("---")
    
    # Vérification clé API
    if API_KEY == "YOUR_API_KEY_HERE" or not API_KEY:
        st.error("⚠️ Veuillez configurer votre clé API Twelve Data")
        st.info("Modifiez la variable API_KEY dans le code avec votre clé valide")
        st.stop()
    
    # Interface utilisateur
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("⚙️ Paramètres")
        min_conf = st.slider("Confluence minimale", 0, 6, 3)
        show_all = st.checkbox("Afficher toutes les paires", value=False)
        
        st.markdown("---")
        st.info(f"**Intervalle:** {INTERVAL}\n**Données:** {OUTPUT_SIZE} bougies")
        
        # Initialisation du state
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = []
        if 'last_scan' not in st.session_state:
            st.session_state.last_scan = None
        
        # Bouton de scan
        if st.button("🚀 Lancer le scan", type="primary"):
            with st.spinner("Scan en cours..."):
                st.session_state.scan_results = perform_scan(min_conf, show_all)
                st.session_state.last_scan = datetime.now(timezone.utc)
    
    with col2:
        st.header("📊 Résultats")
        
        # Affichage des résultats
        if st.session_state.scan_results:
            display_results(st.session_state.scan_results)
        else:
            st.info("👈 Configurez les paramètres et lancez le scan")
        
        # Timestamp
        if st.session_state.last_scan:
            st.caption(f"Dernière mise à jour : {st.session_state.last_scan.strftime('%Y-%m-%d %H:%M:%S')} UTC")

def perform_scan(min_conf, show_all):
    """Effectue le scan des paires"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(FOREX_PAIRS_TD):
        status_text.write(f"🔍 Analyse: **{symbol}** ({i+1}/{len(FOREX_PAIRS_TD)})")
        
        df = get_data(symbol)
        if df is not None:
            signals_result = calculate_signals(df)
            
            if signals_result and (show_all or signals_result['confluence'] >= min_conf):
                color = 'green' if signals_result['direction'] == 'HAUSSIER' else 'red' if signals_result['direction'] == 'BAISSIER' else 'gray'
                
                row = {
                    "Paire": symbol.replace("/", ""),
                    "Confluences": signals_result['stars'],
                    "Direction": f"<span style='color:{color}; font-weight:bold'>{signals_result['direction']}</span>",
                    "Score": f"{signals_result['bull_count']}/{signals_result['bear_count']}"
                }
                row.update(signals_result['signals'])
                results.append(row)
        
        time.sleep(1.2)  # Respect API limits
        progress_bar.progress((i + 1) / len(FOREX_PAIRS_TD))
    
    progress_bar.empty()
    status_text.empty()
    return results

def display_results(results):
    """Affiche les résultats du scan"""
    if not results:
        st.warning("Aucun résultat trouvé")
        return
    
    st.success(f"🎯 **{len(results)} paires trouvées**")
    
    df_results = pd.DataFrame(results)
    
    # Ordre des colonnes
    base_cols = ["Paire", "Confluences", "Direction", "Score"]
    signal_cols = ["HMA", "RSI", "ADX", "HA", "SHA", "Ichimoku"]
    existing_cols = base_cols + [col for col in signal_cols if col in df_results.columns]
    
    df_results = df_results[existing_cols].sort_values(by="Confluences", ascending=False)
    
    # Affichage
    st.markdown(df_results.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Export CSV
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📂 Exporter CSV",
        data=csv_data,
        file_name=f"confluences_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# Instructions
with st.expander("ℹ️ Instructions"):
    st.markdown("""
    **Utilisation:**
    1. Configurez votre clé API Twelve Data
    2. Ajustez la confluence minimale
    3. Lancez le scan et attendez les résultats
    
    **Indicateurs:**
    - ▲ = Signal haussier
    - ▼ = Signal baissier  
    - — = Signal neutre
    """)

# Point d'entrée
if __name__ == "__main__":
    main()
