import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import time

# =============================================================================
# MODIFICATION DEMANDÉE : Fonction de calcul PRÉCIS de l'ADX
# C'est la seule modification apportée à la logique de calcul.
# =============================================================================

def rma(series: pd.Series, length: int) -> pd.Series:
    """Calcule la Relative Moving Average (RMA) de Wilder."""
    return series.ewm(alpha=1/length, min_periods=length).mean()

def calculate_adx_precise(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14) -> pd.Series:
    """Calcule l'ADX en suivant exactement la logique du script Pine Script."""
    df_ = df.copy()
    up = df_['high'].diff()
    down = -df_['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    true_range = ta.true_range(df_['high'], df_['low'], df_['close'])
    rma_true_range = rma(true_range, di_len)
    rma_plus_dm = rma(pd.Series(plus_dm, index=df_.index), di_len)
    rma_minus_dm = rma(pd.Series(minus_dm, index=df_.index), di_len)
    plus_di = 100 * (rma_plus_dm / rma_true_range)
    minus_di = 100 * (rma_minus_dm / rma_true_range)
    dx_denominator = plus_di + minus_di
    dx = 100 * (np.abs(plus_di - minus_di) / dx_denominator.replace(0, 1))
    adx_value = rma(dx, adx_len)
    return adx_value

# =============================================================================
# LOGIQUE ORIGINALE DES INDICATEURS ET DES CONFLUENCES (INCHANGÉE)
# =============================================================================

def calculate_all_indicators(df: pd.DataFrame):
    params = {"hmaLength": 20, "adxThreshold": 20, "rsiLength": 10, "adxLength": 14, "diLength": 14, "ichimokuLength": 9, "smoothedHaLen1": 10, "smoothedHaLen2": 10}

    # HMA
    df['hma'] = ta.hma(df['close'], length=params['hmaLength'])
    df['hmaSlope'] = np.where(df['hma'] > df['hma'].shift(1), 1, -1)

    # Heikin Ashi et Smoothed Heikin Ashi (logique originale, plus simple)
    ha_df = ta.ha(df['open'], df['high'], df['low'], df['close'])
    df['haSignal'] = np.where(ha_df['HA_close'] > ha_df['HA_open'], 1, -1)
    
    smoothed_ha_df = ta.ha(df['open'].ewm(span=params['smoothedHaLen1'], adjust=False).mean(), df['high'].ewm(span=params['smoothedHaLen1'], adjust=False).mean(), df['low'].ewm(span=params['smoothedHaLen1'], adjust=False).mean(), df['close'].ewm(span=params['smoothedHaLen1'], adjust=False).mean())
    o2 = smoothed_ha_df['HA_open'].ewm(span=params['smoothedHaLen2'], adjust=False).mean()
    c2 = smoothed_ha_df['HA_close'].ewm(span=params['smoothedHaLen2'], adjust=False).mean()
    df['smoothedHaSignal'] = np.where(o2 > c2, -1, 1)

    # RSI (sur hlc4)
    hlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['rsi'] = ta.rsi(hlc4, length=params['rsiLength'])
    df['rsiSignal'] = np.where(df['rsi'] > 50, 1, -1)

    # --- LIGNE MODIFIÉE ---
    # On utilise la nouvelle fonction de calcul précis pour l'ADX.
    df['adx'] = calculate_adx_precise(df, di_len=params['diLength'], adx_len=params['adxLength'])
    df['adxHasMomentum'] = df['adx'] >= params['adxThreshold']

    # Ichimoku (logique originale)
    ichimoku_df, _ = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=params['ichimokuLength'], kijun=26, senkou=52)
    if ichimoku_df is not None and not ichimoku_df.empty:
        cloud_top = ichimoku_df.iloc[:,0:2].max(axis=1) # Utilise l'index de colonne, plus robuste
        cloud_bottom = ichimoku_df.iloc[:,0:2].min(axis=1)
        df['ichimokuSignal'] = np.select([df['close'] > cloud_top, df['close'] < cloud_bottom], [1, -1], default=0)
    else:
        df['ichimokuSignal'] = 0

    # Calcul des Confluences
    bull_conditions = [df['hmaSlope'] == 1, df['haSignal'] == 1, df['smoothedHaSignal'] == 1, df['rsiSignal'] == 1, df['adxHasMomentum'], df['ichimokuSignal'] == 1]
    df['bullConfluences'] = np.sum(bull_conditions, axis=0)
    bear_conditions = [df['hmaSlope'] == -1, df['haSignal'] == -1, df['smoothedHaSignal'] == -1, df['rsiSignal'] == -1, df['adxHasMomentum'], df['ichimokuSignal'] == -1]
    df['bearConfluences'] = np.sum(bear_conditions, axis=0)
    df['confluence'] = df[['bullConfluences', 'bearConfluences']].max(axis=1)
    return df

# =============================================================================
# INTERFACE STREAMLIT (Structure originale, INCHANGÉE)
# =============================================================================

st.set_page_config(layout="wide")
st.title("Canadian Confluence Scanner (Twelve Data API)")

api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")

forex_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD', 'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADJPY', 'CADCHF', 'CHFJPY', 'NZDJPY', 'NZDCAD', 'NZDCHF']
timeframes = {"15min": "15min", "30min": "30min", "1h": "1h", "4h": "4h", "1day": "1day"}

selected_tf = st.selectbox("Sélectionnez le Timeframe", list(timeframes.keys()))

def get_twelve_data(symbol, interval, api_key):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=200&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data.get('status') == 'ok':
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.astype(float)
        return df.iloc[::-1]
    else:
        st.warning(f"Pas de données pour {symbol} : {data.get('message', 'Erreur inconnue')}")
        return None

if st.button("Lancer le Scan"):
    if not api_key:
        st.error("Clé API Twelve Data non configurée.")
    else:
        all_results = []
        progress_bar = st.progress(0)
        
        for i, pair in enumerate(forex_pairs):
            df_ohlc = get_twelve_data(pair, timeframes[selected_tf], api_key)
            
            if df_ohlc is not None and not df_ohlc.empty:
                df_final = calculate_all_indicators(df_ohlc.copy())
                last_row = df_final.iloc[-1].copy()
                last_row['Pair'] = pair
                all_results.append(last_row)
            
            progress_bar.progress((i + 1) / len(forex_pairs))
            
            # Pause de 8 secondes pour respecter la limite de l'API (8 appels/minute)
            time.sleep(8) 

        if all_results:
            results_df = pd.DataFrame(all_results)
            def get_star_rating(c): return "⭐" * int(c) if c > 0 else "WAIT"
            def get_signal_char(s): return "▲" if s == 1 else ("▼" if s == -1 else "─")
            display_df = pd.DataFrame()
            display_df['Paire'] = results_df['Pair']
            display_df['Note'] = results_df['confluence'].apply(get_star_rating)
            display_df['ADX'] = results_df.apply(lambda row: f"✔ ({row['adx']:.1f})" if row['adxHasMomentum'] else f"✖ ({row['adx']:.1f})", axis=1)
            display_df['RSI'] = results_df['rsiSignal'].apply(get_signal_char)
            display_df['Ichi'] = results_df['ichimokuSignal'].apply(get_signal_char)
            display_df['HMA'] = results_df['hmaSlope'].apply(get_signal_char)
            display_df['HA'] = results_df['haSignal'].apply(get_signal_char)
            display_df['HA+'] = results_df['smoothedHaSignal'].apply(get_signal_char)
            st.dataframe(display_df, use_container_width=True)
         
