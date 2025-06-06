import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests # Pour faire les appels à l'API Twelve Data

# =============================================================================
# FONCTIONS DE CALCUL (Équivalents Pine Script, Précis et Corrigés)
# =============================================================================

def rma(series: pd.Series, length: int) -> pd.Series:
    """Calcule la Relative Moving Average (RMA) de Wilder."""
    return series.ewm(alpha=1/length, min_periods=length).mean()

def calculate_adx(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14) -> pd.Series:
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

def calculate_heikin_ashi_simple(df: pd.DataFrame):
    """Calcule le signal Heikin Ashi simple."""
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = pd.Series(np.nan, index=df.index)
    if not df.empty:
        ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    return ha_close, ha_open

def calculate_smoothed_heikin_ashi(df: pd.DataFrame, len1: int = 10, len2: int = 10):
    """Calcule le signal Heikin Ashi lissé (+)."""
    o1 = df['open'].ewm(span=len1, adjust=False).mean()
    c1 = df['close'].ewm(span=len1, adjust=False).mean()
    h1 = df['high'].ewm(span=len1, adjust=False).mean()
    l1 = df['low'].ewm(span=len1, adjust=False).mean()
    haclose1 = (o1 + h1 + l1 + c1) / 4
    haopen1 = pd.Series(np.nan, index=df.index)
    if not df.empty:
        haopen1.iloc[0] = (o1.iloc[0] + c1.iloc[0]) / 2
        for i in range(1, len(df)):
            haopen1.iloc[i] = (haopen1.iloc[i-1] + haclose1.iloc[i-1]) / 2
    o2 = haopen1.ewm(span=len2, adjust=False).mean()
    c2 = haclose1.ewm(span=len2, adjust=False).mean()
    return o2, c2

def calculate_all_indicators(df: pd.DataFrame):
    """Fonction principale qui calcule tous les indicateurs et les signaux."""
    params = {
        "hmaLength": 20, "adxThreshold": 20, "rsiLength": 10,
        "adxLength": 14, "diLength": 14, "ichimokuLength": 9,
        "smoothedHaLen1": 10, "smoothedHaLen2": 10,
    }

    df['hma'] = ta.hma(df['close'], length=params['hmaLength'])
    df['hmaSlope'] = np.where(df['hma'] > df['hma'].shift(1), 1, -1)
    
    ha_close, ha_open = calculate_heikin_ashi_simple(df)
    df['haSignal'] = np.where(ha_close > ha_open, 1, -1)

    o2, c2 = calculate_smoothed_heikin_ashi(df, params['smoothedHaLen1'], params['smoothedHaLen2'])
    df['smoothedHaSignal'] = np.where(o2 > c2, -1, 1)

    hlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['rsi'] = ta.rsi(hlc4, length=params['rsiLength'])
    df['rsiSignal'] = np.where(df['rsi'] > 50, 1, -1)

    df['adx'] = calculate_adx(df, di_len=params['diLength'], adx_len=params['adxLength'])
    df['adxHasMomentum'] = df['adx'] >= params['adxThreshold']

    # --- BLOC ICHIMOKU CORRIGÉ ET ROBUSTE ---
    ichimoku_df, _ = ta.ichimoku(df['high'], df['low'], df['close'], 
                                tenkan=params['ichimokuLength'], kijun=26, senkou=52)
    
    if ichimoku_df is not None and not ichimoku_df.empty:
        ichimoku_df.columns = ['senkouA', 'senkouB', 'tenkan', 'kijun', 'chikou']
        df['tenkan'] = ichimoku_df['tenkan']
        df['kijun'] = ichimoku_df['kijun']
        df['senkouA'] = ichimoku_df['senkouA']
        df['senkouB'] = ichimoku_df['senkouB']
        cloud_top = df[['senkouA', 'senkouB']].max(axis=1)
        cloud_bottom = df[['senkouA', 'senkouB']].min(axis=1)
        df['ichimokuSignal'] = np.select(
            [df['close'] > cloud_top, df['close'] < cloud_bottom], [1, -1], default=0)
    else:
        df['ichimokuSignal'] = 0

    # --- CALCUL DES CONFLUENCES ---
    bull_conditions = [
        df['hmaSlope'] == 1, df['haSignal'] == 1, df['smoothedHaSignal'] == 1,
        df['rsiSignal'] == 1, df['adxHasMomentum'], df['ichimokuSignal'] == 1
    ]
    df['bullConfluences'] = np.sum(bull_conditions, axis=0)
    bear_conditions = [
        df['hmaSlope'] == -1, df['haSignal'] == -1, df['smoothedHaSignal'] == -1,
        df['rsiSignal'] == -1, df['adxHasMomentum'], df['ichimokuSignal'] == -1
    ]
    df['bearConfluences'] = np.sum(bear_conditions, axis=0)
    df['confluence'] = df[['bullConfluences', 'bearConfluences']].max(axis=1)
    return df

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(layout="wide")
st.title("Canadian Confluence Scanner (Twelve Data API)")

# Clé API - À stocker de manière sécurisée avec les Secrets de Streamlit
api_key = st.secrets.get("TWELVE_DATA_API_KEY", "VOTRE_API_KEY_PAR_DEFAUT_ICI")

# Listes des actifs
forex_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/AUD', 'EUR/CAD', 'EUR/CHF', 'EUR/NZD', 'GBP/JPY', 'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/NZD', 'AUD/JPY', 'AUD/CAD', 'AUD/CHF', 'AUD/NZD', 'CAD/JPY', 'CAD/CHF', 'CHF/JPY', 'NZD/JPY', 'NZD/CAD', 'NZD/CHF']
timeframes = {"15min": "15min", "30min": "30min", "1h": "1h", "4h": "4h", "1day": "1day"}

selected_tf = st.selectbox("Sélectionnez le Timeframe", list(timeframes.keys()))

def get_twelve_data(symbol, interval, api_key):
    """Récupère les données de l'API Twelve Data."""
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=200&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.astype(float)
        # L'API retourne les données du plus récent au plus ancien, il faut les inverser.
        return df.iloc[::-1]
    return None

if st.button("Lancer le Scan"):
    all_results = []
    progress_bar = st.progress(0)
    
    for i, pair in enumerate(forex_pairs):
        try:
            df_ohlc = get_twelve_data(pair, timeframes[selected_tf], api_key)
            if df_ohlc is None or df_ohlc.empty:
                st.warning(f"Pas de données pour {pair} sur le timeframe {selected_tf}")
                continue
            
            df_final = calculate_all_indicators(df_ohlc.copy())
            last_row = df_final.iloc[-1].copy()
            last_row['Pair'] = pair
            all_results.append(last_row)
        except Exception as e:
            st.error(f"Erreur lors du traitement de la paire {pair}: {e}")
        
        progress_bar.progress((i + 1) / len(forex_pairs))

    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Fonctions pour l'affichage
        def get_star_rating(c): return "⭐" * int(c) if c > 0 else "WAIT"
        def get_signal_char(s): return "▲" if s == 1 else ("▼" if s == -1 else "─")

        # Mise en forme du tableau final
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
    else:
        st.info("Aucun résultat n'a pu être généré. Vérifiez votre clé API ou la disponibilité des données.")
