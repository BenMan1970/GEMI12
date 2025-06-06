import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf

# =============================================================================
# FONCTIONS DE CALCUL (Équivalents Pine Script)
# =============================================================================

def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Calcule la Relative Moving Average (RMA) de Wilder.
    Équivalent exact de ta.rma() en Pine Script.
    """
    return series.ewm(alpha=1/length, min_periods=length).mean()

def calculate_adx(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14) -> pd.Series:
    """
    Calcule l'ADX en suivant exactement la logique du script Pine Script.
    Équivalent de la fonction f_adx()
    """
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
    """
    Calcule le signal Heikin Ashi simple comme dans le script Pine.
    """
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = pd.Series(np.nan, index=df.index)
    if not df.empty:
        ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    return ha_close, ha_open

def calculate_smoothed_heikin_ashi(df: pd.DataFrame, len1: int = 10, len2: int = 10):
    """
    Calcule le signal Heikin Ashi lissé (+) comme dans le script Pine.
    """
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
    """
    Fonction principale qui calcule tous les indicateurs et les signaux.
    """
    params = {
        "hmaLength": 20, "adxThreshold": 20, "rsiLength": 10,
        "adxLength": 14, "diLength": 14, "ichimokuLength": 9,
        "smoothedHaLen1": 10, "smoothedHaLen2": 10,
    }

    # HMA
    df['hma'] = ta.hma(df['close'], length=params['hmaLength'])
    df['hmaSlope'] = np.where(df['hma'] > df['hma'].shift(1), 1, -1)
    
    # Heikin Ashi Simple
    ha_close, ha_open = calculate_heikin_ashi_simple(df)
    df['haSignal'] = np.where(ha_close > ha_open, 1, -1)

    # Smoothed Heikin Ashi
    o2, c2 = calculate_smoothed_heikin_ashi(df, params['smoothedHaLen1'], params['smoothedHaLen2'])
    df['smoothedHaSignal'] = np.where(o2 > c2, -1, 1)

    # RSI (sur hlc4)
    hlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['rsi'] = ta.rsi(hlc4, length=params['rsiLength'])
    df['rsiSignal'] = np.where(df['rsi'] > 50, 1, -1)

    # ADX (avec la fonction de calcul précis)
    df['adx'] = calculate_adx(df, di_len=params['diLength'], adx_len=params['adxLength'])
    df['adxHasMomentum'] = df['adx'] >= params['adxThreshold']

    # --- BLOC ICHIMOKU CORRIGÉ ET ROBUSTE ---
    ichimoku_df, _ = ta.ichimoku(df['high'], df['low'], df['close'], 
                                tenkan=params['ichimokuLength'], kijun=26, senkou=52)
    
    if ichimoku_df is not None:
        ichimoku_df.columns = ['senkouA', 'senkouB', 'tenkan', 'kijun', 'chikou']
        df['tenkan'] = ichimoku_df['tenkan']
        df['kijun'] = ichimoku_df['kijun']
        df['senkouA'] = ichimoku_df['senkouA']
        df['senkouB'] = ichimoku_df['senkouB']
        cloud_top = df[['senkouA', 'senkouB']].max(axis=1)
        cloud_bottom = df[['senkouA', 'senkouB']].min(axis=1)
        df['ichimokuSignal'] = np.select(
            [df['close'] > cloud_top, df['close'] < cloud_bottom],
            [1, -1],
            default=0
        )
    else: # Fallback si ichimoku ne retourne rien
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

def get_star_rating(confluence):
    return "⭐" * int(confluence) if confluence > 0 else "WAIT"

def get_rating_color(confluence):
    if confluence == 6: return "green"
    if confluence == 5: return "lime"
    if confluence == 4: return "yellow"
    if confluence == 3: return "orange"
    if confluence == 2: return "red"
    return "gray"

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(layout="wide")
st.title("Canadian Confluence Scanner")

# Liste des paires de devises
forex_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X', 'NZDUSD=X',
    'EURGBP=X', 'EURJPY=X', 'EURAUD=X', 'EURCAD=X', 'EURCHF=X', 'EURNZD=X',
    'GBPJPY=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPCHF=X', 'GBPNZD=X',
    'AUDJPY=X', 'AUDCAD=X', 'AUDCHF=X', 'AUDNZD=X',
    'CADJPY=X', 'CADCHF=X',
    'CHFJPY=X',
    'NZDJPY=X', 'NZDCAD=X', 'NZDCHF=X'
]

timeframes = {"15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}

selected_tf = st.selectbox("Sélectionnez le Timeframe", list(timeframes.keys()))

if st.button("Lancer le Scan"):
    results = []
    progress_bar = st.progress(0)
    
    for i, pair in enumerate(forex_pairs):
        try:
            # Télécharger les données (on prend plus que nécessaire pour stabiliser les indicateurs)
            df_ohlc = yf.download(pair, period="3mo", interval=timeframes[selected_tf], progress=False)
            
            if df_ohlc.empty:
                print(f"Pas de données pour {pair} sur le timeframe {selected_tf}")
                continue

            # Calculer les indicateurs
            df_final = calculate_all_indicators(df_ohlc.copy())
            
            # Récupérer la dernière ligne pour le résultat
            last_row = df_final.iloc[-1]
            results.append(last_row)

        except Exception as e:
            st.warning(f"Impossible de traiter la paire {pair}: {e}")
        
        progress_bar.progress((i + 1) / len(forex_pairs))

    if results:
        results_df = pd.DataFrame(results)
        results_df['Pair'] = [p.replace('=X', '') for p in forex_pairs if any(p in str(r.name) for r in results)] # Assurer la correspondance
        
        # Sélection des colonnes et mise en forme pour l'affichage
        display_df = pd.DataFrame()
        display_df['Paire'] = results_df['Pair']
        display_df['Note'] = results_df['confluence'].apply(get_star_rating)
        
        # Création des colonnes de signaux
        for name, key in [
            ("ADX", "adxHasMomentum"), ("RSI", "rsiSignal"), ("Ichi", "ichimokuSignal"),
            ("HMA", "hmaSlope"), ("HA", "haSignal"), ("HA+", "smoothedHaSignal")
        ]:
            if name == "ADX":
                display_df[name] = results_df.apply(lambda row: f"✔ ({row['adx']:.1f})" if row[key] else f"✖ ({row['adx']:.1f})", axis=1)
            else:
                display_df[name] = results_df[key].apply(lambda x: "▲" if x == 1 else ("▼" if x == -1 else "─"))
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.write("Aucun résultat n'a pu être généré.")
