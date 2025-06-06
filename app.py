import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import time # Librairie pour ajouter des pauses

# =============================================================================
# FONCTIONS DE CALCUL (Inchangées, elles sont correctes)
# =============================================================================
# ... (Les fonctions de calcul de rma, adx, heikin ashi, etc. sont ici)
# Pour ne pas surcharger la réponse, je ne les répète pas, mais elles doivent être
# présentes dans votre fichier final, comme dans la version précédente que je vous ai donnée.

def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1/length, min_periods=length).mean()

def calculate_adx(df: pd.DataFrame, di_len: int = 14, adx_len: int = 14) -> pd.Series:
    df_ = df.copy(); up = df_['high'].diff(); down = -df_['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    true_range = ta.true_range(df_['high'], df_['low'], df_['close'])
    rma_true_range = rma(true_range, di_len)
    rma_plus_dm = rma(pd.Series(plus_dm, index=df_.index), di_len)
    rma_minus_dm = rma(pd.Series(minus_dm, index=df_.index), di_len)
    plus_di = 100 * (rma_plus_dm / rma_true_range); minus_di = 100 * (rma_minus_dm / rma_true_range)
    dx_denominator = plus_di + minus_di
    dx = 100 * (np.abs(plus_di - minus_di) / dx_denominator.replace(0, 1))
    return rma(dx, adx_len)

def calculate_heikin_ashi_simple(df: pd.DataFrame):
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = pd.Series(np.nan, index=df.index)
    if not df.empty:
        ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        for i in range(1, len(df)): ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    return ha_close, ha_open

def calculate_smoothed_heikin_ashi(df: pd.DataFrame, len1: int = 10, len2: int = 10):
    o1 = df['open'].ewm(span=len1, adjust=False).mean(); c1 = df['close'].ewm(span=len1, adjust=False).mean()
    h1 = df['high'].ewm(span=len1, adjust=False).mean(); l1 = df['low'].ewm(span=len1, adjust=False).mean()
    haclose1 = (o1 + h1 + l1 + c1) / 4
    haopen1 = pd.Series(np.nan, index=df.index)
    if not df.empty:
        haopen1.iloc[0] = (o1.iloc[0] + c1.iloc[0]) / 2
        for i in range(1, len(df)): haopen1.iloc[i] = (haopen1.iloc[i-1] + haclose1.iloc[i-1]) / 2
    o2 = haopen1.ewm(span=len2, adjust=False).mean(); c2 = haclose1.ewm(span=len2, adjust=False).mean()
    return o2, c2

def calculate_all_indicators(df: pd.DataFrame):
    params = {"hmaLength": 20, "adxThreshold": 20, "rsiLength": 10, "adxLength": 14, "diLength": 14, "ichimokuLength": 9, "smoothedHaLen1": 10, "smoothedHaLen2": 10}
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
    ichimoku_df, _ = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=params['ichimokuLength'], kijun=26, senkou=52)
    if ichimoku_df is not None and not ichimoku_df.empty:
        ichimoku_df.columns = ['senkouA', 'senkouB', 'tenkan', 'kijun', 'chikou']
        df['tenkan'], df['kijun'], df['senkouA'], df['senkouB'] = ichimoku_df['tenkan'], ichimoku_df['kijun'], ichimoku_df['senkouA'], ichimoku_df['senkouB']
        cloud_top = df[['senkouA', 'senkouB']].max(axis=1); cloud_bottom = df[['senkouA', 'senkouB']].min(axis=1)
        df['ichimokuSignal'] = np.select([df['close'] > cloud_top, df['close'] < cloud_bottom], [1, -1], default=0)
    else: df['ichimokuSignal'] = 0
    bull_conditions = [df['hmaSlope'] == 1, df['haSignal'] == 1, df['smoothedHaSignal'] == 1, df['rsiSignal'] == 1, df['adxHasMomentum'], df['ichimokuSignal'] == 1]
    df['bullConfluences'] = np.sum(bull_conditions, axis=0)
    bear_conditions = [df['hmaSlope'] == -1, df['haSignal'] == -1, df['smoothedHaSignal'] == -1, df['rsiSignal'] == -1, df['adxHasMomentum'], df['ichimokuSignal'] == -1]
    df['bearConfluences'] = np.sum(bear_conditions, axis=0)
    df['confluence'] = df[['bullConfluences', 'bearConfluences']].max(axis=1)
    return df

# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(layout="wide")
st.title("Canadian Confluence Scanner (Twelve Data API)")

api_key = st.secrets.get("TWELVE_DATA_API_KEY", "")

forex_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURNZD', 'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPNZD', 'AUDJPY', 'AUDCAD', 'AUDCHF', 'AUDNZD', 'CADJPY', 'CADCHF', 'CHFJPY', 'NZDJPY', 'NZDCAD', 'NZDCHF']
timeframes = {"15min": "15min", "30min": "30min", "1h": "1h", "4h": "4h", "1day": "1day"}

selected_tf = st.selectbox("Sélectionnez le Timeframe", list(timeframes.keys()))

def get_twelve_data(symbol, interval, api_key):
    symbol_with_slash = f"{symbol[:3]}/{symbol[3:]}"
    url = f"https://api.twelvedata.com/time_series?symbol={symbol_with_slash}&interval={interval}&outputsize=200&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if data.get('status') != 'ok':
        st.error(f"Erreur API pour {symbol_with_slash}: {data.get('message', 'Réponse invalide')}")
        return None
    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df = df.astype(float)
    return df.iloc[::-1]

if st.button("Lancer le Scan"):
    if not api_key:
        st.error("Clé API Twelve Data non configurée. Veuillez l'ajouter dans les 'Secrets' de Streamlit.")
    else:
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # === CORRECTION : Gestion de la limite de l'API ===
        # On va traiter les paires par lots de 7 pour rester sous la limite de 8 par minute.
        api_call_count = 0
        API_CALL_LIMIT_PER_MINUTE = 7 

        for i, pair in enumerate(forex_pairs):
            status_text.text(f"Traitement de la paire {i+1}/{len(forex_pairs)}: {pair}")
            
            # --- Vérification et pause si la limite est atteinte ---
            if api_call_count >= API_CALL_LIMIT_PER_MINUTE:
                status_text.warning("Limite API atteinte, mise en pause pendant 65 secondes...")
                time.sleep(65) # On attend un peu plus d'une minute pour être sûr.
                api_call_count = 0 # On réinitialise le compteur.

            df_ohlc = get_twelve_data(pair, timeframes[selected_tf], api_key)
            api_call_count += 1 # On incrémente le compteur après chaque appel.

            if df_ohlc is None or df_ohlc.empty:
                continue
            
            df_final = calculate_all_indicators(df_ohlc.copy())
            last_row = df_final.iloc[-1].copy()
            last_row['Pair'] = f"{pair[:3]}/{pair[3:]}"
            all_results.append(last_row)
            
            progress_bar.progress((i + 1) / len(forex_pairs))
        
        status_text.success("Scan terminé !")

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
        else:
            st.info("Le scan n'a retourné aucun résultat. Vérifiez les erreurs API ci-dessus.")
