import pandas as pd
import pandas_ta as ta
import numpy as np

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
    # Copie pour éviter les modifications sur le DataFrame original
    df_ = df.copy()

    # up = ta.change(high)
    # down = -ta.change(low)
    up = df_['high'].diff()
    down = -df_['low'].diff()

    # plusDM = na(up) ? na : (up > down and up > 0 ? up : 0)
    # minusDM = na(down) ? na : (down > up and down > 0 ? down : 0)
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    # tr = ta.tr
    # rmaTrueRange = ta.rma(tr, diLen)
    true_range = ta.true_range(df_['high'], df_['low'], df_['close'])
    rma_true_range = rma(true_range, di_len)

    # rmaPlusDM = ta.rma(plusDM, diLen)
    # rmaMinusDM = ta.rma(minusDM, diLen)
    rma_plus_dm = rma(pd.Series(plus_dm, index=df_.index), di_len)
    rma_minus_dm = rma(pd.Series(minus_dm, index=df_.index), di_len)
    
    # plusDI = rmaPlusDM / rmaTrueRange * 100
    # minusDI = rmaMinusDM / rmaTrueRange * 100
    # On gère la division par zéro
    plus_di = 100 * (rma_plus_dm / rma_true_range)
    minus_di = 100 * (rma_minus_dm / rma_true_range)
    
    # dx = math.abs(plusDI - minusDI) / (plusDI + minusDI == 0 ? 1 : plusDI + minusDI) * 100
    # On gère la division par zéro
    dx_denominator = plus_di + minus_di
    dx = 100 * (np.abs(plus_di - minus_di) / dx_denominator.replace(0, 1))

    # adxValue = ta.rma(dx, adxLen)
    adx_value = rma(dx, adx_len)
    
    return adx_value


def calculate_heikin_ashi_simple(df: pd.DataFrame):
    """
    Calcule le signal Heikin Ashi simple comme dans le script Pine.
    """
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Pine: var float haOpen = na, haOpen := na(haOpen[1]) ? ...
    # C'est une initialisation spéciale, on la réplique avec une boucle.
    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
    return ha_close, ha_open


def calculate_smoothed_heikin_ashi(df: pd.DataFrame, len1: int = 10, len2: int = 10):
    """
    Calcule le signal Heikin Ashi lissé (+) comme dans le script Pine.
    """
    # Étape 1 : Lisser OHLC avec EMA
    o1 = df['open'].ewm(span=len1, adjust=False).mean()
    c1 = df['close'].ewm(span=len1, adjust=False).mean()
    h1 = df['high'].ewm(span=len1, adjust=False).mean()
    l1 = df['low'].ewm(span=len1, adjust=False).mean()

    # Étape 2 : Calculer Heikin Ashi à partir des données lissées
    haclose1 = (o1 + h1 + l1 + c1) / 4
    
    haopen1 = pd.Series(np.nan, index=df.index)
    haopen1.iloc[0] = (o1.iloc[0] + c1.iloc[0]) / 2
    for i in range(1, len(df)):
        haopen1.iloc[i] = (haopen1.iloc[i-1] + haclose1.iloc[i-1]) / 2
        
    # Étape 3 : Lisser les bougies Heikin Ashi résultantes
    o2 = haopen1.ewm(span=len2, adjust=False).mean()
    c2 = haclose1.ewm(span=len2, adjust=False).mean()

    return o2, c2


def calculate_all_indicators(df: pd.DataFrame):
    """
    Fonction principale qui calcule tous les indicateurs et les signaux,
    en miroir du script Pine "Canadian Confluence Premium (Précision ADX)".
    """
    params = {
        "hmaLength": 20,
        "adxThreshold": 20,
        "rsiLength": 10,
        "adxLength": 14,
        "diLength": 14,
        "ichimokuLength": 9,
        "smoothedHaLen1": 10,
        "smoothedHaLen2": 10,
    }

    # --- CALCULS DES INDICATEURS ---
    
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

    # Ichimoku
    # Note : pandas_ta a des longueurs par défaut différentes, il faut les spécifier.
    ichimoku_df = ta.ichimoku(df['high'], df['low'], df['close'], 
                              tenkan=params['ichimokuLength'], kijun=26, senkou=52)
    # Renommer les colonnes pour plus de clarté
    df['tenkan'] = ichimoku_df[f'ITS_{params["ichimokuLength"]}']
    df['kijun'] = ichimoku_df['IKS_26']
    df['senkouA'] = ichimoku_df['ISA_9'] # Note: pandas_ta utilise les noms originaux
    df['senkouB'] = ichimoku_df['ISB_26'] # Le nom est trompeur, mais c'est bien la Senkou B à 52 périodes
    
    cloud_top = df[['senkouA', 'senkouB']].max(axis=1)
    cloud_bottom = df[['senkouA', 'senkouB']].min(axis=1)
    df['ichimokuSignal'] = np.select(
        [df['close'] > cloud_top, df['close'] < cloud_bottom],
        [1, -1],
        default=0
    )

    # --- CALCUL DES CONFLUENCES ---
    
    bull_conditions = [
        df['hmaSlope'] == 1,
        df['haSignal'] == 1,
        df['smoothedHaSignal'] == 1,
        df['rsiSignal'] == 1,
        df['adxHasMomentum'], # C'est déjà un booléen (True/False)
        df['ichimokuSignal'] == 1
    ]
    df['bullConfluences'] = np.sum(bull_conditions, axis=0)

    bear_conditions = [
        df['hmaSlope'] == -1,
        df['haSignal'] == -1,
        df['smoothedHaSignal'] == -1,
        df['rsiSignal'] == -1,
        df['adxHasMomentum'],
        df['ichimokuSignal'] == -1
    ]
    df['bearConfluences'] = np.sum(bear_conditions, axis=0)
    
    df['confluence'] = df[['bullConfluences', 'bearConfluences']].max(axis=1)

    return df


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================
if __name__ == '__main__':
    # Simuler un DataFrame de données OHLCV
    data = {
        'open': [100, 102, 101, 103, 105, 104, 106, 107, 105, 108, 110, 109, 108, 107, 106, 105, 104, 103, 105, 106],
        'high': [103, 104, 103, 105, 106, 106, 108, 109, 108, 110, 112, 110, 109, 108, 107, 106, 105, 105, 107, 108],
        'low': [99, 101, 100, 102, 104, 103, 105, 106, 104, 107, 109, 108, 107, 106, 105, 104, 103, 102, 104, 105],
        'close': [102, 103, 102, 104, 105, 105, 107, 108, 106, 109, 111, 109, 108, 107, 106, 105, 104, 104, 106, 107],
        'volume': [1000]*20
    }
    # Pour un test réel, il faudrait au moins 100 bougies pour que les indicateurs se stabilisent
    # df_ohlc = pd.read_csv("vos_donnees.csv") 
    df_ohlc = pd.DataFrame(data)

    # Calculer tous les indicateurs
    df_final = calculate_all_indicators(df_ohlc)

    # Afficher les dernières lignes avec les résultats
    # On affiche les colonnes clés pour la vérification
    print(df_final[[
        'close', 'adx', 'adxHasMomentum', 'rsi', 'rsiSignal', 
        'hmaSlope', 'haSignal', 'smoothedHaSignal', 'ichimokuSignal',
        'bullConfluences', 'bearConfluences', 'confluence'
    ]].tail(10))
