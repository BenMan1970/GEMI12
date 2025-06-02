import streamlit as st # DOIT ÊTRE LA PREMIÈRE LIGNE ACTIVE
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import traceback
import requests

# Configuration de la page Streamlit (doit être la première commande Streamlit après les imports)
st.set_page_config(page_title="Scanner Confluence Forex (Twelve Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Twelve Data)")
st.markdown("*Utilisation de l'API Twelve Data pour les données de marché H4*")

# --- Configuration API et Paires ---
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
API_KEY = None
try:
    API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
except (FileNotFoundError, KeyError): # FileNotFoundError pour secrets locaux, KeyError pour Streamlit Cloud
    st.error("La clé API Twelve Data (TWELVE_DATA_API_KEY) n'a pas été trouvée dans les secrets Streamlit.")
    st.info("Veuillez l'ajouter : TWELVE_DATA_API_KEY = 'votre_clé_api'")
    st.stop()

if not API_KEY: # Double vérification au cas où la clé serait vide
    st.error("La clé API Twelve Data (TWELVE_DATA_API_KEY) est vide dans les secrets Streamlit.")
    st.stop()

FOREX_PAIRS_TD = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
    'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/JPY',
    'GBP/JPY', 'EUR/GBP'
]
DATA_OUTPUT_SIZE = 250 # Nombre de bougies à récupérer
INTERVAL_TD = "4h" # Intervalle de temps pour Twelve Data

# --- Fonctions Indicateurs ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    if dc.empty or len(dc) < p: return pd.Series([np.nan] * len(dc), index=dc.index)
    hl=int(p/2); sl=int(np.sqrt(p))
    wma1=dc.rolling(window=hl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    wma2=dc.rolling(window=p).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    diff=2*wma1-wma2; return diff.rolling(window=sl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)

def rsi_pine(po4,p=10):
    if po4.empty or len(po4) < p+1 : return pd.Series([50.0] * len(po4), index=po4.index) # Retourner float
    d=po4.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0);ag=rma(g,p);al=rma(l,p);rs=ag/al.replace(0,1e-9);rsi=100-(100/(1+rs));return rsi.fillna(50.0) # Retourner float

def adx_pine(h,l,c,p=14):
    if h.empty or len(h) < p+1 : return pd.Series([0.0] * len(h), index=h.index) # Retourner float
    tr1=h-l;tr2=abs(h-c.shift(1));tr3=abs(l-c.shift(1));tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1);atr=rma(tr,p)
    um=h.diff();dm=l.shift(1)-l
    pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index);mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index)
    satr=atr.replace(0,1e-9);pdi=100*(rma(pdm,p)/satr);mdi=100*(rma(mdm,p)/satr)
    dxden=(pdi+mdi).replace(0,1e-9);dx=100*(abs(pdi-mdi)/dxden);return rma(dx,p).fillna(0.0) # Retourner float

def heiken_ashi_pine(dfo):
    ha=pd.DataFrame(index=dfo.index)
    if dfo.empty:
        ha['HA_Open']=pd.Series(dtype=float);ha['HA_Close']=pd.Series(dtype=float)
        return ha['HA_Open'],ha['HA_Close']
    ha['HA_Close']=(dfo['Open']+dfo['High']+dfo['Low']+dfo['Close'])/4;ha['HA_Open']=np.nan
    if not dfo.empty:
        ha.iloc[0,ha.columns.get_loc('HA_Open')]=(dfo['Open'].iloc[0]+dfo['Close'].iloc[0])/2
        for i in range(1,len(dfo)):
            ha.iloc[i,ha.columns.get_loc('HA_Open')]=(ha.iloc[i-1,ha.columns.get_loc('HA_Open')]+ha.iloc[i-1,ha.columns.get_loc('HA_Close')])/2
    return ha['HA_Open'],ha['HA_Close']

def smoothed_heiken_ashi_pine(dfo,l1=10,l2=10):
    if dfo.empty or len(dfo) < max(l1,l2):
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series
    eo=ema(dfo['Open'],l1);eh=ema(dfo['High'],l1);el=ema(dfo['Low'],l1);ec=ema(dfo['Close'],l1)
    hai=pd.DataFrame({'Open':eo,'High':eh,'Low':el,'Close':ec},index=dfo.index)
    hao_i,hac_i=heiken_ashi_pine(hai);sho=ema(hao_i,l2);shc=ema(hac_i,l2);return sho,shc

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req=max(tenkan_p,kijun_p,senkou_b_p)
    if len(df_high)<min_len_req or len(df_low)<min_len_req or len(df_close)<min_len_req:
        return 0 # Signal neutre
    ts=(df_high.rolling(window=tenkan_p).max()+df_low.rolling(window=tenkan_p).min())/2
    ks=(df_high.rolling(window=kijun_p).max()+df_low.rolling(window=kijun_p).min())/2
    sa=(ts+ks)/2
    sb=(df_high.rolling(window=senkou_b_p).max()+df_low.rolling(window=senkou_b_p).min())/2

    if df_close.empty or sa.empty or sb.empty or \
       pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
        return 0 # Signal neutre si NaN

    ccl=df_close.iloc[-1];cssa=sa.iloc[-1];cssb=sb.iloc[-1]
    ctn=max(cssa,cssb);cbn=min(cssa,cssb);sig=0
    if ccl>ctn: sig=1
    elif ccl<cbn: sig=-1
    return sig

# --- Nouvelle fonction de récupération de données pour Twelve Data ---
@st.cache_data(ttl=60*15) # Cache pour 15 minutes pour H4
def get_data_twelvedata(symbol: str, interval: str = INTERVAL_TD, outputsize: int = DATA_OUTPUT_SIZE):
    print(f"\n--- Début get_data_twelvedata: sym='{symbol}', interval='{interval}', outputsize='{outputsize}' ---")
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "timezone": "UTC" # Spécifier UTC pour la cohérence
    }
    try:
        response = requests.get(TWELVE_DATA_API_URL, params=params)
        response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
        data_json = response.json()

        if data_json.get("status") == "error":
            msg = data_json.get('message', 'Erreur inconnue de l API Twelve Data.')
            st.warning(f"Twelve Data API pour {symbol}: {msg}")
            print(f"Twelve Data API ERREUR pour {symbol}: {msg}")
            return None

        if "values" not in data_json or not data_json["values"]:
            st.warning(f"Twelve Data: Aucune donnée 'values' reçue pour {symbol}.")
            print(f"Twelve Data: Aucune donnée 'values' pour {symbol}.")
            return None

        df = pd.DataFrame(data_json["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        # Colonnes OHLC attendues de l'API (en minuscules)
        ohlc_cols_api = ['open', 'high', 'low', 'close']
        # Colonne volume attendue de l'API (en minuscules)
        vol_col_api = 'volume'
        
        # Vérifier la présence des colonnes OHLC critiques
        for col_name in ohlc_cols_api:
            if col_name not in df.columns:
                st.error(f"Colonne critique '{col_name}' manquante dans les données de Twelve Data pour {symbol}.")
                print(f"Colonne critique '{col_name}' manquante pour {symbol}. Colonnes reçues: {df.columns.tolist()}")
                return None # Données critiques manquantes, impossible de continuer

        # Convertir les colonnes OHLC en numérique
        for col in ohlc_cols_api:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Préparer la sélection et le renommage des colonnes
        columns_to_select_from_api = ohlc_cols_api[:] # Commence avec les colonnes OHLC
        rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'} # Noms finaux avec majuscule

        # Gérer la colonne volume (optionnelle)
        if vol_col_api in df.columns:
            print(f"Colonne '{vol_col_api}' trouvée pour {symbol}. Traitement...")
            df[vol_col_api] = pd.to_numeric(df[vol_col_api], errors='coerce')
            columns_to_select_from_api.append(vol_col_api)
            rename_map[vol_col_api] = 'Volume' # Nom final avec majuscule
            df_final = df[columns_to_select_from_api].copy() # Copier les colonnes sélectionnées
            df_final.rename(columns=rename_map, inplace=True)
        else:
            print(f"Colonne '{vol_col_api}' NON trouvée pour {symbol}. Création d'une colonne 'Volume' avec des zéros.")
            df_final = df[columns_to_select_from_api].copy() # Copier les colonnes OHLC
            df_final.rename(columns=rename_map, inplace=True) # Renommer OHLC en majuscules
            df_final['Volume'] = 0.0 # Ajouter la colonne 'Volume' remplie de 0.0

        df_final = df_final.iloc[::-1] # Inverser pour avoir les plus anciennes en premier
        
        df_final.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # Supprimer les lignes avec NaN dans OHLC

        if df_final.empty or len(df_final) < 60: # Besoin d'assez de données pour les indicateurs
            st.warning(f"Twelve Data: Données insuffisantes pour {symbol} après traitement ({len(df_final)} barres). Min 60 requis.")
            print(f"Twelve Data: Données insuffisantes pour {symbol} ({len(df_final)} barres).")
            return None
        
        # S'assurer que l'index est UTC
        if df_final.index.tz is None:
            df_final.index = df_final.index.tz_localize('UTC')
        elif df_final.index.tz.zone != 'UTC': # Convertir si ce n'est pas déjà UTC
            df_final.index = df_final.index.tz_convert('UTC')

        print(f"Données pour {symbol} OK. Colonnes finales: {df_final.columns.tolist()}. Retour de {len(df_final)} lignes.")
        print(f"--- Fin get_data_twelvedata {symbol} ---\n")
        return df_final

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de requête HTTP pour {symbol} (Twelve Data): {e}")
        print(f"ERREUR REQUESTS pour {symbol} (Twelve Data):\n{traceback.format_exc()}")
        return None
    except KeyError as e: # Capturer spécifiquement les KeyError
        st.error(f"Erreur de clé (colonne manquante probable : {e}) lors du traitement des données Twelve Data pour {symbol}.")
        print(f"ERREUR KEYERROR get_data_twelvedata pour {symbol}: {e}\n{traceback.format_exc()}")
        return None
    except Exception as e:
        st.error(f"Erreur générale lors de la récupération des données Twelve Data pour {symbol}: {e}")
        print(f"ERREUR GENERALE get_data_twelvedata pour {symbol}:\n{traceback.format_exc()}")
        return None

# --- Fonction calculate_all_signals_pine ---
def calculate_all_signals_pine(data): # Attend un DataFrame avec 'Open', 'High', 'Low', 'Close' (et optionnellement 'Volume')
    if data is None or len(data) < 60:
        print(f"calculate_all_signals: Données non fournies ou trop courtes ({len(data) if data is not None else 'None'} lignes).")
        return None
    
    required_cols = ['Open', 'High', 'Low', 'Close'] # 'Volume' n'est pas directement utilisé par ces indicateurs
    if not all(col in data.columns for col in required_cols):
        print(f"calculate_all_signals: Colonnes OHLC manquantes. Colonnes disponibles: {data.columns.tolist()}")
        return None
    
    # Vérifier les types de données et convertir si nécessaire
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            print(f"calculate_all_signals: Colonne {col} n'est pas numérique (type: {data[col].dtype}). Tentative de conversion.")
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isnull().all(): # Si tout devient NaN après conversion
                 print(f"calculate_all_signals: Colonne {col} est entièrement NaN après conversion. Abandon.")
                 return None
    data.dropna(subset=required_cols, inplace=True) # Redrop au cas où la conversion a créé des NaNs
    if len(data) < 60:
        print(f"calculate_all_signals: Données trop courtes après nettoyage ({len(data)} lignes).")
        return None

    close = data['Close']; high = data['High']; low = data['Low']; open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    
    bull_confluences = 0; bear_confluences = 0
    signal_details_pine = {}

    # 1. HMA
    try:
        hma_series = hull_ma_pine(close, 20)
        if not hma_series.empty and len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]; hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev: bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev: bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A (peu de données)"
    except Exception as e: signal_details_pine['HMA'] = "ErrHMA"; print(f"Erreur calcul HMA: {e}")

    # 2. RSI
    try:
        rsi_series = rsi_pine(ohlc4, 10) # ohlc4 utilisé ici
        if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]; signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50: bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50: bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = "N/A"; signal_details_pine['RSI_val'] = "N/A"
    except Exception as e: signal_details_pine['RSI']="ErrRSI";signal_details_pine['RSI_val']="N/A";print(f"Erreur calcul RSI: {e}")

    # 3. ADX
    try:
        adx_series = adx_pine(high, low, close, 14)
        if not adx_series.empty and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]; signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20: bull_confluences += 1; bear_confluences += 1; signal_details_pine['ADX'] = f"✔({adx_val:.0f})" # ADX > 20 compte pour les deux
            else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else: signal_details_pine['ADX'] = "N/A"; signal_details_pine['ADX_val'] = "N/A"
    except Exception as e: signal_details_pine['ADX']="ErrADX";signal_details_pine['ADX_val']="N/A";print(f"Erreur calcul ADX: {e}")

    # 4. Heiken Ashi
    try:
        ha_open, ha_close = heiken_ashi_pine(data[['Open', 'High', 'Low', 'Close']]) # Passe le DF avec les bonnes colonnes
        if not ha_open.empty and not ha_close.empty and \
           not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception as e: signal_details_pine['HA'] = "ErrHA"; print(f"Erreur calcul Heiken Ashi: {e}")

    # 5. Smoothed Heiken Ashi
    try:
        sha_open, sha_close = smoothed_heiken_ashi_pine(data[['Open', 'High', 'Low', 'Close']], 10, 10)
        if not sha_open.empty and not sha_close.empty and \
           not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception as e: signal_details_pine['SHA'] = "ErrSHA"; print(f"Erreur calcul Smoothed Heiken Ashi: {e}")

    # 6. Ichimoku
    try:
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1: bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1: bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
        elif ichimoku_signal_val == 0 and (len(data) < max(9,26,52) or high.rolling(window=max(9,26,52)).max().iloc[-1:].isna().any()):
            signal_details_pine['Ichi'] = "N/D" # Non Disponible (données)
        else: signal_details_pine['Ichi'] = "─"
    except Exception as e: signal_details_pine['Ichi'] = "ErrIchi"; print(f"Erreur calcul Ichimoku: {e}")
    
    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences: direction = "HAUSSIER"
    elif bear_confluences > bull_confluences: direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0 : direction = "CONFLIT"
        
    return {'confluence_P': confluence_value, 'direction_P': direction, 'bull_P': bull_confluences, 
            'bear_P': bear_confluences, 'rsi_P': signal_details_pine.get('RSI_val', "N/A"), 
            'adx_P': signal_details_pine.get('ADX_val', "N/A"), 'signals_P': signal_details_pine}

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

# --- Interface Streamlit ---
col1,col2=st.columns([1,3]) # Définir les colonnes pour l'interface

with col1:
    st.subheader("⚙️ Paramètres")
    min_conf=st.selectbox("Confluence min (0-6)",options=list(range(7)),index=3,format_func=lambda x:f"{x} (confluence)")
    show_all=st.checkbox("Voir toutes les paires (ignorer filtre)");
    pair_to_debug = st.selectbox("🔍 Afficher OHLC pour:", ["Aucune"] + FOREX_PAIRS_TD, index=0)
    scan_btn=st.button(f"🔍 Scanner (Données Twelve Data {INTERVAL_TD})",type="primary",use_container_width=True)

with col2:
    if scan_btn:
        st.info(f"🔄 Scan en cours (Twelve Data {INTERVAL_TD})...");
        scan_results_list=[]; # Renommé pour éviter confusion avec df plus tard
        progress_bar=st.progress(0); # Renommé pour clarté
        status_text=st.empty() # Renommé pour clarté
        
        if pair_to_debug != "Aucune":
            st.subheader(f"Données OHLCV pour {pair_to_debug} (Twelve Data {INTERVAL_TD}):")
            debug_data = get_data_twelvedata(pair_to_debug, interval=INTERVAL_TD, outputsize=100) # Moins de données pour debug
            if debug_data is not None and not debug_data.empty:
                # Afficher les colonnes qui existent réellement dans debug_data
                cols_to_show_debug = [col for col in ['Open','High','Low','Close','Volume'] if col in debug_data.columns]
                st.dataframe(debug_data[cols_to_show_debug].tail(10))
            else: st.warning(f"N'a pas pu charger données de débogage pour {pair_to_debug} via Twelve Data.")
            st.divider()

        for i, symbol_td_scan in enumerate(FOREX_PAIRS_TD):
            pnd = symbol_td_scan.replace('/', '') # Nom court de la paire pour affichage
            
            current_progress=(i+1)/len(FOREX_PAIRS_TD)
            progress_bar.progress(current_progress)
            status_text.text(f"Analyse (TD {INTERVAL_TD}): {pnd} ({i+1}/{len(FOREX_PAIRS_TD)})")
            
            d_h_td = get_data_twelvedata(symbol_td_scan, interval=INTERVAL_TD, outputsize=DATA_OUTPUT_SIZE)
            
            if d_h_td is not None and not d_h_td.empty:
                sigs=calculate_all_signals_pine(d_h_td.copy()) # Passer une copie pour éviter modif du cache
                if sigs:
                    stars_str=get_stars_pine(sigs['confluence_P']) # Renommé pour clarté
                    result_dict={'Paire':pnd,'Direction':sigs['direction_P'],'Conf. (0-6)':sigs['confluence_P'],'Étoiles':stars_str,
                        'RSI':sigs['rsi_P'],'ADX':sigs['adx_P'],'Bull':sigs['bull_P'],'Bear':sigs['bear_P'],'details':sigs['signals_P']}
                    scan_results_list.append(result_dict)
                else:
                    scan_results_list.append({'Paire':pnd,'Direction':'ERREUR CALCUL','Conf. (0-6)':0,'Étoiles':'N/A',
                                   'RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':f'Calcul signaux (TD) échoué pour {pnd}'}})
            else:
                scan_results_list.append({'Paire':pnd,'Direction':'ERREUR DONNÉES TD','Conf. (0-6)':0,'Étoiles':'N/A',
                               'RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,
                               'details':{'Info':f'Données Twelve Data non dispo/symb invalide pour {pnd}'}})
            time.sleep(0.33) # Respecter les limites de l'API (environ 3 req/sec pour certains plans gratuits)
        
        progress_bar.empty();status_text.empty() # Nettoyer la barre de progression et le texte
        
        if scan_results_list:
            results_df=pd.DataFrame(scan_results_list) # Renommé pour clarté
            
            # Appliquer le filtre de confluence
            if not show_all:
                filtered_df = results_df[results_df['Conf. (0-6)']>=min_conf].copy()
                st.success(f"🎯 {len(filtered_df)} paire(s) avec {min_conf}+ confluence (Twelve Data {INTERVAL_TD}).")
            else:
                filtered_df = results_df.copy()
                st.info(f"🔍 Affichage des {len(filtered_df)} paires (Twelve Data {INTERVAL_TD}).")
            
            if not filtered_df.empty:
                sorted_df=filtered_df.sort_values('Conf. (0-6)',ascending=False) # Renommé pour clarté
                visible_cols=[c for c in['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear']if c in sorted_df.columns]
                st.dataframe(sorted_df[visible_cols],use_container_width=True,hide_index=True)
                
                with st.expander(f"📊 Détails des signaux (Twelve Data {INTERVAL_TD})"):
                    for _,row_data in sorted_df.iterrows(): # Renommé pour clarté
                        signal_map=row_data.get('details',{}); # Renommé pour clarté
                        if not isinstance(signal_map,dict):signal_map={'Info':'Détails non dispo'}
                        st.write(f"**{row_data.get('Paire','N/A')}** - {row_data.get('Étoiles','N/A')} ({row_data.get('Conf. (0-6)','N/A')}) - Dir: {row_data.get('Direction','N/A')}")
                        
                        detail_cols=st.columns(6); # Renommé pour clarté
                        signal_order=['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx,signal_key in enumerate(signal_order): # Renommé pour clarté
                            detail_cols[idx].metric(label=signal_key,value=signal_map.get(signal_key,"N/P"))
                        st.divider()
            else:st.warning(f"❌ Aucune paire avec critères filtrage (Twelve Data {INTERVAL_TD}).")
        else:st.error(f"❌ Aucune paire traitée (Twelve Data {INTERVAL_TD}). Vérifiez logs serveur.")

with st.expander(f"ℹ️ Comment ça marche (Logique Pine Script avec Données Twelve Data {INTERVAL_TD})"):
    st.markdown(f"""**6 Signaux Confluence:** HMA(20), RSI(10 sur OHLC/4), ADX(14)>=20, Heiken Ashi(Simple), Smoothed Heiken Ashi(10,10), Ichimoku(9,26,52 simplifié).
**Comptage & Étoiles:** Basé sur le nombre de signaux concordants.
**Source des Données:** API Twelve Data, intervalle {INTERVAL_TD}.
**Paires Scannées:** {', '.join(FOREX_PAIRS_TD)}.""")

st.caption(f"Scanner {INTERVAL_TD} (Twelve Data). Dernière actualisation de l'interface : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
