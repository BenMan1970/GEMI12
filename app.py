import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import traceback
import requests # Remplacer yfinance par requests

# Configuration de la page Streamlit
st.set_page_config(page_title="Scanner Confluence Forex (Twelve Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Twelve Data)")
st.markdown("*Utilisation de l'API Twelve Data pour les données de marché H4*")

# --- Configuration API et Paires ---
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
API_KEY = None
try:
    API_KEY = st.secrets["TWELVE_DATA_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("La clé API Twelve Data (TWELVE_DATA_API_KEY) n'a pas été trouvée dans les secrets Streamlit.")
    st.info("Veuillez l'ajouter : TWELVE_DATA_API_KEY = 'votre_clé_api'")
    st.stop()

if not API_KEY:
    st.error("La clé API Twelve Data (TWELVE_DATA_API_KEY) est vide dans les secrets Streamlit.")
    st.stop()

FOREX_PAIRS_TD = [ # Format pour Twelve Data
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
    'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/JPY',
    'GBP/JPY', 'EUR/GBP'
]
# Nombre de bougies à récupérer (assez pour les indicateurs comme Ichimoku sur 52 périodes)
DATA_OUTPUT_SIZE = 250 # Plus de points pour H4 pour être sûr pour Ichimoku etc.
INTERVAL_TD = "4h"

# --- Fonctions Indicateurs (Identiques à votre version yfinance) ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    if dc.empty or len(dc) < p: return pd.Series([np.nan] * len(dc), index=dc.index)
    hl=int(p/2); sl=int(np.sqrt(p))
    wma1=dc.rolling(window=hl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    wma2=dc.rolling(window=p).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    diff=2*wma1-wma2; return diff.rolling(window=sl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)

def rsi_pine(po4,p=10):
    if po4.empty or len(po4) < p+1 : return pd.Series([50] * len(po4), index=po4.index)
    d=po4.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0);ag=rma(g,p);al=rma(l,p);rs=ag/al.replace(0,1e-9);rsi=100-(100/(1+rs));return rsi.fillna(50)

def adx_pine(h,l,c,p=14):
    if h.empty or len(h) < p+1 : return pd.Series([0] * len(h), index=h.index)
    tr1=h-l;tr2=abs(h-c.shift(1));tr3=abs(l-c.shift(1));tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1);atr=rma(tr,p)
    um=h.diff();dm=l.shift(1)-l
    pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index);mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index)
    satr=atr.replace(0,1e-9);pdi=100*(rma(pdm,p)/satr);mdi=100*(rma(mdm,p)/satr)
    dxden=(pdi+mdi).replace(0,1e-9);dx=100*(abs(pdi-mdi)/dxden);return rma(dx,p).fillna(0)

def heiken_ashi_pine(dfo): # Prend le DataFrame avec 'Open', 'High', 'Low', 'Close'
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

def smoothed_heiken_ashi_pine(dfo,l1=10,l2=10): # Prend le DataFrame avec 'Open', 'High', 'Low', 'Close'
    if dfo.empty or len(dfo) < max(l1,l2):
        empty_series = pd.Series([np.nan] * len(dfo), index=dfo.index)
        return empty_series, empty_series
    eo=ema(dfo['Open'],l1);eh=ema(dfo['High'],l1);el=ema(dfo['Low'],l1);ec=ema(dfo['Close'],l1)
    hai=pd.DataFrame({'Open':eo,'High':eh,'Low':el,'Close':ec},index=dfo.index) # Créer un DF pour heiken_ashi_pine
    hao_i,hac_i=heiken_ashi_pine(hai);sho=ema(hao_i,l2);shc=ema(hac_i,l2);return sho,shc

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    # Version simplifiée de votre code yfinance pour Ichimoku
    min_len_req=max(tenkan_p,kijun_p,senkou_b_p)
    if len(df_high)<min_len_req or len(df_low)<min_len_req or len(df_close)<min_len_req:
        # print(f"Ichi: Données insuffisantes ({len(df_close)}) vs requis {min_len_req}.")
        return 0
    ts=(df_high.rolling(window=tenkan_p).max()+df_low.rolling(window=tenkan_p).min())/2
    ks=(df_high.rolling(window=kijun_p).max()+df_low.rolling(window=kijun_p).min())/2
    sa=(ts+ks)/2 # Senkou A non décalé ici pour comparaison directe
    sb=(df_high.rolling(window=senkou_b_p).max()+df_low.rolling(window=senkou_b_p).min())/2 # Senkou B non décalé

    # Vérifier que les dernières valeurs ne sont pas NaN
    if df_close.empty or sa.empty or sb.empty or \
       pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
        # print("Ichi: NaN détecté dans close/spans finaux.")
        return 0

    ccl=df_close.iloc[-1]
    cssa=sa.iloc[-1] # SSA actuel (non décalé)
    cssb=sb.iloc[-1] # SSB actuel (non décalé)
    
    # Pour un signal plus robuste, il faudrait décaler SSA et SSB (nuage futur)
    # et aussi considérer Chikou Span. Cette version est très simplifiée.
    # Signal basé sur position du prix par rapport au nuage *actuel* (non décalé)
    ctn=max(cssa,cssb) # Haut du nuage actuel
    cbn=min(cssa,cssb) # Bas du nuage actuel
    sig=0
    if ccl>ctn: sig=1   # Prix au-dessus du nuage
    elif ccl<cbn: sig=-1 # Prix en-dessous du nuage
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
        "timezone": "UTC"
    }
    try:
        response = requests.get(TWELVE_DATA_API_URL, params=params)
        response.raise_for_status()  # Erreur si statut HTTP 4xx/5xx
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
        # Convertir en numérique, les erreurs deviennent NaN
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.iloc[::-1] # Inverser: les plus anciennes en premier

        # Renommer les colonnes pour correspondre à ce qu'attendent les fonctions d'indicateurs
        df.rename(columns={
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) # Enlever lignes avec NaN dans OHLC

        if df.empty or len(df) < 60: # Besoin d'assez de données pour les indicateurs
            st.warning(f"Twelve Data: Données insuffisantes pour {symbol} après traitement ({len(df)} barres). Min 60 requis.")
            print(f"Twelve Data: Données insuffisantes pour {symbol} ({len(df)} barres).")
            return None
        
        # S'assurer que l'index est UTC (Twelve Data le fait déjà si timezone=UTC est spécifié)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz.zone != 'UTC':
            df.index = df.index.tz_convert('UTC')

        print(f"Données pour {symbol} OK via Twelve Data. Retour de {len(df)} lignes.\n--- Fin get_data_twelvedata {symbol} ---\n")
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de requête HTTP pour {symbol} (Twelve Data): {e}")
        print(f"ERREUR REQUESTS pour {symbol} (Twelve Data):\n{traceback.format_exc()}")
        return None
    except Exception as e:
        st.error(f"Erreur générale lors de la récupération des données Twelve Data pour {symbol}: {e}")
        print(f"ERREUR GENERALE get_data_twelvedata pour {symbol}:\n{traceback.format_exc()}")
        return None

# --- Fonction calculate_all_signals_pine (Identique à votre version yfinance) ---
def calculate_all_signals_pine(data): # Attend 'Open', 'High', 'Low', 'Close'
    if data is None or len(data) < 60: # Augmenté un peu pour être sûr avec HMA 20 etc.
        print(f"calculate_all_signals: Données non fournies ou trop courtes ({len(data) if data is not None else 'None'} lignes).")
        return None
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        print(f"calculate_all_signals: Colonnes OHLC manquantes. Colonnes dispo: {data.columns}")
        return None
    
    # Vérifier les types de données (doivent être float/int pour calculs)
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


    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open'] # Renommé pour éviter conflit avec fonction open()
    ohlc4 = (open_price + high + low + close) / 4
    
    bull_confluences = 0
    bear_confluences = 0
    signal_details_pine = {}

    # 1. HMA
    try:
        hma_series = hull_ma_pine(close, 20)
        if not hma_series.empty and len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]
            hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev: bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev: bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A (peu de données)"
    except Exception as e: signal_details_pine['HMA'] = "ErrHMA"; print(f"Erreur calcul HMA: {e}")

    # 2. RSI
    try:
        rsi_series = rsi_pine(ohlc4, 10) # ohlc4 utilisé ici
        if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]
            signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50: bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50: bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = "N/A"; signal_details_pine['RSI_val'] = "N/A"
    except Exception as e: signal_details_pine['RSI'] = "ErrRSI"; signal_details_pine['RSI_val'] = "N/A"; print(f"Erreur calcul RSI: {e}")

    # 3. ADX
    try:
        adx_series = adx_pine(high, low, close, 14)
        if not adx_series.empty and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]
            signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20: bull_confluences += 1; bear_confluences += 1; signal_details_pine['ADX'] = f"✔({adx_val:.0f})" # ADX > 20 compte pour les deux
            else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else: signal_details_pine['ADX'] = "N/A"; signal_details_pine['ADX_val'] = "N/A"
    except Exception as e: signal_details_pine['ADX'] = "ErrADX"; signal_details_pine['ADX_val'] = "N/A"; print(f"Erreur calcul ADX: {e}")

    # 4. Heiken Ashi
    try: # heiken_ashi_pine attend un DataFrame avec 'Open', 'High', 'Low', 'Close'
        ha_open, ha_close = heiken_ashi_pine(data[['Open', 'High', 'Low', 'Close']])
        if not ha_open.empty and not ha_close.empty and \
           not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception as e: signal_details_pine['HA'] = "ErrHA"; print(f"Erreur calcul Heiken Ashi: {e}")

    # 5. Smoothed Heiken Ashi
    try: # smoothed_heiken_ashi_pine attend un DataFrame avec 'Open', 'High', 'Low', 'Close'
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
        # Gestion du cas où ichimoku retourne 0 à cause de données insuffisantes explicitement
        elif ichimoku_signal_val == 0 and (len(data) < max(9,26,52) or high.rolling(window=max(9,26,52)).max().iloc[-1:].isna().any()):
            signal_details_pine['Ichi'] = "N/D" # Non Disponible (données)
        else: signal_details_pine['Ichi'] = "─"
    except Exception as e: signal_details_pine['Ichi'] = "ErrIchi"; print(f"Erreur calcul Ichimoku: {e}")
    
    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences: direction = "HAUSSIER"
    elif bear_confluences > bull_confluences: direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0 : direction = "CONFLIT" # Cas où bull et bear sont égaux mais non nuls
        
    return {
        'confluence_P': confluence_value, 'direction_P': direction,
        'bull_P': bull_confluences, 'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"),
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

# --- Interface Streamlit (adaptée pour Twelve Data) ---
col1,col2=st.columns([1,3])
with col1:
    st.subheader("⚙️ Paramètres");min_conf=st.selectbox("Confluence min (0-6)",options=[0,1,2,3,4,5,6],index=3,format_func=lambda x:f"{x} (confluence)")
    show_all=st.checkbox("Voir toutes les paires (ignorer filtre)");
    pair_to_debug = st.selectbox("🔍 Afficher OHLC pour:", ["Aucune"] + FOREX_PAIRS_TD, index=0) # Utilise FOREX_PAIRS_TD
    scan_btn=st.button(f"🔍 Scanner (Données Twelve Data {INTERVAL_TD})",type="primary",use_container_width=True) # Texte du bouton mis à jour

with col2:
    if scan_btn:
        st.info(f"🔄 Scan en cours (Twelve Data {INTERVAL_TD})...");pr_res=[];pb=st.progress(0);stx=st.empty()
        
        if pair_to_debug != "Aucune":
            st.subheader(f"Données OHLC pour {pair_to_debug} (Twelve Data {INTERVAL_TD}):")
            # Utilise la nouvelle fonction pour le debug
            debug_data = get_data_twelvedata(pair_to_debug, interval=INTERVAL_TD, outputsize=100) # Moins de données pour debug
            if debug_data is not None and not debug_data.empty:
                st.dataframe(debug_data[['Open','High','Low','Close']].tail(10))
            else: st.warning(f"N'a pas pu charger données de débogage pour {pair_to_debug} via Twelve Data.")
            st.divider()

        for i, symbol_td_scan in enumerate(FOREX_PAIRS_TD): # Itère sur les paires Twelve Data
            # pnd est le nom court, symbol_td_scan est le symbole complet pour l'API
            pnd = symbol_td_scan.split('/')[0] + symbol_td_scan.split('/')[1] if '/' in symbol_td_scan else symbol_td_scan
            
            cp=(i+1)/len(FOREX_PAIRS_TD);pb.progress(cp);stx.text(f"Analyse (TD {INTERVAL_TD}):{pnd}({i+1}/{len(FOREX_PAIRS_TD)})")
            
            # Utilise la nouvelle fonction get_data_twelvedata
            d_h_td = get_data_twelvedata(symbol_td_scan, interval=INTERVAL_TD, outputsize=DATA_OUTPUT_SIZE)
            
            if d_h_td is not None and not d_h_td.empty:
                sigs=calculate_all_signals_pine(d_h_td)
                if sigs:
                    strs=get_stars_pine(sigs['confluence_P'])
                    rd={'Paire':pnd,'Direction':sigs['direction_P'],'Conf. (0-6)':sigs['confluence_P'],'Étoiles':strs,
                        'RSI':sigs['rsi_P'],'ADX':sigs['adx_P'],'Bull':sigs['bull_P'],'Bear':sigs['bear_P'],'details':sigs['signals_P']}
                    pr_res.append(rd)
                else:
                    pr_res.append({'Paire':pnd,'Direction':'ERREUR CALCUL','Conf. (0-6)':0,'Étoiles':'N/A',
                                   'RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Calcul signaux (TD) échoué'}})
            else:
                pr_res.append({'Paire':pnd,'Direction':'ERREUR DONNÉES TD','Conf. (0-6)':0,'Étoiles':'N/A',
                               'RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,
                               'details':{'Info':'Données Twelve Data non dispo/symb invalide (logs serveur)'}})
            time.sleep(0.3) # Petite pause pour l'API Twelve Data (peut être ajustée)
        
        pb.empty();stx.empty()
        
        if pr_res:
            dfa=pd.DataFrame(pr_res)
            dfd=dfa[dfa['Conf. (0-6)']>=min_conf].copy()if not show_all else dfa.copy()
            
            if not show_all:st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence (Twelve Data {INTERVAL_TD}).")
            else:st.info(f"🔍 Affichage des {len(dfd)} paires (Twelve Data {INTERVAL_TD}).")
            
            if not dfd.empty:
                dfds=dfd.sort_values('Conf. (0-6)',ascending=False)
                vcs=[c for c in['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear']if c in dfds.columns]
                st.dataframe(dfds[vcs],use_container_width=True,hide_index=True)
                
                with st.expander(f"📊 Détails des signaux (Twelve Data {INTERVAL_TD})"):
                    for _,r in dfds.iterrows():
                        sm=r.get('details',{});
                        if not isinstance(sm,dict):sm={'Info':'Détails non dispo'}
                        st.write(f"**{r.get('Paire','N/A')}** - {r.get('Étoiles','N/A')} ({r.get('Conf. (0-6)','N/A')}) - Dir: {r.get('Direction','N/A')}")
                        dc=st.columns(6);so=['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx,sk in enumerate(so):dc[idx].metric(label=sk,value=sm.get(sk,"N/P"))
                        st.divider()
            else:st.warning(f"❌ Aucune paire avec critères filtrage (Twelve Data {INTERVAL_TD}). Vérifiez erreurs données/symbole.")
        else:st.error(f"❌ Aucune paire traitée (Twelve Data {INTERVAL_TD}). Vérifiez logs serveur.")

with st.expander(f"ℹ️ Comment ça marche (Logique Pine Script avec Données Twelve Data {INTERVAL_TD})"):
    st.markdown(f"""**6 Signaux Confluence:** HMA(20), RSI(10 sur OHLC/4), ADX(14)>=20, Heiken Ashi(Simple), Smoothed Heiken Ashi(10,10), Ichimoku(9,26,52 simplifié).
**Comptage & Étoiles:** Basé sur le nombre de signaux concordants.
**Source des Données:** API Twelve Data, intervalle {INTERVAL_TD}.
**Paires Scannées:** {', '.join(FOREX_PAIRS_TD)}.""")

# Afficher l'heure actuelle pour vérifier le rafraîchissement
st.caption(f"Scanner {INTERVAL_TD} (Twelve Data). Dernière actualisation de l'interface : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")
