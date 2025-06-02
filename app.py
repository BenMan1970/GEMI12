# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import traceback
import requests

# --- CONFIG STREAMLIT ---
st.set_page_config(page_title="Scanner Confluence Forex (Twelve Data)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Twelve Data)")
st.markdown("*Utilisation de l'API Twelve Data pour les données de marché H1*")

# --- API CONFIG ---
TWELVE_DATA_API_URL = "https://api.twelvedata.com/time_series"
API_KEY = st.secrets.get("TWELVE_DATA_API_KEY")
if not API_KEY:
    st.error("Clé API manquante. Ajoutez-la dans .streamlit/secrets.toml : TWELVE_DATA_API_KEY = '...' ")
    st.stop()

FOREX_PAIRS_TD = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/GBP",
    "XAU/USD", "US30/USD", "NAS100/USD", "SPX/USD"
]
INTERVAL = "1h"
OUTPUT_SIZE = 250

# --- INDICATEURS ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

# ... (autres fonctions inchangées)

# --- INTERFACE UTILISATEUR ---
st.sidebar.header("Paramètres")
min_conf = st.sidebar.slider("Confluence minimale", 0, 6, 3)
show_all = st.sidebar.checkbox("Afficher toutes les paires", value=False)

if st.sidebar.button("Lancer le scan"):
    results = []
    for i, symbol in enumerate(FOREX_PAIRS_TD):
        st.sidebar.write(f"{symbol} ({i+1}/{len(FOREX_PAIRS_TD)})")
        df = get_data(symbol)
        time.sleep(1.0)
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

    if results:
        df_res = pd.DataFrame(results).sort_values(by="Confluences", ascending=False)
        st.markdown(df_res.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.download_button("📂 Exporter CSV", data=df_res.to_csv(index=False).encode('utf-8'), file_name="confluences.csv", mime="text/csv")
    else:
        st.warning("Aucun résultat correspondant aux critères.")

st.caption(f"Dernière mise à jour : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
