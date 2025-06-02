# --- FETCH DATA ---
@st.cache_data(ttl=900)
def get_data(symbol):
    try:
        r = requests.get(TWELVE_DATA_API_URL, params={
            "symbol": symbol,
            "interval": INTERVAL,
            "outputsize": OUTPUT_SIZE,
            "apikey": API_KEY,
            "timezone": "UTC"
        })
        j = r.json()
        if "values" not in j:
            return None
        df = pd.DataFrame(j["values"])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        df = df.astype(float)
        df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"}, inplace=True)
        return df[['Open','High','Low','Close']]
    except Exception:
        return None

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
