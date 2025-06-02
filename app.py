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
        response.raise_for_status()
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

        # Colonnes attendues pour OHLC
        ohlc_cols = ['open', 'high', 'low', 'close']
        # Colonne volume, si présente
        vol_col_original = 'volume' # Nom attendu de l'API
        
        # Vérifier si les colonnes OHLC de base sont présentes
        for col_name in ohlc_cols:
            if col_name not in df.columns:
                st.error(f"Colonne '{col_name}' manquante dans les données de Twelve Data pour {symbol}.")
                print(f"Colonne '{col_name}' manquante pour {symbol}. Colonnes reçues: {df.columns.tolist()}")
                return None # Données critiques manquantes

        # Convertir OHLC en numérique
        for col in ohlc_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Traitement optionnel du volume
        columns_to_select = ohlc_cols[:] # Commence avec les colonnes OHLC
        rename_map = {
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close'
        }

        if vol_col_original in df.columns:
            print(f"Colonne '{vol_col_original}' trouvée pour {symbol}. Traitement...")
            df[vol_col_original] = pd.to_numeric(df[vol_col_original], errors='coerce')
            columns_to_select.append(vol_col_original)
            rename_map[vol_col_original] = 'Volume' # Renommer en 'Volume' avec majuscule
        else:
            print(f"Colonne '{vol_col_original}' NON trouvée pour {symbol}. Elle sera ignorée.")
            # Créer une colonne 'Volume' avec des zéros ou NaNs si elle n'existe pas,
            # pour que le reste du code ne casse pas s'il s'attend à une colonne 'Volume'.
            df['Volume'] = 0 # Ou np.nan si vous préférez
            # Note: La colonne 'Volume' (avec majuscule) est créée ici, donc pas besoin de l'ajouter à columns_to_select
            # ou rename_map si elle n'est pas dans les données originales.

        df = df[columns_to_select] # Sélectionne uniquement les colonnes traitées (OHLC et volume si existant)
        df = df.iloc[::-1]

        df.rename(columns=rename_map, inplace=True)
        
        # Si 'Volume' n'était pas dans les données originales et a été créé comme df['Volume']=0,
        # il est déjà nommé correctement. Sinon, le rename_map s'en est chargé.

        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)

        if df.empty or len(df) < 60:
            st.warning(f"Twelve Data: Données insuffisantes pour {symbol} après traitement ({len(df)} barres). Min 60 requis.")
            print(f"Twelve Data: Données insuffisantes pour {symbol} ({len(df)} barres).")
            return None
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz.zone != 'UTC':
            df.index = df.index.tz_convert('UTC')

        print(f"Données pour {symbol} OK via Twelve Data. Colonnes finales: {df.columns.tolist()}. Retour de {len(df)} lignes.")
        print(f"--- Fin get_data_twelvedata {symbol} ---\n")
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de requête HTTP pour {symbol} (Twelve Data): {e}")
        print(f"ERREUR REQUESTS pour {symbol} (Twelve Data):\n{traceback.format_exc()}")
        return None
    except KeyError as e: # Capturer spécifiquement les KeyError si une colonne est toujours attendue à tort
        st.error(f"Erreur de clé (colonne manquante probable : {e}) lors du traitement des données Twelve Data pour {symbol}.")
        print(f"ERREUR KEYERROR get_data_twelvedata pour {symbol}: {e}\n{traceback.format_exc()}")
        return None
    except Exception as e:
        st.error(f"Erreur générale lors de la récupération des données Twelve Data pour {symbol}: {e}")
        print(f"ERREUR GENERALE get_data_twelvedata pour {symbol}:\n{traceback.format_exc()}")
        return None
