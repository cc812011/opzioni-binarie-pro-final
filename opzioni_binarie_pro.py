import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import pandas_ta as pta  # ← Nuova libreria
import time

st.set_page_config(page_title="Opzioni Binarie PRO", layout="wide")
st.title("🎯 Previsioni Opzioni Binarie - PRO")

# Sidebar
assets = ["EURUSD=X", "BTC-USD", "ETH-USD", "AAPL", "TSLA"]
selected_assets = st.sidebar.multiselect("Seleziona Asset", assets, default=["EURUSD=X", "BTC-USD"])

interval = st.sidebar.selectbox("Intervallo", ["1m", "5m"])
period = st.sidebar.selectbox("Periodo", ["7d", "30d"])
target_minutes = st.sidebar.slider("Previsione (minuti)", 1, 15, 5)
confidence_threshold = st.sidebar.slider("Confidenza minima (%)", 60, 90, 70)

@st.cache_data(ttl=60)
def get_data(ticker):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    df = df.dropna()  # Pulizia base
    return df

all_signals = []

for ticker in selected_assets:
    df = get_data(ticker)
    if len(df) < 50:
        continue
    
    data = df.copy()
    
    # Aggiungi tutti gli indicatori tecnici con pandas-ta (più stabile)
    data.ta.strategy()  # Aggiunge ~150 indicatori automaticamente
    
    # Crea il target (previsione se il prezzo sale dopo X minuti)
    future = data["Close"].shift(-target_minutes)
    data["Target"] = (future > data["Close"]).astype(int)
    data = data.dropna()
    
    # Feature columns (tutte tranne OHLCV e Target)
    feature_cols = [col for col in data.columns if col not in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Target"]]
    
    if len(feature_cols) == 0 or len(data) < 30:
        continue
    
    X = data[feature_cols]
    y = data["Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    latest = data[feature_cols].iloc[-1:].values
    prob = model.predict_proba(latest)[0]
    confidence = max(prob) * 100
    direction = "🟢 UP (CALL)" if prob[1] > 0.5 else "🔴 DOWN (PUT)"
    
    current_volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
    
    if confidence >= confidence_threshold and current_volume > 1000:  # filtro volume base
        all_signals.append({
            "Asset": ticker,
            "Direzione": direction,
            "Confidenza": f"{confidence:.1f}%",
            "Prezzo Attuale": f"{data['Close'].iloc[-1]:.4f}",
            "Volume": f"{current_volume:,.0f}"
        })

# Mostra risultati
st.subheader("🔴 Segnali Attivi (solo > soglia)")
if all_signals:
    st.success(f"✅ Trovati **{len(all_signals)} segnali forti**!")
    st.dataframe(pd.DataFrame(all_signals), use_container_width=True)
    
    # Grafico candele per il primo segnale (o tutti)
    for sig in all_signals[:2]:  # mostra max 2 grafici
        ticker = sig["Asset"]
        df_plot = get_data(ticker)
        fig = go.Figure(data=[go.Candlestick(x=df_plot.index,
                        open=df_plot['Open'],
                        high=df_plot['High'],
                        low=df_plot['Low'],
                        close=df_plot['Close'])])
        fig.update_layout(title=f"{ticker} - Ultimi {period}", height=400)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⏳ Nessun segnale sopra la soglia al momento... Prova a diminuire la confidenza o cambia asset.")

st.caption(f"Ultimo aggiornamento: {datetime.now().strftime('%H:%M:%S')}")
st.warning("⚠️ Solo a scopo educativo - Nessuna garanzia di profitti. Usa sempre conto DEMO su Pocket Option.")
