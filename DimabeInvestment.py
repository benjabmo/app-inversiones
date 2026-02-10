import ssl 
# --- PARCHE ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ------------------------

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
import requests
import numpy as np
from scipy.optimize import minimize
import io
import plotly.graph_objects as go
import os
import gspread 
import time

# 1. Configuración de la página
st.set_page_config(layout='wide', page_title='Dimabe Investments')
st.title('Gestión de Inversiones')
st.markdown('---')

# 2. Funciones de carga de datos 
@st.cache_data
def tickers_sp500():
    """Descargar la lista de S&P 500 desde Wikipedia simulando ser un navegador"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Le decimos que somos un navegador Mozilla
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        
        # 1. Hacer petición con los headers
        respuesta = requests.get(url, headers=headers)
        
        # 2. Leer tabla desde el texto de la respuesta
        html = pd.read_html(respuesta.text)
        
        df = html[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # Arreglo para BRK.B
        return tickers
    except Exception as e:
        st.error(f"Error cargando S&P 500: {e}")
        return []

@st.cache_data
def descargar_y_convertir_a_clp(start_date):
    """
    Descarga TODO el S&P 500 completo + Chile + Cripto y convierte a CLP.
    Versión 'Full Market': Se demora en cargar, pero tiene toda la data.
    """
    # A. Definir el Universo
    # LISTA IPSA (Principales acciones de Chile)
    tickers_chile = [
        'SQM-B.SN', 'CHILE.SN', 'BSANTANDER.SN', 'COPEC.SN', 'CENCOSUD.SN', 'FALABELLA.SN',
        'CMPC.SN', 'ENELAM.SN', 'VAPORES.SN', 'CAP.SN', 'ANDINA-B.SN', 'CCU.SN',
        'AGUAS-A.SN', 'BCI.SN', 'CENCOSHOPP.SN', 'COLBUN.SN', 'CONCHATORO.SN',
        'ENTEL.SN', 'IAM.SN', 'ILC.SN', 'LTM.SN', 'MALLPLAZA.SN', 'PARAUCO.SN',
        'QUINENCO.SN', 'RIPLEY.SN', 'SMU.SN', 'SONDA.SN', 'ENELCHILE.SN'
    ]
    
    # 1. Obtener lista COMPLETA de Wikipedia (500+ acciones)
    tickers_usa = tickers_sp500()
    
    # 2. Asegurar que estéb las EMPRESAS MÁS GRANDES y el Benchmark (SPY)
    indispensables = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'SPY', 'ASML', 'MELI', 'MSTR', 
                      'AMD', 'BTC-USD', 'ETH-USD']
    for t in indispensables:
        if t not in tickers_usa:
            tickers_usa.append(t)
    
    tickers_crypto = ['BTC-USD', 'ETH-USD']
    
    # Juntar todo
    todos_tickers = list(set(tickers_chile + tickers_usa + tickers_crypto + ['CLP=X']))
    
    # B. Descarga Masiva
    
    data = yf.download(todos_tickers, start=start_date)['Close']
    
    # Limpieza de zona horaria
    data.index = data.index.tz_localize(None)
    
    # C. Relleno del Dólar
    dolar = data['CLP=X']
    dolar = dolar.ffill().bfill()
    
    data_clp = data.copy()
    
    # D. Conversión Masiva a CLP
    # Optimizar para que no sea lento con 500 acciones
    cols_a_convertir = [col for col in data.columns if not col.endswith('.SN') and col != 'CLP=X']
    
    # Vectorización 
    data_clp[cols_a_convertir] = data[cols_a_convertir].multiply(dolar, axis=0)
    
    # Limpieza final
    data_clp = data_clp.ffill()
    data_clp = data_clp.dropna(axis=1, how='all') # Borra las que fallaron
    
    # Filtro de Calidad: Borrar acciones que tengan menos del 90% de datos
    limit = len(data_clp) * 0.9
    data_clp = data_clp.dropna(axis=1, thresh=limit)

    return data_clp, dolar

def calcular_rsi(data, window=14):
    """Cálculo del RSI usando el suavizado de Wilder"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 3. Sidebar y Carga
st.sidebar.header('Panel de Control')
fecha_inicio = st.sidebar.date_input('Inicio de Análisis',
pd.to_datetime('2020-01-01'))

with st.spinner("⏳ Descargando datos del Mercado..."):
    df_precios_clp, df_dolar = descargar_y_convertir_a_clp(fecha_inicio)

if 'descarga_ok' not in st.session_state:
    st.toast("Datos actualizados correctamente", icon="✅")
    st.session_state['descarga_ok'] = True

## Mostrar dólar actual
precio_dolar_hoy = df_dolar.iloc[-1]
st.sidebar.metric('Dólar Observado (Hoy)', f'${precio_dolar_hoy:,.0f} CLP')

# 4. Estructura de Pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Top de Mercado', 'Análisis Técnico', 'Inversiones', 'Mi Billetera 💰', 'Noticias'])

# =====================================
# PESTAÑA 1: MONITOR DE MERCADO 
# =====================================

with tab1:
    st.header("🌍 Visión Global de Mercado")
    st.caption("Los signos vitales de la economía hoy.")

    # --- 1. BARRA DE INDICADORES MACRO ---
    with st.container():
        macro_tickers = {
            "S&P 500 🇺🇸": "^GSPC",
            "Cobre (Futuros) ⛏️": "HG=F",
            "Oro 🥇": "GC=F",
            "Petróleo WTI 🛢️": "CL=F"
        }
        
        cols_macro = st.columns(len(macro_tickers))
        
        for i, (nombre, ticker) in enumerate(macro_tickers.items()):
            with cols_macro[i]:
                try:
                    df_macro = yf.Ticker(ticker).history(period="5d")
                    if not df_macro.empty:
                        precio_hoy = float(df_macro['Close'].iloc[-1])
                        precio_ayer = float(df_macro['Close'].iloc[-2])
                        delta_val = precio_hoy - precio_ayer
                        delta_pct = (delta_val / precio_ayer) * 100
                        
                        if "S&P" in nombre:
                            st.metric(nombre, f"{precio_hoy:,.0f}", f"{delta_pct:.2f}%")
                        else:
                            st.metric(nombre, f"${precio_hoy:,.2f}", f"{delta_pct:.2f}%")
                    else:
                        st.metric(nombre, "Sin Datos", None)
                except:
                    st.metric(nombre, "Error", None)
    
    st.write("")

    # --- 2. GUÍA TEÓRICA ---
    ## == Usar st.expander para no ocupar tanto espacio ==
    with st.expander("ℹ️ ¿Qué significa esto para mi bolsillo? (Guía Rápida)"):
        st.markdown("""
        Esta barra te dice el "clima" financiero. Úsala antes de comprar o vender:
        
        * 🇺🇸 **S&P 500 (El Jefe):** Mide la salud de las 500 empresas más grandes de EE.UU.
            * 🟢 **Si sube:** Hay optimismo. Tus acciones tecnológicas (ASML, Nvidia, Google) deberían subir.
            * 🔴 **Si baja:** Hay miedo o recesión. Todo tiende a caer.
            
        * ⛏️ **Cobre (El Sueldo de Chile):** Fundamental para nuestra economía.
            * 🟢 **Si sube:** Entran más dólares a Chile -> **El Dólar baja** (bueno para importar). Acciones como **SQM** o **Bancos Chilenos** suelen subir.
            * 🔴 **Si baja:** El Dólar suele dispararse (malo para comprar cosas afuera).
            
        * 🥇 **Oro (El Refugio):** La gente compra oro cuando tiene miedo.
            * 🟢 **Si sube mucho:** Significa que los inversores están asustados (guerras, crisis). Es señal de precaución.
            
        * 🛢️ **Petróleo (La Inflación):**
            * 🟢 **Si sube:** Aumenta el costo de transporte y energía. Puede generar inflación y hacer que las acciones caigan a largo plazo.
        """)

    st.divider()

    # --- 3. RANKING DE ACCIONES ---
    st.subheader("🏆 Ranking: Ganadores y Perdedores")
    
    col1, col2 = st.columns(2)
    
    tickers_ranking = [
        "ASML", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "MELI", 
        "SQM", "CHILE.SN", "BSANTANDER.SN", "CENCOSUD.SN", "QUINENCO.SN", 
        "BTC-USD", "ETH-USD", "MSTR"
    ]

    with st.spinner("Escaneando precios..."):
        ranking_data = []
        for t in tickers_ranking:
            try:
                ticker_obj = yf.Ticker(t)
                hist = ticker_obj.history(period="2d")
                
                if len(hist) >= 2:
                    hoy = float(hist['Close'].iloc[-1])
                    ayer = float(hist['Close'].iloc[-2])
                    var = ((hoy - ayer) / ayer) * 100
                    
                    ranking_data.append({
                        "Ticker": t.replace(".SN", " 🇨🇱"), 
                        "Precio": hoy,
                        "Variación %": var
                    })
            except:
                pass
        
        if ranking_data:
            df_rank = pd.DataFrame(ranking_data)
            
            top_gainers = df_rank.sort_values("Variación %", ascending=False).head(5)
            top_losers = df_rank.sort_values("Variación %", ascending=True).head(5)
            
            with col1:
                st.success("🚀 Top 5 Subidas")
                for _, row in top_gainers.iterrows():
                    st.markdown(f"**{row['Ticker']}**: :green[+{row['Variación %']:.2f}%] (${row['Precio']:,.2f})")
            
            with col2:
                st.error("📉 Top 5 Caídas")
                for _, row in top_losers.iterrows():
                    st.markdown(f"**{row['Ticker']}**: :red[{row['Variación %']:.2f}%] (${row['Precio']:,.2f})")
        else:
            st.warning("No hay datos de mercado disponibles ahora.")

# =====================================
# PESTAÑA 2: ANÁLISIS TÉCNICO
# =====================================
    
with tab2:
    st.header('Análisis de Precios')

 # 1. Selectores Pro
    col_s1, col_s2, col_s3 = st.columns([2,1,1])
    with col_s1:
        activo = st.selectbox('Activo a analizar:', df_precios_clp.columns.tolist(), index=0)
    with col_s2:
        temporalidad = st.radio('Temporalidad:', ['Diario', 'Semanal'], horizontal=True)
    with col_s3:
        tipo_rsi = st.caption("RSI: Estándar Wilder 14")

    # 2. Procesamiento de datos
    df_t = df_precios_clp[activo].to_frame(name='Close')
    if temporalidad == 'Semanal':
        df_t = df_t.resample('W').last()

    # Indicadores Institucionales
    df_t['EMA21'] = df_t['Close'].ewm(span=21, adjust=False).mean()
    df_t['SMA50'] = df_t['Close'].rolling(window=50).mean()
    df_t['SMA200'] = df_t['Close'].rolling(window=200).mean()
    df_t['RSI'] = calcular_rsi(df_t['Close'])

    # 3. Métricas
    m1, m2, m3, m4 = st.columns(4)
    precio_act = df_t['Close'].iloc[-1]
    rsi_act = df_t['RSI'].iloc[-1]
    
    m1.metric("Precio Actual", f"${precio_act:,.0f}")
    m2.metric("RSI (14)", f"{rsi_act:.1f}", delta="SOBRECOMPRA" if rsi_act > 70 else "SOBREVENTA" if rsi_act < 30 else "NEUTRO")
    
    # Señal de Medias
    if precio_act > df_t['SMA50'].iloc[-1]:
        m3.write("✅ **Tendencia:** Alcista (>SMA50)")
    else:
        m3.write("❌ **Tendencia:** Bajista (<SMA50)")
        
    # 4. Gráfico de Precio (Candelas o Línea)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t.index, y=df_t['Close'], name='Precio', line=dict(color='#17BECF', width=2)))
    fig.add_trace(go.Scatter(x=df_t.index, y=df_t['EMA21'], name='EMA 21 (Rápida)', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df_t.index, y=df_t['SMA50'], name='SMA 50 (Media)', line=dict(color='magenta', width=1)))
    fig.add_trace(go.Scatter(x=df_t.index, y=df_t['SMA200'], name='SMA 200 (Lenta)', line=dict(color='green', width=1.5)))
    
    fig.update_layout(title=f'Gráfico {temporalidad}: {activo}', height=500, template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # 5. Gráfico RSI
    fig_rsi = px.line(df_t, y='RSI', title='Momentum RSI (Wilder)')
    fig_rsi.add_hline(y=70, line_dash='dash', line_color='red')
    fig_rsi.add_hline(y=30, line_dash='dash', line_color='green')
    fig_rsi.update_layout(height=250, template='plotly_dark', yaxis_range=[0,100])
    st.plotly_chart(fig_rsi, use_container_width=True)

# =====================================
# PESTAÑA 3: ESTRATEGIA (CON RSI INTEGRADO Y SIN ERRORES)
# =====================================

with tab3:
    st.header("Seguimiento de Acciones Principales")
    st.caption("Estrategia Semanal Sincronizada: RSI Wilder + Medias Institucionales (21/50/200).")

    # --- 0. SIMBOLOGÍA ACTUALIZADA ---
    with st.expander("ℹ️ Guía de Señales", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown("### 🏅 Compra Perfecta")
            st.success("RSI Sano + Sobre MA200")
            st.caption("El activo está barato pero la tendencia de largo plazo sigue siendo alcista.")
        with c2:
            st.markdown("### 💎 Rebote")
            st.info("Solo RSI < 30")
            st.caption("Sobrevendido. Oportunidad de rebote técnico, pero con mayor riesgo.")
        with c3:
            st.markdown("### 🚀 Tendencia")
            st.markdown(":blue[MA21 > MA50]")
            st.caption("Confirmación de impulso alcista en el mediano plazo.")
        with c4:
            st.markdown('### ⚠️ TOMA GANANCIAS')
            st.warning('Solo RSI > 75')
            st.caption('Sobrecomprado. Vender las acciones, porque se puede venir una bajada. TENER OJO CON EL VOLUMEN')
        with c5:
            st.markdown("### 🛑 Venta")
            st.error("Cruce Bajista")
            st.caption("La media rápida (MA21) cayó bajo la media (MA50). Salida de seguridad.")

    st.divider()

    # 1. FUNCIÓN CEREBRO (Sincronizada con TradingView)
    def obtener_diagnostico(df):
        try:
            if len(df) < 20: return "ESPERAR", "Faltan datos", "#757575", "⏳"
            
            rsi = df['RSI'].iloc[-1]
            ma21 = df['MA21'].iloc[-1]
            ma50 = df['MA50'].iloc[-1]
            ma200 = df['MA200'].iloc[-1] if len(df) >= 200 else df['MA50'].iloc[-1]
            precio = df['Close'].iloc[-1]
            
            if ma21 > ma50 and rsi < 45 and precio > ma200:
                return "🏅 COMPRA PERFECTA", f"Cruce Alcista Semanal + RSI Sano ({rsi:.1f})", "#00C853", "🌟"
            elif ma21 < ma50:
                return "🛑 VENTA / SALIDA", f"Cruce Bajista (MA21 < MA50). RSI: {rsi:.1f}", "#D50000", "🔻"
            elif rsi < 30:
                return "💎 REBOTE TÉCNICO", f"Sobreventa Extrema (RSI {rsi:.1f})", "#0091EA", "💎"
            elif rsi > 75:
                return "⚠️ TOMA GANANCIAS", f"Sobrecompra (RSI {rsi:.1f})", "#FF6D00", "⚠️"
            elif precio > ma200:
                return "🚀 MANTENER", "Tendencia alcista firme sobre MA200", "#AA00FF", "🔭"
            else:
                return "⏸️ NEUTRO", "Sin señal clara", "#757575", "⏳"
        except:
            return "ERROR", "Fallo en cálculos", "#808080", "⚪"

    # 2. CARGA DE DATOS (Mejora: Remuestreo Semanal para evitar errores de índice)
    @st.cache_data(ttl=900)
    def get_single_ticker_data(symbol):
        try:
            ticker = yf.Ticker(symbol)
            # Traemos 5 años para asegurar tener datos de la MA200
            df_raw = ticker.history(period='5y', auto_adjust=False)
            if df_raw.empty: return None
            
            df_raw.columns = [c.capitalize() for c in df_raw.columns]
            df_raw.index = df_raw.index.tz_localize(None)
            
            # CONVERSIÓN A SEMANAL
            df = df_raw['Close'].resample('W').last().to_frame()
            df['Open'] = df_raw['Open'].resample('W').first()
            df['High'] = df_raw['High'].resample('W').max()
            df['Low'] = df_raw['Low'].resample('W').min()
            
            # Medias Institucionales
            df['MA21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            
            # RSI WILDER 
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df.dropna(subset=['RSI', 'MA21'])
        except: return None

    # 3. GRÁFICO TÉCNICO 
    def plot_candle_strategy(df, symbol, title):
        if df is None or df.empty: return go.Figure()
        df_plot = df.tail(100)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA21'], line=dict(color='orange', width=1.5), name='EMA 21'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], line=dict(color='blue', width=1.5), name='SMA 50'))
        if len(df) >= 200:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA200'], line=dict(color='purple', width=2, dash='dot'), name='SMA 200'))
        
        fig.update_layout(height=400, template='plotly_dark', xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0))
        return fig

    # ==========================================
    # VISUALIZACIÓN
    # ==========================================
    st.subheader('1. Estrategia Cripto')
    t1, t2 = st.tabs(['₿ Bitcoin', 'Ξ Ethereum'])

    with t1:
        df_btc = get_single_ticker_data('BTC-USD')
        if df_btc is not None:
            diag, expl, col_s, ico = obtener_diagnostico(df_btc)
            # Sincronizamos con el Dólar de la App para CLP
            p_clp = df_btc['Close'].iloc[-1]
            st.markdown(f"<div style='background-color:{col_s}15; padding:20px; border-radius:12px; border:2px solid {col_s}; text-align:center;'><h2 style='color:{col_s}; margin:0;'>{ico} {diag}</h2><p><b>{expl}</b></p></div>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Bitcoin", f"${p_clp:,.0f} USD")
                st.metric("RSI Semanal", f"{df_btc['RSI'].iloc[-1]:.1f}")
                # MicroStrategy Check
                df_mstr = get_single_ticker_data('MSTR')
                if df_mstr is not None:
                    s_mstr, _, c_mstr, i_mstr = obtener_diagnostico(df_mstr)
                    st.caption(f"MSTR: {i_mstr} {s_mstr}")
            with c2:
                st.plotly_chart(plot_candle_strategy(df_btc, 'BTC', 'BTC Semanal'), use_container_width=True)

    with t2:
        df_eth = get_single_ticker_data('ETH-USD')
        if df_eth is not None:
            diag, expl, col_s, ico = obtener_diagnostico(df_eth)
            p_clp = df_eth['Close'].iloc[-1] * df_dolar.iloc[-1]
            st.markdown(f"<div style='background-color:{col_s}15; padding:20px; border-radius:12px; border:2px solid {col_s}; text-align:center;'><h2 style='color:{col_s}; margin:0;'>{ico} {diag}</h2><p><b>{expl}</b></p></div>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Ethereum (CLP)", f"${p_clp:,.0f}")
                st.metric("RSI Semanal", f"{df_eth['RSI'].iloc[-1]:.1f}")
            with c2:
                st.plotly_chart(plot_candle_strategy(df_eth, 'ETH', 'ETH Semanal'), use_container_width=True)

    st.markdown("---")

    # ==========================================
    # PORTAFOLIO GENERAL
    # ==========================================
    grupos = {
        "MERCADO GLOBAL 🌎": ["MSFT", "GOOGL", "AMZN", "ASML", 'NVDA', 'AMD', 'MELI', 'TSLA'],
        "MERCADO CHILENO 🇨🇱": ["SQM-B.SN", "CHILE.SN", "QUINENCO.SN", "CENCOSUD.SN", 'LTM.SN', 'VAPORES.SN']
    }

    col_izq, col_der = st.columns(2)
    for i, (titulo, tks) in enumerate(grupos.items()):
        columna = col_izq if i == 0 else col_der
        with columna:
            st.subheader(titulo)
            for t in tks:
                data = get_single_ticker_data(t)
                if data is not None and not data.empty:
                    # Determinación de Moneda y Formato
                    es_chile = t.endswith('.SN')
                    moneda = 'CLP' if es_chile else 'USD'
                    simbolo = '$'
                    # Convertir a CLP si es USA para que coincida con Pestaña 2
                    p_val = data['Close'].iloc[-1]
                    rsi_v = data['RSI'].iloc[-1]
                    s_txt, r_txt, c_hex, icon = obtener_diagnostico(data)
                    
                    with st.container():
                        k1, k2 = st.columns([1.5, 1])
                        k1.markdown(f"**{t.replace('.SN','')}**")
                        k2.markdown(f"<div style='background-color:{c_hex}; color:white; padding:2px; border-radius:4px; font-size:12px; text-align:center;'>{icon} {s_txt.split(' ')[0]}</div>", unsafe_allow_html=True)
                        
                        r_p, r_r = st.columns(2)
                        if es_chile:
                            r_p.write(f'Precio: {simbolo}{p_val:,.0f} {moneda}')
                        else:
                            r_p.write(f'Precio: {simbolo}{p_val:,.2f} {moneda}')
                        
                        r_r.write(f'RSI: **{rsi_v:.1f}**')
                        
                        with st.expander("Ver Análisis"):
                            st.write(f"**Diagnóstico:** {s_txt}")
                            st.caption(f"**Por qué:** {r_txt}")
                            st.plotly_chart(plot_candle_strategy(data, t, t), use_container_width=True)
                    st.divider()

# =====================================
# PESTAÑA 4: BILLETERA PRO
# =====================================
with tab4:
    st.header("☁️ Gestión de Patrimonio (Base de Datos Real)")
    
    col_link1, col_link2 = st.columns(2)
    col_link1.link_button("📱 Racional", "https://app.racional.cl", use_container_width=True)
    col_link2.link_button("⚡ Buda.com", "https://www.buda.com/chile", use_container_width=True)
    st.divider()

    # 1. CONEXIÓN A GOOGLE SHEETS
    def conectar_google_sheets():
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            gc = gspread.service_account_from_dict(creds_dict)
            sh = gc.open("Base_Datos_Dimabe") 
            return sh.sheet1
        except Exception as e:
            st.error(f"⚠️ Error de conexión: {e}")
            return None

    sheet = conectar_google_sheets()
    df_nube = pd.DataFrame()

    # Función de limpieza de números 
    def limpiar_numero_estricto(valor):
        texto = str(valor).strip()
        if not texto: return 0.0
        # Reemplazamos coma por punto para Python
        texto = texto.replace(',', '.')
        try:
            return float(texto)
        except:
            return 0.0

    # 2. CARGA DE DATOS 
    if sheet:
        try:
            # Bajamos todo como texto
            data_raw = sheet.get_all_values()
            
            if len(data_raw) > 1: 
                # Asumir orden estricto de columnas:
                # Col 0: Fecha | Col 1: Ticker | Col 2: Cantidad | Col 3: Inversion_USD
                headers = ["Fecha", "Ticker", "Cantidad", "Inversion_USD"]
                rows = data_raw[1:] # Se salta la fila de titulos puesta en el Excel
                
                # Se crea el DataFrame forzando los nombres puestos en las columnas
                df_nube = pd.DataFrame(rows)
                
                # Asegurar que se tenga 4 columnas, si tiene menos, se rellena
                if df_nube.shape[1] >= 4:
                    df_nube = df_nube.iloc[:, :4] # Se queda con las primeras 4
                    df_nube.columns = headers
                    
                    # Limpieza numérica
                    df_nube['Cantidad'] = df_nube['Cantidad'].apply(limpiar_numero_estricto)
                    df_nube['Inversion_USD'] = df_nube['Inversion_USD'].apply(limpiar_numero_estricto)
                else:
                    df_nube = pd.DataFrame() # Estructura incorrecta
            else:
                df_nube = pd.DataFrame()
                
        except Exception as e:
            st.warning(f"Esperando datos... ({e})")

    # 3. SECCIÓN DE GESTIÓN 
    c_add, c_del = st.columns([2, 1])
    
    # --- COLUMNA IZQUIERDA: AGREGAR ---
    with c_add:
        with st.expander("➕ Registrar Inversión", expanded=True):
            with st.form("entry_form", clear_on_submit=True):
                mis_acciones = ["ASML", "MSFT", "GOOGL", "AMZN", "NVDA", "AMD", "MELI", "TSLA", "BTC-USD", "ETH-USD", "MSTR", 
                                "CHILE.SN", "SQM-B.SN", "QUINENCO.SN", "CENCOSUD.SN", 'CFMITNIPSA.SN', 'LTM.SN', 'VAPORES.SN']
                
                k1, k2 = st.columns(2)
                with k1: 
                    tick = st.selectbox("Activo", mis_acciones)
                    cant = st.number_input("Cantidad", min_value=0.0, format="%.6f", step=0.0001)
                with k2: 
                    fech = st.date_input("Fecha", datetime.now())
                    cost = st.number_input("Total Pagado (USD)", min_value=0.0, format="%.2f", step=10.0)
                
                st.write("")
                if st.form_submit_button("Guardar en Nube", type="primary", use_container_width=True):
                    if sheet and cant > 0 and cost > 0:
                        try:
                            # Guardar con formato string seguro
                            row = [str(fech), tick, str(cant).replace('.', ','), str(cost).replace('.', ',')]
                            sheet.append_row(row)
                            st.toast("✅ Guardado")
                            import time
                            time.sleep(1)
                            st.rerun()
                        except: st.error("Error guardando")

    # --- COLUMNA DERECHA: BORRAR ---
    with c_del:
        with st.expander("🗑️ Borrar", expanded=True):
            if not df_nube.empty:
                # Se crea una lista para elegir que acción borrar
                # Formato: "Fila X: Ticker (Fecha)"
                opciones_borrar = [f"{i}: {row['Ticker']} ({row['Fecha']})" for i, row in df_nube.iterrows()]
                seleccion = st.selectbox("Elegir:", options=opciones_borrar)
                
                if st.button("Eliminar Fila", type="secondary", use_container_width=True):
                    if seleccion:
                        # Extraemos el índice original (El número antes de los dos puntos)
                        indice_df = int(seleccion.split(':')[0])
                        # En Google Sheets, la fila es indice_df + 2 (porque empieza en 1 y la fila 1 es header)
                        fila_sheet = indice_df + 2
                        
                        try:
                            sheet.delete_rows(fila_sheet)
                            st.toast("🗑️ Eliminado")
                            import time
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("Nada que borrar")

    st.divider()

    # 4. VISUALIZACIÓN (TABLA Y GRÁFICO)
    if not df_nube.empty:
        st.subheader("📊 Portafolio en la Nube")
        
        # --- CÁLCULOS ---
        try:
            usd_val = yf.Ticker("CLP=X").history(period="1d")['Close'].iloc[-1]
        except: usd_val = 850.0

        datos_calc = []
        total_inv_clp = 0 # Lo que gaste (convertido a CLP aprox)
        total_val_clp = 0 # Lo que tengo hoy
        
        for index, row in df_nube.iterrows():
            sym = row['Ticker']
            qty = row['Cantidad']
            cost_usd = row['Inversion_USD'] 
            
            # 1. Costo histórico estimado en CLP
            cost_clp = cost_usd * usd_val if ".SN" not in sym else cost_usd
            
            # 2. Valor actual en vivo
            val_now = 0
            try:
                live = yf.Ticker(sym).history(period="1d")
                if not live.empty:
                    p = live['Close'].iloc[-1]
                    if ".SN" in sym:
                        val_now = qty * p
                    else:
                        val_now = qty * p * usd_val
            except: pass
            
            total_inv_clp += cost_clp
            total_val_clp += val_now
            
            ganancia = val_now - cost_clp
            rent = (ganancia / cost_clp * 100) if cost_clp > 0 else 0
            
            datos_calc.append({
                "Fecha": row['Fecha'], 
                "Activo": sym.replace(".SN", ""), 
                "Tenencia": qty,
                "Costo Orig (USD)": cost_usd, 
                "Valor Hoy (CLP)": val_now,
                "Ganancia": ganancia, 
                "Rent %": rent
            })
            
        # --- MÉTRICAS ---
        rent_tot = total_val_clp - total_inv_clp
        pct_tot = (rent_tot/total_inv_clp*100) if total_inv_clp > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Patrimonio Actual", f"${total_val_clp:,.0f}")
        m2.metric("Inversión Total (Est.)", f"${total_inv_clp:,.0f}") 
        m3.metric("Rentabilidad", f"${rent_tot:,.0f}", f"{pct_tot:.2f}%")
        
        st.divider()

        # --- GRÁFICOS Y TABLA DETALLADA ---
        df_fin = pd.DataFrame(datos_calc)
        
        c_graf, c_tab = st.columns([1, 2])
        
        with c_graf:
            st.caption("Composición")
            if total_val_clp > 0:
                fig = px.pie(df_fin, values='Valor Hoy (CLP)', names='Activo', hole=0.5)
                fig.update_layout(showlegend=False, margin=dict(t=20, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
        
        with c_tab:
            st.caption("Detalle")
            df_show = df_fin.copy()
            # Formato de dinero
            df_show['Valor Hoy (CLP)'] = df_show['Valor Hoy (CLP)'].apply(lambda x: f"${x:,.0f}")
            df_show['Ganancia'] = df_show['Ganancia'].apply(lambda x: f"${x:,.0f}")
            df_show['Costo Orig (USD)'] = df_show['Costo Orig (USD)'].apply(lambda x: f"${x:,.2f}") 
            df_show['Rent %'] = df_show['Rent %'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(
                df_show[['Fecha', 'Activo', 'Tenencia', 'Costo Orig (USD)', 'Valor Hoy (CLP)', 'Ganancia', 'Rent %']], 
                use_container_width=True, 
                hide_index=True
            )
            
    else:
        st.info("✅ Conexión exitosa. La hoja está vacía.")

# =====================================
# PESTAÑA 5: NOTICIAS 
# =====================================
import xml.etree.ElementTree as ET # Biblioteca estándar para leer RSS

with tab5:
    st.header("📰 El Diario Financiero")
    st.caption("Titulares en tiempo real vía RSS (Conexión Directa).")

    # 1. Función para leer RSS directamente (Sin usar yfinance)
    def obtener_noticias_rss(ticker):
        # URL oficial del Feed de Yahoo Finance
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        
        try:
            # Usamos requests con un 'User-Agent' para que no nos bloqueen
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                lista_noticias = []
                
                # Buscamos los artículos dentro del XML
                # Limitamos a 3 noticias para no saturar
                for item in root.findall('.//item')[:3]:
                    titulo = item.find('title').text
                    link = item.find('link').text
                    pub_date = item.find('pubDate').text
                    
                    # Limpieza de fechas
                    if pub_date:
                        pub_date = pub_date[:16] 
                        
                    lista_noticias.append({
                        'titulo': titulo,
                        'link': link,
                        'fecha': pub_date
                    })
                return lista_noticias
            else:
                return []
        except:
            return []

    # 2. Definición de Fuentes
    fuentes_noticias = {
        "🇺🇸 Wall Street & S&P 500": "SPY",    
        "₿ Cripto & Bitcoin": "BTC-USD",          
        "🤖 Tecnología & IA": "NVDA",  
        "🇨🇱 Mercado Chileno & Litio": "SQM"    
    }

    col_news1, col_news2 = st.columns(2)

    for i, (titulo_seccion, ticker_clave) in enumerate(fuentes_noticias.items()):
        
        columna_actual = col_news1 if i % 2 == 0 else col_news2
        
        with columna_actual:
            st.subheader(titulo_seccion)
            
            # Llamamos a la nueva función RSS
            noticias = obtener_noticias_rss(ticker_clave)
            
            if noticias:
                for n in noticias:
                    # Tarjeta visual limpia
                    st.markdown(f"""
                    <div style="
                        padding: 12px; 
                        border-radius: 8px; 
                        border: 1px solid #ddd; 
                        margin-bottom: 12px;
                        background-color: #ffffff;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <a href="{n['link']}" target="_blank" style="text-decoration: none; color: #0066cc; font-weight: 600; font-size: 15px; display: block; margin-bottom: 4px;">
                            {n['titulo']}
                        </a>
                        <div style="font-size: 11px; color: #888;">
                            📅 {n['fecha']} • Fuente: Yahoo RSS
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"Sin noticias recientes para {ticker_clave}")
            
            st.write("")

