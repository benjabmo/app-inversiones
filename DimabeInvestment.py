import ssl 
# --- EL PARCHE MÁGICO ---
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
        # ESTA ES LA CLAVE: Le decimos que somos un navegador Mozilla
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        
        # 1. Hacemos la petición con los headers
        respuesta = requests.get(url, headers=headers)
        
        # 2. Leemos la tabla desde el texto de la respuesta
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
    Versión 'Full Market': Puede tardar un poco más en cargar, pero tiene toda la data.
    """
    # A. Definimos el Universo
    # LISTA IPSA EXPANDIDA (Principales acciones de Chile)
    tickers_chile = [
        'SQM-B.SN', 'CHILE.SN', 'BSANTANDER.SN', 'COPEC.SN', 'CENCOSUD.SN', 'FALABELLA.SN',
        'CMPC.SN', 'ENELAM.SN', 'VAPORES.SN', 'CAP.SN', 'ANDINA-B.SN', 'CCU.SN',
        'AGUAS-A.SN', 'BCI.SN', 'CENCOSHOPP.SN', 'COLBUN.SN', 'CONCHATORO.SN',
        'ENTEL.SN', 'IAM.SN', 'ILC.SN', 'LTM.SN', 'MALLPLAZA.SN', 'PARAUCO.SN',
        'QUINENCO.SN', 'RIPLEY.SN', 'SMU.SN', 'SONDA.SN', 'ENELCHILE.SN'
    ]
    
    # 1. Obtenemos la lista COMPLETA de Wikipedia (500+ acciones)
    tickers_usa = tickers_sp500()
    
    # 2. Aseguramos que estén los "Gigantes" y el Benchmark (SPY)
    # (A veces Wikipedia tarda en actualizar o usa nombres raros, así que aseguramos estos)
    indispensables = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'SPY', 'ASML', 'MELI', 'MSTR', 
                      'AMD', 'BTC-USD', 'ETH-USD']
    for t in indispensables:
        if t not in tickers_usa:
            tickers_usa.append(t)
    
    tickers_crypto = ['BTC-USD', 'ETH-USD']
    
    # Juntamos todo (Usamos set para eliminar duplicados si los hubiera)
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
    # Optimizamos para que no sea lento con 500 acciones
    cols_a_convertir = [col for col in data.columns if not col.endswith('.SN') and col != 'CLP=X']
    
    # Vectorización (Más rápido que un loop for normal)
    data_clp[cols_a_convertir] = data[cols_a_convertir].multiply(dolar, axis=0)
    
    # Limpieza final
    data_clp = data_clp.ffill()
    data_clp = data_clp.dropna(axis=1, how='all') # Borra las que fallaron
    
    # Filtro de Calidad: Borrar acciones que tengan menos del 90% de datos
    # (Para evitar acciones nuevas que rompan los gráficos)
    limit = len(data_clp) * 0.9
    data_clp = data_clp.dropna(axis=1, thresh=limit)

    return data_clp, dolar

def calcular_rsi(data, window=14):
    '''Calcula el Relative Strength Index (RSI) para una serie de datos.'''
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Top de Mercado', 'Análisis Técnico', 'Inversiones', 'Mi Billetera 💰', 'Noticias', 'Backtesting'])

# =====================================
# PESTAÑA 1: MONITOR DE MERCADO (CON GUÍA)
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

    # --- 2. LA GUÍA TEÓRICA (TU "TORPEDO") ---
    # Usamos st.expander para que no ocupe espacio si no lo necesitas
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
# PESTAÑA 2: 
# =====================================
    
with tab2:
    st.header('Análisis de Precios')

    # 1. Selectores
    col_sel1, col_sel2, col_sel3 = st.columns(3)

    with col_sel1:
        ## Lista de activos disponibles en la data descargada
        lista_activos = df_precios_clp.columns.tolist()
        activo_elegido = st.selectbox('Selecciona Activo:', lista_activos, 
                                      index=lista_activos.index('SQM-B.SN') if 'SQM-B.SN' in lista_activos else 0)
    
    with col_sel2:
        sma_corta = st.number_input('Media Movil Corta (Días)', value=20, min_value=5)
    
    with col_sel3:
        sma_larga = st.number_input('Media Movil Larga (Días)', value=50, min_value=10)
    
    # 2. Preparación de datos para el activo elegido
    if activo_elegido:
        ## Extraemos la serie de precios del activo
        serie_precios = df_precios_clp[activo_elegido]

        ## Calcular indicadores
        sma_s = serie_precios.rolling(window=sma_corta).mean()
        sma_l = serie_precios.rolling(window=sma_larga).mean()
        rsi = calcular_rsi(serie_precios)

        ## Precio actual y variación
        precio_actual = serie_precios.iloc[-1]
        precio_ayer = serie_precios.iloc[-2]
        delta = precio_actual - precio_ayer
        delta_pct = delta / precio_ayer

        ## Métricas grandes
        c1, c2, c3 = st.columns(3)
        c1.metric('Precio Actual (CLP)', f'${precio_actual:,.0f}',
                  f'{delta_pct:.2%}')
        c2.metric('RSI (14)', f'{rsi.iloc[-1]:.1f}', delta_color='off')

        ## Interpretación rápida del RSI
        valor_rsi = rsi.iloc[-1]
        estado_rsi = 'Neutro'
        if valor_rsi > 70: estado_rsi = 'Sobrecomprado (Posible Caída)'
        elif valor_rsi < 30: estado_rsi = 'Sobrevendido (Oportunidad)'
        c3.write(f'**Señal RSI:** {estado_rsi}')

        ## Gráfico principal (Precio + SMA)
        fig_price = px.line(serie_precios, title=f'Evolución de Precio: {activo_elegido}')
        fig_price.update_traces(line=dict(color='#1f77b4', width=3), name='Precio')

        ## Agregar las SMAs
        fig_price.add_scatter(x=serie_precios.index, y=sma_s, mode='lines', 
                              name=f'SMA {sma_corta}', line=dict(color='#ff7f0e', width=2))
        fig_price.add_scatter(x=serie_precios.index, y=sma_l, mode='lines', 
                              name=f'SMA {sma_larga}', line=dict(color='#2ca02c', width=2))
        
        fig_price.update_layout(yaxis_title='Precio en CLP', xaxis_title='Fecha')
        
        st.plotly_chart(fig_price, use_container_width=True)

        ## Gráfico secundario (RSI)
        st.markdown('#### Índice de Fuerza Relativa (RSI)')
        fig_rsi = px.line(rsi, title='Momentum (RSI)')

        ## Zonas Clave (70 y 30)
        fig_rsi.add_hline(y=70, line_dash='dash', line_color='red', 
                          annotation_text='Sobrecompra')
        fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', 
                          annotation_text='Sobreventa')
        fig_rsi.update_yaxes(range=[0, 100])

        st.plotly_chart(fig_rsi, use_container_width=True)

# =====================================
# PESTAÑA 3: ESTRATEGIA (MODO FRANCOTIRADOR 🎯)
# =====================================

with tab3:
    st.header("Tablero de Control Pro")
    st.caption("Estrategia Combinada: Busca la confluencia de RSI + Medias Móviles.")

    # --- 0. SIMBOLOGÍA ACTUALIZADA ---
    with st.expander("ℹ️ Guía de Señales (Tu Estrategia Personalizada)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("### 🦄 Compra Perfecta")
            st.success("RSI Bajo + Cruce Alcista")
            st.caption("El 'Santo Grial'. El precio estaba barato y acaba de confirmar que empieza a subir.")
        with c2:
            st.markdown("### 💎 Rebote")
            st.info("Solo RSI < 30")
            st.caption("El activo está muy barato (sobrevendido). Es buena compra, pero cuidado si sigue cayendo.")
        with c3:
            st.markdown("### 🚀 Tendencia")
            st.markdown(":blue[MA7 > MA99]")
            st.caption("Tendencia alcista fuerte a largo plazo. Ideal para mantener.")
        with c4:
            st.markdown("### 🛑 Venta")
            st.error("Cruce Bajista")
            st.caption("La media rápida (MA7) cayó bajo la media (MA25). Salir para proteger capital.")

    st.divider()

    # 1. FUNCIÓN CEREBRO (LÓGICA MEJORADA)
    def obtener_diagnostico(df):
        try:
            # Obtenemos datos
            rsi = df['RSI'].iloc[-1]
            ma7 = df['MA7'].iloc[-1]
            ma25 = df['MA25'].iloc[-1]
            ma99 = df['MA99'].iloc[-1]
            
            # --- TUS REGLAS DE ORO ---
            
            # 1. LA SEÑAL MAESTRA (COMBINACIÓN)
            # Buscamos: RSI "sano" (no carisimo) Y Cruce Alcista confirmado
            # Usamos RSI < 45 para dar margen a que ocurra el cruce
            if ma7 > ma25 and rsi < 45:
                return "🦄 COMPRA PERFECTA (GOLDEN)", f"Cruce Alcista + RSI Sano ({rsi:.0f})", "#00C853", "🌟"
            
            # 2. SEÑAL DE VENTA (ESTRICTA)
            # Si se rompe la tendencia de corto plazo, vendemos.
            elif ma7 < ma25:
                return "🛑 VENTA / SALIDA", "Cruce Bajista (MA7 cayó bajo MA25)", "#D50000", "🔻"

            # 3. COMPRA POR SUELO (SOLO RSI)
            elif rsi < 30:
                return "💎 OPORTUNIDAD DE REBOTE", f"Precio muy barato (RSI {rsi:.0f})", "#0091EA", "💎"

            # 4. TOMA DE GANANCIAS
            elif rsi > 70:
                return "⚠️ TOMA GANANCIAS", f"Precio muy caro (RSI {rsi:.0f})", "#FF6D00", "⚠️"

            # 5. TENDENCIA LARGA (SI YA COMPRASTE, MANTÉN)
            elif ma7 > ma99:
                return "🚀 MANTENER TENDENCIA", "Tendencia alcista firme", "#AA00FF", "🔭"

            else:
                return "⏸️ NEUTRO / ESPERAR", "Sin configuración clara", "#757575", "⏳"
        except:
            return "ERROR", "Datos insuficientes", "#808080", "⚪"

    # 2. CARGA DE DATOS (Mismo de antes)
    @st.cache_data(ttl=900)
    def get_single_ticker_data(symbol):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='2y', auto_adjust=False)
            if df is None or df.empty: return None
            df.columns = [c.capitalize() for c in df.columns]
            if 'Close' not in df.columns and 'Adj close' in df.columns: df['Close'] = df['Adj close']
            if 'Close' not in df.columns: return None
            df.index = df.index.tz_localize(None)
            
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA25'] = df['Close'].rolling(window=25).mean()
            df['MA99'] = df['Close'].rolling(window=99).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            return df.dropna()
        except: return None

    # 3. GRÁFICO TÉCNICO
    def plot_candle_strategy(df, symbol, title):
        if df is None or df.empty: return go.Figure()
        df_plot = df.tail(150)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Precio'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA7'], line=dict(color='orange', width=1), name='MA7 (Rápida)'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA25'], line=dict(color='blue', width=1), name='MA25 (Media)'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA99'], line=dict(color='purple', width=2, dash='dot'), name='MA99 (Larga)'))
        
        # Agregamos líneas de RSI visuales (Opcional pero útil)
        # Nota: En un gráfico de velas no se ve el RSI, así que dejamos solo las medias
        
        fig.update_layout(title=dict(text=title), height=350, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, template='plotly_white', legend=dict(orientation='h', y=1.1, x=0))
        return fig

    # ==========================================
    # VISUALIZACIÓN
    # ==========================================
    st.subheader('1. Estrategia Cripto & High Beta')
    
    tab_btc, tab_eth = st.tabs(['₿ Bitcoin', 'Ξ Ethereum'])

    # --- BITCOIN ---
    with tab_btc:
        df_btc = get_single_ticker_data('BTC-USD')
        if df_btc is not None:
            senal, explicacion, color_senal, icono = obtener_diagnostico(df_btc)
            last_price = df_btc['Close'].iloc[-1]

            # Tarjeta de Señal Grande
            st.markdown(f"""
            <div style="background-color: {color_senal}15; padding: 20px; border-radius: 12px; border: 2px solid {color_senal}; margin-bottom: 25px; text-align: center;">
                <h2 style="color: {color_senal}; margin:0;">{icono} {senal}</h2>
                <p style="margin-top:5px; font-size: 18px; color: #333;"><b>{explicacion}</b></p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("Bitcoin", f"${last_price:,.0f}")
                # MSTR Check
                df_mstr = get_single_ticker_data('MSTR')
                if df_mstr is not None and not df_mstr.empty:
                    s_mstr, _, c_mstr, i_mstr = obtener_diagnostico(df_mstr)
                    st.metric("MicroStrategy", f"${df_mstr['Close'].iloc[-1]:,.2f}")
                    st.caption(f"{i_mstr} {s_mstr}")
                else:
                    st.caption("MSTR: Cargando...")
            
            with c2:
                st.plotly_chart(plot_candle_strategy(df_btc, 'BTC-USD', 'Gráfico Bitcoin'), use_container_width=True)

    # --- ETHEREUM ---
    with tab_eth:
        df_eth = get_single_ticker_data('ETH-USD')
        if df_eth is not None:
            senal, explicacion, color_senal, icono = obtener_diagnostico(df_eth)
            
            st.markdown(f"""
            <div style="background-color: {color_senal}15; padding: 20px; border-radius: 12px; border: 2px solid {color_senal}; margin-bottom: 25px; text-align: center;">
                <h2 style="color: {color_senal}; margin:0;">{icono} {senal}</h2>
                <p style="margin-top:5px; font-size: 18px; color: #333;"><b>{explicacion}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(plot_candle_strategy(df_eth, 'ETH-USD', 'Gráfico Ethereum'), use_container_width=True)

    st.markdown("---")

    # ==========================================
    # PORTAFOLIO GENERAL
    # ==========================================
    grupos = {
        "🛡️ Núcleo": ["MSFT", "GOOGL", "AMZN", "ASML", "CHILE.SN", "QUINENCO.SN", "CENCOSUD.SN"],
        "🚀 Crecimiento": ["NVDA", "AMD", "MELI", "SQM-B.SN", "TSLA"]
    }

    col_izq, col_der = st.columns(2)

    for i, (titulo, tickers) in enumerate(grupos.items()):
        columna = col_izq if i == 0 else col_der
        with columna:
            st.subheader(titulo)
            for ticker in tickers:
                df_t = get_single_ticker_data(ticker)
                if df_t is not None:
                    precio = df_t['Close'].iloc[-1]
                    nombre = ticker.replace(".SN", " 🇨🇱")
                    senal_txt, razon_txt, color_hex, icon = obtener_diagnostico(df_t)
                    
                    with st.container():
                        k1, k2 = st.columns([1.5, 1])
                        k1.markdown(f"**{nombre}**")
                        # Badge de señal
                        k2.markdown(f"""
                        <div style="background-color:{color_hex}; color:white; padding:2px 8px; border-radius:4px; font-size:12px; text-align:center;">
                        {icon} {senal_txt.split(' ')[0]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write(f"${precio:,.0f}")
                        
                        with st.expander(f"Ver Análisis"):
                            st.write(f"**Diagnóstico:** {senal_txt}")
                            st.caption(f"**Por qué:** {razon_txt}")
                            st.plotly_chart(plot_candle_strategy(df_t, nombre, ticker), use_container_width=True)
                    st.divider()

# =====================================
# PESTAÑA 4: MI BILLETERA (CON ACCESOS DIRECTOS)
# =====================================

with tab4:
    st.header("💰 Gestión de Patrimonio")
    
    # --- 0. ACCESOS DIRECTOS (TUS PLATAFORMAS) ---
    # Aquí agregamos los botones para ir directo a comprar
    st.caption("Accesos rápidos para operar:")
    
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        # Botón para ir a Racional
        st.link_button("📱 Ir a Racional (Acciones)", "https://app.racional.cl", use_container_width=True)
    with col_link2:
        # Botón para ir a Buda
        st.link_button("⚡ Ir a Buda.com (Cripto)", "https://www.buda.com/chile", use_container_width=True)
        
    st.divider()

    # --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
    ARCHIVO_PORTAFOLIO = 'mi_portafolio.csv'

    if 'mi_portafolio' not in st.session_state:
        st.session_state['mi_portafolio'] = []
        if os.path.exists(ARCHIVO_PORTAFOLIO):
            try:
                df_guardado = pd.read_csv(ARCHIVO_PORTAFOLIO)
                st.session_state['mi_portafolio'] = df_guardado.to_dict('records')
            except:
                pass

    def guardar_en_disco():
        df_a_guardar = pd.DataFrame(st.session_state['mi_portafolio'])
        df_a_guardar.to_csv(ARCHIVO_PORTAFOLIO, index=False)

    # --- 2. CALLBACK PARA GUARDAR SIN ERRORES ---
    def agregar_inversion():
        ticker = st.session_state.input_ticker
        fecha = str(st.session_state.input_fecha)
        cant = st.session_state.input_cantidad
        costo = st.session_state.input_costo

        if cant > 0 and costo > 0:
            st.session_state['mi_portafolio'].append({
                "Fecha": fecha,
                "Ticker": ticker,
                "Cantidad": cant,
                "Inversion_USD": costo
            })
            guardar_en_disco()
            
            # Resetear inputs
            st.session_state.input_cantidad = 0.0
            st.session_state.input_costo = 0.0
            
            st.toast(f"✅ Compra de {ticker} registrada!", icon="💰")
        else:
            st.toast("⚠️ Faltan datos (Cantidad o Costo)", icon="❌")

    # --- 3. FORMULARIO ---
    with st.expander("➕ Registrar Nueva Inversión (Manual)", expanded=True):
        
        mis_acciones = [
            "ASML", "MSFT", "GOOGL", "AMZN", 
            "NVDA", "AMD", "MELI", "TSLA",   
            "BTC-USD", "ETH-USD", "MSTR",    
            "CHILE.SN", "SQM-B.SN", "QUINENCO.SN", "CENCOSUD.SN"
        ]
        
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        
        with c1:
            st.selectbox("Activo", mis_acciones, key="input_ticker")
        with c2:
            st.date_input("Fecha", datetime.now(), key="input_fecha")
        with c3:
            st.number_input("Cantidad", min_value=0.0, format="%.4f", step=0.01, key="input_cantidad")
        with c4:
            st.number_input("Gastado (USD)", min_value=0.0, format="%.2f", step=10.0, key="input_costo")
            
        st.write("")
        st.button("Guardar Inversión", on_click=agregar_inversion, use_container_width=True, type="primary")

    st.divider()

    # --- 4. VISUALIZACIÓN ---
    if len(st.session_state['mi_portafolio']) > 0:
        
        try:
            usd_obj = yf.Ticker("CLP=X").history(period="1d")
            dolar_clp = float(usd_obj['Close'].iloc[-1])
        except:
            dolar_clp = 850.0

        datos_tabla = []
        total_invertido_clp = 0
        total_actual_clp = 0
        
        copia_portafolio = st.session_state['mi_portafolio'].copy()
        copia_portafolio.reverse()
        
        if len(copia_portafolio) > 3: bar = st.progress(0)
        total_items = len(copia_portafolio)

        for i, item in enumerate(copia_portafolio):
            if len(copia_portafolio) > 3: bar.progress((i + 1) / total_items)
            
            symbol = item["Ticker"]
            qty = item["Cantidad"]
            costo_usd = item["Inversion_USD"]
            fecha_reg = item.get("Fecha", "-")
            
            valor_actual_clp = 0.0
            costo_clp = costo_usd * dolar_clp if ".SN" not in symbol else costo_usd
            
            df_live = get_single_ticker_data(symbol)
            if df_live is not None and not df_live.empty:
                try:
                    precio_hoy = float(df_live['Close'].iloc[-1])
                    if ".SN" in symbol:
                        valor_actual_clp = qty * precio_hoy
                    else:
                        valor_actual_clp = qty * precio_hoy * dolar_clp
                except:
                    pass
            
            ganancia = valor_actual_clp - costo_clp
            rentabilidad = (ganancia / costo_clp * 100) if costo_clp > 0 else 0

            total_actual_clp += valor_actual_clp
            total_invertido_clp += costo_clp
            
            datos_tabla.append({
                "Fecha": fecha_reg,
                "Activo": symbol.replace(".SN", ""),
                "Tenencia": qty,
                "Inversión (CLP)": costo_clp,
                "Valor Hoy (CLP)": valor_actual_clp,
                "Ganancia": ganancia,
                "Rentabilidad %": rentabilidad
            })
        
        if len(copia_portafolio) > 3: bar.empty()

        if datos_tabla:
            df_fin = pd.DataFrame(datos_tabla)
            
            # Métricas
            rent_total = total_actual_clp - total_invertido_clp
            pct_total = (rent_total / total_invertido_clp * 100) if total_invertido_clp > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Patrimonio Total", f"${total_actual_clp:,.0f}")
            m2.metric("Costo Histórico", f"${total_invertido_clp:,.0f}")
            m3.metric("Rentabilidad Total", f"${rent_total:,.0f}", f"{pct_total:.2f}%")
            
            st.divider()

            # Gráficos
            c_graf, c_tab = st.columns([1, 2])
            with c_graf:
                if total_actual_clp > 0:
                    fig = px.pie(df_fin, values='Valor Hoy (CLP)', names='Activo', hole=0.5)
                    fig.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
            
            with c_tab:
                df_show = df_fin.copy()
                for col in ['Inversión (CLP)', 'Valor Hoy (CLP)', 'Ganancia']:
                    df_show[col] = df_show[col].apply(lambda x: f"${x:,.0f}")
                df_show['Rentabilidad %'] = df_show['Rentabilidad %'].apply(lambda x: f"{x:+.2f}%")
                
                st.dataframe(df_show, use_container_width=True, hide_index=True)
            
            st.write("")
            with st.expander("🗑️ Borrar Historial"):
                if st.button("Eliminar Todos los Datos"):
                    st.session_state['mi_portafolio'] = []
                    guardar_en_disco()
                    st.rerun()
    else:
        st.info("👋 Agrega tu primera inversión arriba.")

# =====================================
# PESTAÑA 5: NOTICIAS (MÉTODO RSS BLINDADO)
# =====================================
import xml.etree.ElementTree as ET # Biblioteca estándar para leer RSS

with tab5:
    st.header("📰 El Diario Financiero")
    st.caption("Titulares en tiempo real vía RSS (Conexión Directa).")

    # 1. Función para leer RSS directamente (Sin usar yfinance)
    # Esto es mucho más robusto porque el formato RSS casi nunca cambia.
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
                    
                    # Limpiamos la fecha (le quitamos la hora para que sea corta)
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

