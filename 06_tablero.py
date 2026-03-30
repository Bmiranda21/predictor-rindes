# ============================================================
#  TABLERO — PREDICTOR DE RINDE SOJA MULTI-ADR (GIS Ready)
#  Correr con: streamlit run 06_tablero.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictor de Rinde Multi-ADR",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────
#  ESTILOS
# ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #0f1117; }
    .kpi-card { background: linear-gradient(135deg, #1a1d2e 0%, #16192a 100%); border: 1px solid #2a2d3e; border-radius: 12px; padding: 20px 24px; text-align: center; transition: border-color 0.2s; }
    .kpi-card:hover { border-color: #4f6ef7; }
    .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #6b7280; margin-bottom: 8px; }
    .kpi-value { font-family: 'DM Mono', monospace; font-size: 32px; font-weight: 500; color: #f0f0f0; line-height: 1; }
    .kpi-value-big { font-family: 'DM Mono', monospace; font-size: 48px; font-weight: 700; line-height: 1; }
    .kpi-sub { font-size: 12px; color: #6b7280; margin-top: 6px; }
    .kpi-delta-pos { color: #34d399; font-size: 13px; font-weight: 600; }
    .kpi-delta-neg { color: #f87171; font-size: 13px; font-weight: 600; }
    .kpi-delta-neu { color: #94a3b8; font-size: 13px; font-weight: 600; }
    .section-title { font-size: 13px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #4f6ef7; margin: 32px 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #1e2130; }
    .update-badge { background: #1a1d2e; border: 1px solid #2a2d3e; border-radius: 20px; padding: 4px 14px; font-size: 11px; color: #6b7280; font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
#  CONSTANTES Y CARGA RÁPIDA
# ────────────────────────────────────────────────────────────

DATASET_PATH = "dataset_modelo.csv"
PRED_PATH = "prediccion_2025_26.csv"
METRICAS_PATH = "metricas_por_adr.csv"
GEOJSON_PATH = "adrs.geojson"

FEATURES = [
    "pp_total_mm", "pp_siembra_mm", "pp_floracion_mm", "pp_llenado_mm",
    "tmed_campaña", "tmax_floracion", "tmin_llenado",
    "anomalia_pp", "ratio_pp_floracion", "estres_termico", "tendencia"
]
TARGET = "rinde_kgha"

@st.cache_data(ttl=3600)
def cargar_datos():
    df_hist = pd.read_csv(DATASET_PATH)
    df_pred = pd.read_csv(PRED_PATH)
    df_met = pd.read_csv(METRICAS_PATH)
    with open(GEOJSON_PATH, encoding="utf-8") as f:
        geojson = json.load(f)
    return df_hist, df_pred, df_met, geojson

df, df_predicciones, df_metricas, geojson_adrs = cargar_datos()
lista_adrs = sorted(df["nombre_adr"].unique())

def validar_modelo_adr(df_adr):
    resultados = []
    for i in range(len(df_adr)):
        idx_train = [j for j in range(len(df_adr)) if j != i]
        X_tr = df_adr[FEATURES].iloc[idx_train].values
        y_tr = df_adr[TARGET].iloc[idx_train].values
        X_te = df_adr[FEATURES].iloc[[i]].values
        y_te = df_adr[TARGET].iloc[i]
        
        m = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)[0]
        
        resultados.append({
            "campaña": df_adr.iloc[i]["campaña"],
            "real_qqha": round(y_te/100, 1),
            "pred_qqha": round(pred/100, 1),
            "error_qqha": round((pred - y_te)/100, 1),
        })
    return pd.DataFrame(resultados)

# ────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📍 Selector de Zona")
    adr_seleccionado = st.selectbox("Elegí el ADR a analizar:", lista_adrs, index=lista_adrs.index("TANDIL") if "TANDIL" in lista_adrs else 0)
    
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    anio_campaña = 2025
    campaña_label = "2025/26"
    st.markdown(f"**Campaña activa:** `{campaña_label}`")
    
    actualizar = st.button("🔄 Recargar datos", use_container_width=True, type="primary")
    if actualizar:
        st.cache_data.clear()
        st.rerun()
        
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:11px; color:#6b7280;'>
    <b>Modelo:</b> XGBoost por ADR<br>
    <b>Fuentes:</b> NASA POWER<br>
    <b>GIS:</b> Custom GeoJSON (ADRs)<br><br>
    <span class='update-badge'>v2.5 · Ceres Tolvas</span>
    </div>
    """, unsafe_allow_html=True)

# Filtramos la data para el ADR seleccionado
df_adr = df[df["nombre_adr"] == adr_seleccionado].sort_values("anio_inicio").reset_index(drop=True)
prom_hist = df_adr["rinde_kgha"].mean() / 100
datos_pred = df_predicciones[df_predicciones["nombre_adr"] == adr_seleccionado].iloc[0]
pred_qqha = datos_pred["rinde_pred_qqha"]
dif_qqha = pred_qqha - prom_hist

# ────────────────────────────────────────────────────────────
#  HEADER Y TABS
# ────────────────────────────────────────────────────────────

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
    <h1 style='font-family: DM Sans; font-size: 28px; font-weight: 700; color: #f0f0f0; margin: 0; padding: 0;'>
        🌱 Predictor de Rinde Soja — ADR {adr_seleccionado}
    </h1>
    <p style='color: #6b7280; font-size: 13px; margin: 4px 0 0 0;'>
        Campaña {campaña_label} &nbsp;·&nbsp; Análisis zonal ponderado
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right; font-size:11px; color:#6b7280; margin-top:8px;'>
        🟢 NASA POWER OK &nbsp;·&nbsp; 🗺️ GIS OK<br>
        Datos históricos: 2000 - 2024
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2 = st.tabs(["📊 Tablero Interactivo", "🧠 Metodología y Backstage"])

# =====================================================================
# PESTAÑA 1: EL TABLERO
# =====================================================================
with tab1:
    # SECCIÓN 1 — MAPA ESTRATÉGICO
    st.markdown("<div class='section-title'>MAPA DE ESTIMACIONES REGIONALES (ADRs)</div>", unsafe_allow_html=True)
    
    # Mapa de coropletas usando GeoJSON
    fig_mapa = px.choropleth_mapbox(
        df_predicciones,
        geojson=geojson_adrs,
        locations="nombre_adr",
        featureidkey="properties.nombre_adr", # Ajustar si el campo en el geojson tiene otro nombre
        color="rinde_pred_qqha",
        color_continuous_scale="RdYlGn",
        mapbox_style="carto-darkmatter",
        zoom=5.8,
        center={"lat": -37.5, "lon": -60.0},
        opacity=0.5,
        labels={'rinde_pred_qqha': 'Rinde (qq/ha)'},
        hover_data={"nombre_adr": True, "rinde_pred_qqha": ":.1f", "pp_acumulada_mm": ":.0f"}
    )
    fig_mapa.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#6b7280", size=10),
        coloraxis_colorbar=dict(title="qq/ha", thickness=15, len=0.5)
    )
    st.plotly_chart(fig_mapa, use_container_width=True)

    # SECCIÓN 2 — KPIs
    st.markdown("<div class='section-title'>ESTADÍSTICAS SELECCIONADAS</div>", unsafe_allow_html=True)
    
    color_pred  = "#34d399" if dif_qqha > 1 else "#f87171" if dif_qqha < -1 else "#f0f0f0"
    delta_class = "kpi-delta-pos" if dif_qqha > 1 else "kpi-delta-neg" if dif_qqha < -1 else "kpi-delta-neu"
    delta_icon  = "▲" if dif_qqha > 1 else "▼" if dif_qqha < -1 else "●"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card' style='border-color: {color_pred}40;'><div class='kpi-label'>Rinde Estimado ({adr_seleccionado})</div><div class='kpi-value-big' style='color: {color_pred};'>{pred_qqha:.1f}</div><div class='kpi-sub'>qq/ha</div><div class='{delta_class}' style='margin-top:8px;'>{delta_icon} {dif_qqha:+.1f} vs promedio</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Promedio Histórico Zonal</div><div class='kpi-value'>{prom_hist:.1f}</div><div class='kpi-sub'>qq/ha · (24 campañas)</div></div>", unsafe_allow_html=True)
    with c3:
        mejor = df_adr.loc[df_adr["rinde_kgha"].idxmax()]
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Mejor Campaña Histórica</div><div class='kpi-value' style='color:#34d399;'>{mejor['rinde_kgha']/100:.1f}</div><div class='kpi-sub'>qq/ha · {mejor['campaña']}</div></div>", unsafe_allow_html=True)
    with c4:
        peor = df_adr.loc[df_adr["rinde_kgha"].idxmin()]
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Peor Campaña Histórica</div><div class='kpi-value' style='color:#f87171;'>{peor['rinde_kgha']/100:.1f}</div><div class='kpi-sub'>qq/ha · {peor['campaña']}</div></div>", unsafe_allow_html=True)

    # SECCIÓN 3 — CLIMA Y GRÁFICO
    col_g1, col_g2 = st.columns([1, 2])
    
    with col_g1:
        st.markdown(f"<div class='section-title'>CLIMA ({adr_seleccionado})</div>", unsafe_allow_html=True)
        pp_total = datos_pred["pp_acumulada_mm"]
        anomalia = datos_pred["anomalia_mm"]
        anom_color = "#34d399" if anomalia > 20 else "#f87171" if anomalia < -50 else "#94a3b8"
        
        st.markdown(f"<div class='kpi-card' style='margin-bottom:15px;'><div class='kpi-label'>Lluvia Acumulada</div><div class='kpi-value'>{pp_total:.0f}</div><div class='kpi-sub'>mm estim. 25/26</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-card' style='border-color:{anom_color}40;'><div class='kpi-label'>Anomalía Lluvia</div><div class='kpi-value' style='color:{anom_color};'>{anomalia:+.0f}</div><div class='kpi-sub'>mm vs histórico</div></div>", unsafe_allow_html=True)
        
    with col_g2:
        st.markdown("<div class='section-title'>HISTORIAL DE RINDES</div>", unsafe_allow_html=True)
        campañas_graf = list(df_adr["campaña"]) + [campaña_label]
        rindes_graf   = list(df_adr["rinde_kgha"] / 100) + [pred_qqha]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=campañas_graf[:-1], y=rindes_graf[:-1], name="Histórico", marker_color="#3b82f6", marker_opacity=0.6))
        fig.add_trace(go.Bar(x=[campaña_label], y=[pred_qqha], name="Predicción", marker_color="#f97316"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", height=300, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    # SECCIÓN 4 — VALIDACIÓN
    st.markdown("<div class='section-title'>VALIDACIÓN DEL MODELO</div>", unsafe_allow_html=True)
    df_val_adr = validar_modelo_adr(df_adr)
    mae_zonal = mean_absolute_error(df_val_adr["real_qqha"], df_val_adr["pred_qqha"])

    vcol1, vcol2 = st.columns([1, 2])
    with vcol1:
        st.markdown(f"<div style='background:#1a1d2e; border:1px solid #2a2d3e; border-radius:8px; padding:20px; text-align:center;'><h2 style='color:#f0f0f0; margin:0;'>± {mae_zonal:.1f} qq/ha</h2><p style='color:#9ca3af; font-size:12px; margin-top:5px;'>Error Medio Absoluto (MAE)</p></div>", unsafe_allow_html=True)
    with vcol2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_val_adr["campaña"], y=df_val_adr["error_qqha"], mode="lines+markers", line=dict(color="#4ade80", width=2)))
        fig2.add_hline(y=0, line_color="#6b7280", line_width=1, line_dash="dash")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", height=180, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
# PESTAÑA 2: METODOLOGÍA
# =====================================================================
with tab2:
    st.markdown("<div class='section-title'>PROCESAMIENTO GEOESPACIAL Y ML</div>", unsafe_allow_html=True)
    st.markdown("""
    ### 📂 Manejo de Archivos GIS
    Para este tablero, se integró el archivo **`adrs.geojson`**. A diferencia de los límites políticos tradicionales (Partidos), este archivo contiene las **geometrías propias de la empresa**, permitiendo visualizar exactamente las áreas de influencia (ADRs) que el equipo delimitó.
    
    ### ⚙️ El Proceso en 3 Pasos:
    1. **Superposición**: Se calculó qué porcentaje de cada partido cae dentro de cada ADR.
    2. **Clima NASA**: Se descargó el clima histórico de los partidos y se ponderó según la superficie ocupada dentro del ADR.
    3. **Modelado Individual**: Se entrenó un XGBoost específico para cada polígono del GeoJSON, logrando una predicción personalizada por zona.
    """)
st.markdown("<div class='section-title'>HOJA DE RUTA — PRÓXIMAS MEJORAS</div>", unsafe_allow_html=True)

st.markdown("""
    ---
    ### 🌽 Incorporación de Maíz
    El modelo actual predice únicamente **soja**. La extensión a maíz implica:
    - Redefinir el período crítico: el maíz florece entre **diciembre y enero** (R1-R3), 
      con llenado de grano en **febrero-marzo**.
    - Reentrenar el modelo con rendimientos históricos de maíz por partido (disponibles en MAGYP).
    - El pipeline de clima ya está construido — solo se adaptan los períodos fenológicos.
    - **Impacto esperado**: predicción independiente por cultivo para cada ADR, con el mismo nivel de detalle que soja.

    ---
    ### 📡 NDVI Satelital (Índice de Vegetación)
    El NDVI mide el verdor del cultivo desde satélite en tiempo real. Incorporarlo implica:
    - Descargar imágenes de **Sentinel-2 o MODIS** via Google Earth Engine para cada ADR.
    - Calcular el NDVI máximo alcanzado durante la campaña y su anomalía respecto al histórico.
    - El NDVI en floración es uno de los predictores más poderosos del rendimiento final.
    - **Impacto esperado**: mejora significativa del R² — especialmente en campañas intermedias 
      donde el clima no explica toda la variabilidad.
    - Limitante: La geolocalización de los lotes con cultivo de Soja/Maíz

    ---
    ### 🌊 Mejora del R² — Más Variables (ejemlpo)
    El R² actual (~0.39) puede mejorar incorporando:
    
    | Variable | Fuente | Impacto esperado |
    |---|---|---|
    | NDVI campaña | Sentinel-2 / MODIS | Alto |
    | Índice de sequía (SPI) | CHIRPS | Medio-alto |
    | Agua útil del suelo | INTA / SMN | Medio |
    | Fecha de siembra promedio | Bolsa de Cereales | Medio |
    | Temperatura de canopia | MODIS LST | Medio |

    El evento **2008/09** (sequía extrema) es el principal responsable del R² bajo. 
    Con NDVI satelital ese evento hubiera sido captado temprano, ya que el verdor 
    colapsó semanas antes de la cosecha.

    ---
    ### 📊 Estimación de Volumen Total
    Hoy el modelo predice **rinde en kg/ha**. El siguiente paso es multiplicar por 
    las hectáreas sembradas para estimar **producción total por ADR** (Dato que hoy en día obtenemos de Ciampagna):
    
    ```
    Producción (tn) = Rinde estimado (kg/ha) × Hectáreas sembradas / 1000
    ```
    
    Las hectáreas sembradas por ADR se pueden obtener de:
    - Declaraciones juradas de los socios/clientes
    - Datos de siembra de la Bolsa de Cereales
    - Estimación satelital por detección de cultivos (más avanzado)
    

    ---
    ### 🤖 Mejoras al Modelo de ML
    - **Más campañas**: cada año nuevo que pasa agrega una fila al dataset. Con 30+ campañas el R² mejora naturalmente.
    - **Modelos de ensamble**: combinar XGBoost con una red neuronal simple puede capturar patrones no lineales.
    - **Calibración por ADR**: algunos ADRs tienen mejor historial de datos que otros — 
      entrenar modelos específicos por zona mejora la precisión local.
    - **Intervalos de confianza**: en lugar de un solo número, mostrar un rango probable 
      (ej: *"entre 18 y 23 qq/ha con 80% de probabilidad"*) es más útil para la toma de decisiones.

    """)
# ────────────────────────────────────────────────────────────
#  FOOTER
# ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"""
<div style='text-align:center; font-size:11px; color:#374151; padding: 8px 0;'>
    Predictor de Rinde CeresTolvas &nbsp;·&nbsp; GIS-Enabled Dashboard &nbsp;·&nbsp; {datetime.now().strftime("%d/%m/%Y")}
</div>
""", unsafe_allow_html=True)