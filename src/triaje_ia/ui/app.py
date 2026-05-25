"""
src/triaje_ia/ui/app.py
────────────────────────
Interfaz Streamlit — Sistema híbrido LLM + ML para triaje clínico.
Ejecutar: uv run streamlit run src/triaje_ia/ui/app.py
"""

import base64
import os
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from triaje_ia.llm.factory import crear_extractor
from triaje_ia.llm.validator import validar_vector_clinico, NivelAlerta
from triaje_ia.inference.predictor import TriajePredictor
from triaje_ia.ml.explicabilidad import (
    crear_explainer, explicar_prediccion, generar_shap_explanation_object,
)

LOGO_PATH = Path("src/triaje_ia/ui/assets/logo.png")

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def _cargar_predictor():
    """Carga TriajePredictor una sola vez (lee active_model.json)."""
    return TriajePredictor()


# Backend LLM via variable de entorno (ollama local / api cloud)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
_extraer_fn = crear_extractor(LLM_BACKEND)

# Colores ESI por nivel de acuity
ESI_CONFIG = {
    1: {"label": "ESI 1 — Resucitación",    "color": "#DC2626", "bg": "#FEF2F2", "border": "#FECACA"},
    2: {"label": "ESI 2 — Emergencia",       "color": "#EA580C", "bg": "#FFF7ED", "border": "#FED7AA"},
    3: {"label": "ESI 3 — Urgente",          "color": "#CA8A04", "bg": "#FEFCE8", "border": "#FEF08A"},
    4: {"label": "ESI 4 — Menos urgente",    "color": "#16A34A", "bg": "#F0FDF4", "border": "#BBF7D0"},
    5: {"label": "ESI 5 — No urgente",       "color": "#2563EB", "bg": "#EFF6FF", "border": "#BFDBFE"},
}


def logo_base64() -> str:
    if LOGO_PATH.exists():
        data = LOGO_PATH.read_bytes()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return ""


st.set_page_config(
    page_title="trIAje — Sistema de Apoyo Clínico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Merriweather:wght@400;700&display=swap');

:root {
    --bg-app:         #F8FAFC;
    --bg-card:        #FFFFFF;
    --border:         #E2E8F0;
    --text-primary:   #334155;
    --text-secondary: #64748B;
    --text-muted:     #94A3B8;
    --btn-bg:         #0369A1;
    --btn-hover:      #075985;
    --btn-disabled:   #CBD5E1;
    --focus-ring:     #0EA5E9;
    --focus-ring-a:   rgba(14,165,233,0.18);
    --shadow-sm:      0 1px 3px rgba(0,0,0,0.05);
    --radius:         10px;
    --font-sans:      'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-serif:     'Merriweather', Georgia, 'Times New Roman', serif;
    --font-mono:      ui-monospace, 'SF Mono', 'Cascadia Code', Consolas, monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background-color: var(--bg-app) !important;
    font-family: var(--font-sans);
    color: var(--text-primary);
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden; display: none; }
.block-container { padding: clamp(1.5rem, 3vw, 2.5rem) clamp(1.5rem, 4vw, 3.5rem) 5rem !important; max-width: 1360px !important; }
[data-testid="column"] { padding: 0 !important; }

/* ── Animaciones Base ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.4s ease-out forwards; }

/* ── Logo & Header ── */
.logo-wrap { display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.2rem; }
.logo-img  { height: 56px; width: auto; object-fit: contain; flex-shrink: 0; }
.logo-text {
    font-family: var(--font-serif);
    font-size: 2.2rem; font-weight: 700;
    color: #0F172A; letter-spacing: -0.02em; line-height: 1;
}
.logo-text .ia { font-style: italic; color: var(--btn-bg); }
.author-text { font-family: var(--font-sans); font-size: 0.9rem; font-weight: 500; color: var(--text-secondary); margin-bottom: 0.5rem; }
.tagline { font-size: 0.85rem; color: var(--text-secondary); font-weight: 400; letter-spacing: 0.01em; margin-top: 0.2rem; line-height: 1.5; }
.header-divider { height: 1px; background: var(--border); margin: 1.5rem 0 2rem 0; }

/* ── Labels ── */
.sec-label {
    font-family: var(--font-mono); font-size: 0.85rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #0F172A; font-weight: 700;
    margin-bottom: 1.2rem; display: block;
}

/* ── Cards Generales ── */
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; box-shadow: var(--shadow-sm); padding: 1.4rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; font-family: var(--font-mono) !important;
    font-size: 0.75rem !important; color: var(--text-primary) !important;
    min-height: 38px !important; padding: 0 0.75rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--focus-ring) !important;
    box-shadow: 0 0 0 3px var(--focus-ring-a) !important;
}
[data-testid="stSelectbox"] label { display: none !important; }

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; font-family: var(--font-sans) !important;
    font-size: 0.9rem !important; color: var(--text-primary) !important;
    line-height: 1.6 !important; padding: 1.1rem 1.2rem !important;
    box-shadow: var(--shadow-sm) !important; resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--focus-ring) !important;
    box-shadow: 0 0 0 3px var(--focus-ring-a) !important; outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stTextArea"] label { display: none !important; }

/* ── Button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background-color: var(--btn-bg) !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-family: var(--font-sans) !important; font-size: 0.9rem !important;
    font-weight: 500 !important; padding: 0.72rem 1.6rem !important;
    width: 100% !important; margin-top: 0.5rem !important;
    box-shadow: 0 2px 4px rgba(3,105,161,0.15) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: var(--btn-hover) !important; transform: translateY(-1px);
    box-shadow: 0 4px 6px rgba(3,105,161,0.2) !important;
}
[data-testid="stButton"] > button[kind="primary"]:disabled {
    background-color: var(--btn-disabled) !important; color: #94A3B8 !important;
    transform: none !important; box-shadow: none !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] { border-top: 1px solid var(--border); padding-top: 0.25rem; margin-top: 0.25rem; }
[data-testid="stTabs"] button[data-testid="stTab"] {
    font-family: var(--font-mono) !important; font-size: 0.65rem !important;
    letter-spacing: 0.12em !important; color: var(--text-secondary) !important;
    text-transform: uppercase; transition: color 0.2s;
}
[data-testid="stTabs"] button[data-testid="stTab"][aria-selected="true"] {
    color: #0F172A !important; border-bottom-color: var(--btn-bg) !important; font-weight: 600 !important;
}
[data-testid="stTabs"] [data-testid="stTabContent"] { animation: fadeIn 0.3s ease-out; }

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-family: var(--font-mono) !important; font-size: 0.76rem !important; color: var(--focus-ring) !important; }

/* ── Vitals grid ── */
.vitals-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.65rem; margin-top: 0.5rem; }

/* ── Skeleton ── */
@keyframes shimmer {
    0%   { background-position: -700px 0; }
    100% { background-position:  700px 0; }
}
.skeleton-line {
    border-radius: 6px;
    background: linear-gradient(90deg, #E2E8F0 25%, #F1F5F9 50%, #E2E8F0 75%);
    background-size: 700px 100%;
    animation: shimmer 1.5s infinite linear;
}

/* ── ML result panel & Alertas ── */
.ml-pending {
    background: var(--bg-card); border: 1px dashed var(--border); border-radius: 12px;
    padding: 1.5rem; margin-top: 1rem;
    display: flex; align-items: center; gap: 1rem;
    box-shadow: var(--shadow-sm);
}
.esi-badge {
    display: inline-flex; align-items: center; gap: 0.8rem;
    padding: 0.85rem 1.6rem; border-radius: 12px;
    font-family: var(--font-mono); font-size: 1.15rem;
    font-weight: 700; letter-spacing: 0.04em;
    border-width: 2px; border-style: solid;
}
.alert-clinical {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-left: 4px solid #DC2626; border-radius: 10px;
    padding: 0.75rem 1.1rem; margin-bottom: 0.8rem;
    font-size: 0.85rem; color: #991B1B; animation: fadeIn 0.4s ease-out;
}
.warning-validation {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-left: 4px solid #D97706; border-radius: 10px;
    padding: 0.6rem 1rem; margin-bottom: 0.6rem;
    font-size: 0.82rem; color: #92400E; animation: fadeIn 0.4s ease-out;
}

/* ── Probabilidades (CSS GLOBAL para que no desaparezcan) ── */
.prob-label {
    font-family: var(--font-mono); font-size: 0.65rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-secondary); margin-bottom: 0.2rem;
}
.prob-bar-bg {
    height: 12px; background: #F8FAFC; border-radius: 10px;
    overflow: hidden; position: relative; width: 100%;
}
@keyframes growBar {
    from { width: 0%; }
    to { width: var(--target-width); }
}
.prob-bar-fill {
    height: 100%; border-radius: 10px;
    min-width: 1.5%;
    width: var(--target-width);
    animation: growBar 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

@media (max-width: 960px) {
    .vitals-grid { grid-template-columns: repeat(2, 1fr); }
    .block-container { padding: 1.5rem 1.25rem 3rem !important; }
    .logo-img { height: 44px; }
}
</style>
""", unsafe_allow_html=True)


MODELOS_LLM = (
    [
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.2",
        "qwen2.5",
        "qwen2.5:7b",
        "llama3.2:1b",
        "mistral",
    ]
    if LLM_BACKEND == "ollama"
    else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_model = st.columns([10, 11], gap="large")

with col_title:
    st.markdown(
        '<div class="logo-wrap">'
        '<span class="logo-text">tr<span class="ia">IA</span>je</span>'
        '</div>'
        '<div class="author-text">Carlos Rubio Martínez</div>'
        '<p class="tagline">Sistema de apoyo al triaje clínico &nbsp;&middot;&nbsp; TFG</p>',
        unsafe_allow_html=True,
    )

with col_model:
    st.markdown("""
    <div style="display:flex; flex-direction:column; justify-content:flex-end; height:100%; padding-bottom:3px;">
        <span class="sec-label" style="margin-bottom:0.4rem;">Modelo LLM local</span>
    </div>
    """, unsafe_allow_html=True)
    modelo_llm = st.selectbox("Modelo", MODELOS_LLM, index=0, label_visibility="collapsed")

st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([10, 11], gap="large")


# ── Columna izquierda ─────────────────────────────────────────
with col_left:
    st.markdown('<span class="sec-label">Historia clínica del paciente</span>', unsafe_allow_html=True)

    narrativa = st.text_area(
        label="Historia",
        placeholder="Introduzca la historia clínica tal como la redactaría en triaje...",
        height=390,
        label_visibility="collapsed",
    )

    analizar = st.button(
        "Analizar caso clínico →",
        type="primary",
        use_container_width=True,
        disabled=not narrativa.strip(),
    )

    st.markdown("""
    <p style="font-size:0.72rem; color:#94A3B8; margin-top:0.9rem; line-height:1.75;">
        Procesamiento íntegramente local. El LLM extrae variables clínicas
        estructuradas a partir del texto libre de triaje.<br>
        <strong style="color:#64748B; font-weight:500;">
            No sustituye el criterio del profesional sanitario.
        </strong>
    </p>
    """, unsafe_allow_html=True)


# ── Columna derecha ───────────────────────────────────────────
with col_right:
    st.markdown('<span class="sec-label">Vector clínico extraído</span>', unsafe_allow_html=True)

    if "ultimo_vector"   not in st.session_state:
        st.session_state.ultimo_vector    = None
        st.session_state.ultima_narrativa = ""

    if analizar and narrativa.strip():

        # Cache: no rellamar si la narrativa no cambió
        if narrativa.strip() == st.session_state.ultima_narrativa and st.session_state.ultimo_vector:
            vector = st.session_state.ultimo_vector
        else:
            resultado_placeholder = st.empty()

            # Skeleton loading
            with resultado_placeholder.container():
                st.markdown("""
                <div style="background:#FFF; border:1px solid #E2E8F0; border-radius:12px;
                            padding:1.6rem 1.8rem; box-shadow:0 1px 3px rgba(0,0,0,0.05); margin-bottom:1.25rem;">
                    <div class="skeleton-line" style="height:10px; width:40%; margin-bottom:1rem;"></div>
                    <div class="skeleton-line" style="height:22px; width:60%; margin-bottom:1.15rem;"></div>
                    <div class="skeleton-line" style="height:8px; width:100%;"></div>
                </div>
                <div class="vitals-grid">
                    <div class="skeleton-line" style="height:70px; border-radius:10px;"></div>
                    <div class="skeleton-line" style="height:70px; border-radius:10px;"></div>
                    <div class="skeleton-line" style="height:70px; border-radius:10px;"></div>
                    <div class="skeleton-line" style="height:70px; border-radius:10px;"></div>
                </div>
                """, unsafe_allow_html=True)

            try:
                spinner_msg = (
                    "Extrayendo vector clínico con Ollama..."
                    if LLM_BACKEND == "ollama"
                    else "Extrayendo vector clínico via API..."
                )
                with st.spinner(spinner_msg):
                    vector = _extraer_fn(narrativa, modelo=modelo_llm)

                st.session_state.ultimo_vector    = vector
                st.session_state.ultima_narrativa = narrativa.strip()
                resultado_placeholder.empty()

            except ConnectionError:
                resultado_placeholder.empty()
                st.error("⚡ No se puede conectar con Ollama. Ejecute: `ollama serve`")
                st.stop()
            except Exception as e:
                resultado_placeholder.empty()
                st.error(f"Error en la extracción: {e}")
                st.stop()

        # ── Renderizar vector ─────────────────────────────────
        vector = st.session_state.ultimo_vector

        # Realizamos la inferencia ML aquí (si está disponible) para usar el color ESI en la UI
        color_esi_global = "transparent"
        try:
            predictor_temp = _cargar_predictor()
            st.session_state.ultimo_resultado_ml = predictor_temp.predict(vector, st.session_state.ultima_narrativa)
            clase_global = st.session_state.ultimo_resultado_ml.clase_predicha
            color_esi_global = ESI_CONFIG[clase_global]["color"]
        except Exception:
            st.session_state.ultimo_resultado_ml = None
            color_esi_global = "transparent"

        # Banner de éxito
        st.markdown("""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-left:4px solid #22C55E;
                    border-radius:10px; padding:0.75rem 1.1rem; margin-bottom:1.25rem;
                    display:flex; align-items:center; gap:0.65rem;">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                 stroke="#16A34A" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"/>
            </svg>
            <span style="font-family:var(--font-mono); font-size:0.67rem; letter-spacing:0.12em;
                         text-transform:uppercase; color:#166534; font-weight:500;">
                Extracción completada
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs: Vitales / Datos clínicos ───────────────────
        tab_vitales, tab_clinico = st.tabs(["CONSTANTES VITALES", "DATOS CLÍNICOS"])

        with tab_vitales:
            vitales_raw = {
                "PA Sistólica":  getattr(vector, "presion_sistolica",  None),
                "PA Diastólica": getattr(vector, "presion_diastolica", None),
                "FC (lpm)":      getattr(vector, "frecuencia_cardiaca", None),
                "SpO₂ (%)":      getattr(vector, "saturacion_oxigeno", None),
                "Temperatura":   getattr(vector, "temperatura",         None),
            }

            html = '<div class="vitals-grid">'
            for nombre, valor in vitales_raw.items():
                val_html = (
                    f'<div style="font-family:var(--font-mono); font-size:1.6rem;'
                    f' font-weight:700; color:#0F172A; line-height:1; letter-spacing:-0.03em;">{valor}</div>'
                    if valor is not None else
                    '<div style="font-size:1.2rem; color:#CBD5E1; line-height:1;">—</div>'
                )
                b_color = color_esi_global if color_esi_global != "transparent" else "#E2E8F0"
                html += f"""
                <div style="background:#FFF; border:1px solid #E2E8F0; border-top:3px solid {b_color}; border-radius:10px;
                            padding:1.15rem 1.25rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.03), 0 2px 4px -1px rgba(0,0,0,0.02); 
                            display:flex; flex-direction:column; gap:0.5rem;">
                    <div style="font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase;
                                color:#64748B; font-weight:700; line-height:1;">{nombre}</div>
                    {val_html}
                </div>"""
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

        with tab_clinico:
            sintomas   = getattr(vector, "sintomas_presentes", []) or []
            patologias = getattr(vector, "patologias_previas",  []) or []
            medicacion = getattr(vector, "medicacion_habitual", []) or []

            def render_tags(items, bg, text_color, border):
                return "".join([
                    f'<span style="display:inline-flex; align-items:center; background:{bg}; color:{text_color};'
                    f' border:1px solid {border}; font-size:0.82rem; font-weight:500; line-height:1.4;'
                    f' padding:4px 14px; border-radius:24px; margin:0 6px 8px 0; box-shadow:0 1px 2px rgba(0,0,0,0.02);">{i}</span>'
                    for i in items
                ])

            secciones = [
                ("Síntomas presentes",      sintomas,   "#EFF6FF", "#1D4ED8", "#93C5FD"),
                ("Antecedentes patológicos", patologias, "#FFF7ED", "#C2410C", "#FDBA74"),
                ("Medicación habitual",      medicacion, "#F0FDF4", "#15803D", "#86EFAC"),
            ]
            alguna = False
            for titulo, items, bg, tc, border in secciones:
                if items:
                    alguna = True
                    st.markdown(f"""
                    <div style="margin-bottom:1.5rem;">
                        <div style="font-size:0.7rem; letter-spacing:0.12em; text-transform:uppercase;
                                    color:#475569; margin-bottom:0.6rem; font-weight:700;">{titulo}</div>
                        <div style="display:flex; flex-wrap:wrap;">{render_tags(items, bg, tc, border)}</div>
                    </div>
                    """, unsafe_allow_html=True)

            if not alguna:
                st.markdown("""
                <p style="font-size:0.8rem; color:#94A3B8; margin-top:0.5rem;">
                    No se detectaron síntomas, antecedentes ni medicación en la narrativa.
                </p>
                """, unsafe_allow_html=True)

    else:
        # Estado vacío
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    min-height:390px; text-align:center; gap:1rem; background:var(--bg-card);
                    border:1px dashed var(--border); border-radius:12px; padding:2.5rem;
                    box-shadow:var(--shadow-sm);">
            <div style="width:52px; height:52px; border:1.5px solid #CBD5E1; border-radius:50%;
                        display:flex; align-items:center; justify-content:center; opacity:0.55; background:#F8FAFC;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                     stroke="#64748B" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
            </div>
            <div>
                <p style="font-family:var(--font-serif); font-size:1.1rem; font-weight:700;
                           color:#334155; margin-bottom:0.4rem; letter-spacing:-0.01em;">
                    Esperando historia clínica
                </p>
                <p style="font-size:0.85rem; color:var(--text-secondary); max-width:260px; line-height:1.6; margin:0 auto;">
                    Redacte los motivos de consulta del paciente y pulse
                    <strong style="color:#334155; font-weight:600;">Analizar caso clínico</strong>.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECCIÓN ML — Predicción diagnóstica
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)
st.markdown('<span class="sec-label">Predicción diagnóstica · Clasificación ML</span>', unsafe_allow_html=True)

# Intentar cargar el predictor
try:
    predictor = _cargar_predictor()
    _modelo_disponible = True
except FileNotFoundError:
    _modelo_disponible = False

if not _modelo_disponible:
    st.markdown("""
    <div class="ml-pending">
        <div style="width:36px; height:36px; border:1.5px solid #CBD5E1; border-radius:8px;
                    display:flex; align-items:center; justify-content:center; flex-shrink:0; opacity:0.6;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="#64748B" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="9"/><polyline points="12 6 12 12 16 14"/>
            </svg>
        </div>
        <div>
            <p style="font-size:0.82rem; color:#334155; font-weight:500; margin-bottom:2px;">
                Modelo no encontrado
            </p>
            <p style="font-size:0.73rem; color:#94A3B8; line-height:1.65; margin:0;">
                Verifica que <code>models/active_model.json</code> existe
                y que los artefactos referenciados están en <code>models/</code>.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.ultimo_vector is not None:
    vector = st.session_state.ultimo_vector
    narrativa_actual = st.session_state.ultima_narrativa

    # 1. Validación semántica
    alertas = validar_vector_clinico(vector, narrativa_actual)

    # 2-4. Inferencia unificada
    if st.session_state.get("ultimo_resultado_ml"):
        result = st.session_state.ultimo_resultado_ml
    else:
        result = predictor.predict(vector, narrativa_actual)
        st.session_state.ultimo_resultado_ml = result
    probas_fila = result.probas
    clase_predicha = result.clase_predicha
    confianza = result.confianza
    threshold_activado = result.threshold_a1_activado
    umbral_alerta_a1 = getattr(predictor, "_warning_threshold_a1", 0.40)
    X = result.X

    esi = ESI_CONFIG[clase_predicha]

    # Alerta clinica si la probabilidad de A1 supera el umbral elegido.
    if threshold_activado:
        st.markdown(f"""
        <div class="alert-clinical" style="display:flex; align-items:flex-start; gap:0.6rem;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; margin-top:2px;">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                <line x1="12" y1="9" x2="12" y2="13"></line>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            <div>
                <strong>Alerta de seguridad clinica:</strong>
                P(acuity=1) = {probas_fila[0]:.1%} supera el umbral de aviso
                ({umbral_alerta_a1:.0%}).
                Revisar posible criticidad A1 antes de confirmar el nivel sugerido.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Warnings del validator ──
    alertas_warn = [a for a in alertas if a.nivel in (NivelAlerta.WARNING, NivelAlerta.ERROR)]
    for alerta in alertas_warn:
        if alerta.nivel == NivelAlerta.ERROR:
            icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; transform:translateY(2px); margin-right:4px;"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'
        else:
            icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; transform:translateY(2px); margin-right:4px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
            
        st.markdown(f"""
        <div class="warning-validation" style="display:flex; align-items:flex-start;">
            {icon}
            <span><strong>[{alerta.campo}]</strong> {alerta.mensaje}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Banner ESI Principal ──
    st.markdown(f"""
    <div style="background:{esi['bg']}; border:1px solid {esi['border']}; border-radius:12px; padding:1.25rem 1.5rem; margin-bottom:1.5rem; display:flex; align-items:center; justify-content:space-between; box-shadow:0 1px 2px rgba(0,0,0,0.02);">
        <div style="display:flex; align-items:center; gap:1.2rem;">
            <div style="width:46px; height:46px; border-radius:50%; background:#FFF; display:flex; align-items:center; justify-content:center; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{esi['color']}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                </svg>
            </div>
            <div>
                <div style="font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:{esi['color']}; font-weight:700; margin-bottom:0.15rem; opacity:0.8;">Nivel sugerido</div>
                <div style="font-family:var(--font-mono); font-size:1.4rem; font-weight:700; color:{esi['color']}; line-height:1; letter-spacing:-0.02em;">{esi['label']}</div>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.65rem; letter-spacing:0.05em; text-transform:uppercase; color:var(--text-secondary); margin-bottom:0.15rem; font-weight:600;">Confianza</div>
            <div style="font-family:var(--font-mono); font-size:1.25rem; font-weight:700; color:#0F172A;">{confianza:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Barras de probabilidad ──
    clases_nombre = ["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"]
    colores_barra = ["#DC2626", "#EA580C", "#CA8A04", "#16A34A", "#2563EB"]

    if isinstance(probas_fila, (list, np.ndarray)) and len(probas_fila) == 5:
        barras_html = ""
        for i in range(5):
            try:
                p = float(probas_fila[i])
            except (ValueError, TypeError):
                p = 0.0
                
            width_pct = p * 100
            is_max = (i == clase_predicha - 1)
            
            # Estilos condicionales
            text_color = "#0F172A" if is_max else "#94A3B8"
            font_weight = "700" if is_max else "500"
            bar_opacity = "1" if is_max else "0.3"
            icon_opacity = "1" if is_max else "0.4"
            
            dot_html = f"<span style='display:inline-block; width:8px; height:8px; border-radius:50%; background-color:{colores_barra[i]}; opacity:{icon_opacity}; box-shadow:0 1px 2px rgba(0,0,0,0.15); margin-right:0.4rem; transform:translateY(-1px);'></span>"
            
            barras_html += (
                f"<div style='margin-bottom:1.1rem;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.35rem;'>"
                f"<span class='prob-label' style='color:{text_color}; font-weight:{font_weight}; display:flex; align-items:center;'>{dot_html}{clases_nombre[i]}</span>"
                f"<span style=\"font-family:var(--font-mono); font-size:0.75rem; color:{text_color}; font-weight:{font_weight};\">{p:.1%}</span>"
                f"</div>"
                f"<div class='prob-bar-bg'>"
                f"<div class='prob-bar-fill' style='--target-width:{width_pct:.1f}%; background:{colores_barra[i]}; opacity:{bar_opacity};'></div>"
                f"</div>"
                f"</div>"
            )

        st.markdown(
            f"<div class='card fade-in' style='margin-bottom:1.2rem;'>"
            f"<div style='font-family:var(--font-mono); font-size:0.85rem; letter-spacing:0.12em; text-transform:uppercase; color:#0F172A; margin-bottom:1.2rem; font-weight:700;'>Distribución de probabilidades</div>"
            f"{barras_html}"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("Distribución de probabilidades no disponible para esta predicción.")

    # ── SHAP Bar Chart (Plotly) ──
    try:
        explainer = crear_explainer(predictor._clf, clase=clase_predicha)
        shap_explanation = generar_shap_explanation_object(
            explainer, X, clase=clase_predicha, modelo=predictor._clf,
        )
        import plotly.graph_objects as go
        import pandas as pd
        
        shap_vals = np.array(shap_explanation.values).flatten()
        feat_names = np.array(shap_explanation.feature_names)
        
        # Diccionario para nombres de variables más limpios
        MAP_FEAT = {
            "resprate": "Frecuencia respiratoria",
            "age": "Edad",
            "o2sat": "Saturación O₂",
            "heartrate": "Frecuencia cardíaca",
            "sbp": "PA Sistólica",
            "dbp": "PA Diastólica",
            "temp": "Temperatura",
            "llegada_ambulancia": "Llegada en ambulancia"
        }
        
        def clean_feat_name(f):
            if f in MAP_FEAT: return MAP_FEAT[f]
            if f.startswith("cc_"): return "Motivo: " + f[3:].replace("_", " ").capitalize()
            if f.startswith("bert_svd"): return f"Contexto semántico ({f[-2:]})"
            return f.replace("_", " ").capitalize()
            
        feat_names_clean = [clean_feat_name(f) for f in feat_names]
        
        df_shap = pd.DataFrame({'Feature': feat_names_clean, 'SHAP': shap_vals})
        df_shap['Abs_SHAP'] = df_shap['SHAP'].abs()
        df_shap = df_shap.sort_values(by='Abs_SHAP', ascending=True).tail(10)
        # Añadir espacios invisibles para evitar que Plotly agrupe nombres idénticos accidentalmente
        df_shap['Feature'] = [f + (" " * i) for i, f in enumerate(df_shap['Feature'])]
        
        # Colores premium (relleno semitransparente con borde sólido)
        fill_colores = ['rgba(220, 38, 38, 0.85)' if v > 0 else 'rgba(14, 165, 233, 0.85)' for v in df_shap['SHAP']]
        line_colores = ['#B91C1C' if v > 0 else '#0284C7' for v in df_shap['SHAP']]
        
        fig = go.Figure(go.Bar(
            x=df_shap['SHAP'],
            y=df_shap['Feature'],
            orientation='h',
            marker=dict(color=fill_colores, line=dict(color=line_colores, width=1.5)),
            text=[f"{v:+.3f}" for v in df_shap['SHAP']],
            textposition='outside',
            textfont=dict(family="Inter, sans-serif", size=11, color="#1E293B", weight="bold"),
            hovertemplate="<b>%{y}</b><br>Impacto SHAP: %{x:+.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            margin=dict(l=0, r=45, t=15, b=10),
            height=340,
            bargap=0.3,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color="#334155", size=12),
            xaxis=dict(
                showgrid=True, gridcolor="#E2E8F0", 
                zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.5, 
                showticklabels=True, tickfont=dict(color="#64748B", size=10)
            ),
            yaxis=dict(
                showgrid=False, 
                tickfont=dict(color="#0F172A", size=11, weight="bold")
            ),
            showlegend=False,
            dragmode=False
        )
        
        st.markdown("""
        <div style="font-family:var(--font-mono); font-size:0.85rem; letter-spacing:0.12em; text-transform:uppercase;
                    color:#0F172A; margin-bottom:1rem; font-weight:700;">Por qué esta predicción</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Top features texto
        resultado_shap = explicar_prediccion(
            explainer, X, clase_predicha=clase_predicha, modelo=predictor._clf,
        )
        if resultado_shap.top_positivas:
            tops = ", ".join(
                f"<strong>{f.nombre}</strong> ({f.shap_value:+.3f})"
                for f in resultado_shap.top_positivas[:3]
            )
            st.markdown(f"""
            <p style="font-size:0.73rem; color:#64748B; line-height:1.7; margin-top:0.3rem;">
                Factores principales ↑: {tops}
            </p>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f"""
        <p style="font-size:0.75rem; color:#94A3B8; margin-top:0.5rem;">
            SHAP no disponible: {e}
        </p>
        """, unsafe_allow_html=True)

else:
    # Sin análisis activo
    st.markdown("""
    <div class="ml-pending fade-in">
        <div style="width:40px; height:40px; border:1.5px solid #CBD5E1; border-radius:10px;
                    display:flex; align-items:center; justify-content:center; flex-shrink:0; opacity:0.6; background:#F8FAFC;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="#64748B" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
            </svg>
        </div>
        <div>
            <p style="font-family:var(--font-serif); font-size:0.95rem; color:#334155; font-weight:700; margin-bottom:0.2rem;">
                Sin predicción activa
            </p>
            <p style="font-size:0.8rem; color:var(--text-secondary); line-height:1.5; margin:0;">
                Analice un caso clínico para ver la predicción del nivel de urgencia,
                la distribución de probabilidades y la justificación clínica.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:4rem; padding-top:1.5rem; border-top:1px solid var(--border); font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);">
    Carlos Rubio Martínez &middot; TFG
</div>
""", unsafe_allow_html=True)
