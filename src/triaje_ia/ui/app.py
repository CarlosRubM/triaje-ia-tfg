"""
src/triaje_ia/ui/app.py
────────────────────────
Interfaz Streamlit — Extracción de vector clínico con LLM.
Ejecutar: uv run streamlit run src/triaje_ia/ui/app.py
"""

import base64
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from triaje_ia.llm.extractor import extraer_vector_clinico

LOGO_PATH = Path("src/triaje_ia/ui/assets/logo.png")


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
    --font-sans:  -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui,
                  'Helvetica Neue', Arial, sans-serif;
    --font-serif: Georgia, 'Times New Roman', serif;
    --font-mono:  ui-monospace, 'SF Mono', 'Cascadia Code', 'Fira Code',
                  Consolas, 'Liberation Mono', monospace;
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
.block-container { padding: 2.5rem 3.5rem 5rem 3.5rem !important; max-width: 1360px !important; }
[data-testid="column"] { padding: 0 !important; }

/* ── Logo & Header ── */
.logo-wrap { display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.45rem; }
.logo-img  { height: 56px; width: auto; object-fit: contain; flex-shrink: 0; }
.logo-text {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 2.2rem; font-weight: 400; font-style: normal;
    color: #0F172A; letter-spacing: -0.02em; line-height: 1;
}
.logo-text .ia { font-style: italic; color: var(--btn-bg); }
.badge {
    font-family: var(--font-mono); font-size: 0.57rem; color: var(--text-secondary);
    background: #E2E8F0; padding: 3px 9px; border-radius: 4px;
    letter-spacing: 0.08em; text-transform: uppercase;
    vertical-align: middle; margin-left: 0.6rem; font-style: normal;
}
.tagline { font-size: 0.78rem; color: var(--text-secondary); font-weight: 400; letter-spacing: 0.01em; margin-top: 0.5rem; }
.header-divider { height: 1px; background: var(--border); margin: 1.5rem 0 2rem 0; }

/* ── Labels ── */
.sec-label {
    font-family: var(--font-mono); font-size: 0.7rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: #1E293B; font-weight: 600;
    margin-bottom: 0.8rem; display: block;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; font-family: var(--font-mono) !important;
    font-size: 0.75rem !important; color: var(--text-primary) !important;
    min-height: 38px !important; padding: 0 0.75rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSelectbox"] label { display: none !important; }

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; font-family: var(--font-sans) !important;
    font-size: 0.87rem !important; color: var(--text-primary) !important;
    line-height: 1.8 !important; padding: 1.1rem 1.2rem !important;
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
    font-family: var(--font-sans) !important; font-size: 0.85rem !important;
    font-weight: 500 !important; padding: 0.72rem 1.6rem !important;
    width: 100% !important; margin-top: 0.5rem !important;
    box-shadow: 0 1px 3px rgba(3,105,161,0.28) !important;
    transition: background 0.2s, transform 0.15s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: var(--btn-hover) !important; transform: translateY(-1px);
}
[data-testid="stButton"] > button[kind="primary"]:disabled {
    background-color: var(--btn-disabled) !important; color: #94A3B8 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] { border-top: 1px solid var(--border); padding-top: 0.25rem; margin-top: 0.25rem; }
[data-testid="stTabs"] button[data-testid="stTab"] {
    font-family: var(--font-mono) !important; font-size: 0.62rem !important;
    letter-spacing: 0.14em !important; color: var(--text-secondary) !important;
    text-transform: uppercase;
}
[data-testid="stTabs"] button[data-testid="stTab"][aria-selected="true"] {
    color: #0F172A !important; border-bottom-color: var(--btn-bg) !important;
}

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

/* ── ML pending banner ── */
.ml-pending {
    background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 10px;
    padding: 1.1rem 1.4rem; margin-top: 1rem;
    display: flex; align-items: center; gap: 0.85rem;
}

@media (max-width: 960px) {
    .vitals-grid { grid-template-columns: repeat(2, 1fr); }
    .block-container { padding: 1.5rem 1.25rem 3rem !important; }
    .logo-img { height: 44px; }
}
</style>
""", unsafe_allow_html=True)


MODELOS_LLM = ["llama3.2", "qwen2.5", "qwen2.5:7b", "llama3.2:1b", "mistral"]


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
col_title, col_model = st.columns([10, 11], gap="large")

with col_title:
    st.markdown(
        '<div class="logo-wrap">'
        '<span class="logo-text">tr<span class="ia">IA</span>je'
        '<span class="badge">TFG &middot; En desarrollo</span>'
        '</span></div>'
        '<p class="tagline">Apoyo a la decisión clínica &nbsp;&middot;&nbsp; '
        'Extracción semántica con LLM + Clasificación ML</p>',
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
                with st.spinner("Extrayendo vector clínico con Ollama..."):
                    vector = extraer_vector_clinico(narrativa, modelo=modelo_llm)

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
                    f'<div style="font-family:Georgia,serif; font-size:1.45rem;'
                    f' font-weight:400; color:#0F172A; line-height:1.1;">{valor}</div>'
                    if valor is not None else
                    '<div style="font-size:1rem; color:#CBD5E1;">—</div>'
                )
                html += f"""
                <div style="background:#FFF; border:1px solid #E2E8F0; border-radius:10px;
                            padding:0.85rem 1rem; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
                    <div style="font-size:0.55rem; letter-spacing:0.15em; text-transform:uppercase;
                                color:#64748B; margin-bottom:0.35rem;">{nombre}</div>
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
                    f'<span style="display:inline-block; background:{bg}; color:{text_color};'
                    f' border:1px solid {border}; font-size:0.77rem; font-weight:400;'
                    f' padding:3px 12px; border-radius:20px; margin:2px 3px 2px 0;">{i}</span>'
                    for i in items
                ])

            secciones = [
                ("Síntomas presentes",      sintomas,   "#EFF6FF", "#1D4ED8", "#BFDBFE"),
                ("Antecedentes patológicos", patologias, "#FFF7ED", "#C2410C", "#FED7AA"),
                ("Medicación habitual",      medicacion, "#F0FDF4", "#166534", "#BBF7D0"),
            ]
            alguna = False
            for titulo, items, bg, tc, border in secciones:
                if items:
                    alguna = True
                    st.markdown(f"""
                    <div style="margin-bottom:1.1rem;">
                        <div style="font-size:0.57rem; letter-spacing:0.15em; text-transform:uppercase;
                                    color:#64748B; margin-bottom:0.5rem;">{titulo}</div>
                        {render_tags(items, bg, tc, border)}
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
                    min-height:390px; text-align:center; gap:1rem; background:#FFFFFF;
                    border:1px dashed #CBD5E1; border-radius:12px; padding:2.5rem;
                    box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="width:52px; height:52px; border:1.5px solid #CBD5E1; border-radius:50%;
                        display:flex; align-items:center; justify-content:center; opacity:0.55;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                     stroke="#94A3B8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M9 12h6M12 9v6"/><circle cx="12" cy="12" r="9"/>
                </svg>
            </div>
            <div>
                <p style="font-family:Georgia,serif; font-size:1.1rem; font-weight:400;
                           color:#64748B; margin-bottom:0.4rem; letter-spacing:-0.01em;">
                    Sin análisis activo
                </p>
                <p style="font-size:0.77rem; color:#94A3B8; max-width:245px; line-height:1.75; margin:0;">
                    Redacte la historia clínica del paciente y pulse
                    <strong style="color:#64748B; font-weight:600;">Analizar caso clínico</strong>.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECCIÓN ML — pendiente de entrenamiento
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)
st.markdown('<span class="sec-label">Predicción diagnóstica · Clasificación ML</span>', unsafe_allow_html=True)

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
            Pendiente de entrenamiento con MIMIC-IV-ED
        </p>
        <p style="font-size:0.73rem; color:#94A3B8; line-height:1.65; margin:0;">
            Una vez entrenado el modelo XGBoost, esta sección mostrará el nivel ESI predicho,
            la confianza y la distribución de probabilidades por clase.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)