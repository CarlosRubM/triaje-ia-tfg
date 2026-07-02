"""
src/triaje_ia/ui/app.py
────────────────────────
Interfaz Streamlit — Sistema híbrido LLM + ML para triaje clínico.
Flujo wizard de 3 fases: Formulario → Procesamiento → Resultado.
Ejecutar: uv run streamlit run src/triaje_ia/ui/app.py
"""

from concurrent.futures import ThreadPoolExecutor
import html
import time
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from triaje_ia.llm.factory import crear_extractor
from triaje_ia.llm.validator import validar_vector_clinico, NivelAlerta
from triaje_ia.inference.predictor import TriajePredictor
from triaje_ia.ui.clinical_form import (
    TriageFormData,
    generar_narrativa_triaje,
    generar_narrativa_triaje_texto_libre,
    validar_entrada_triaje,
)
from triaje_ia.ui.vector_review import (
    construir_vector_desde_revision,
)
from triaje_ia.ui.clinical_display import (
    display_clinical_terms,
    join_review_terms_display,
)
from triaje_ia.ui.shap_presenter import clean_feature_name, split_shap_factors
from triaje_ia.ml.explicabilidad import (
    crear_explainer, explicar_prediccion, generar_shap_explanation_object,
)

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def _cargar_predictor():
    """Carga TriajePredictor una sola vez (lee active_model.json)."""
    return TriajePredictor()


# Extracción clínica local con Ollama.
_extraer_fn = crear_extractor("ollama")
MODELO_LLM_FIJO = "llama3.1:8b-instruct-q4_K_M"

# Colores ESI por nivel de acuity
ESI_CONFIG = {
    1: {"label": "ESI 1 — Resucitación",    "color": "#DC2626", "bg": "#FEF2F2", "border": "#FECACA"},
    2: {"label": "ESI 2 — Emergencia",       "color": "#EA580C", "bg": "#FFF7ED", "border": "#FED7AA"},
    3: {"label": "ESI 3 — Urgente",          "color": "#CA8A04", "bg": "#FEFCE8", "border": "#FEF08A"},
    4: {"label": "ESI 4 — Menos urgente",    "color": "#16A34A", "bg": "#F0FDF4", "border": "#BBF7D0"},
    5: {"label": "ESI 5 — No urgente",       "color": "#2563EB", "bg": "#EFF6FF", "border": "#BFDBFE"},
}

SINTOMAS_FRECUENTES = [
    "dolor toracico",
    "disnea",
    "fiebre",
    "dolor abdominal",
    "mareo/sincope",
    "traumatismo",
    "vomitos",
    "alteracion neurologica",
]

SIGNOS_ALARMA = [
    "mal estado general",
    "compromiso respiratorio",
    "alteracion del nivel de consciencia",
    "dolor toracico activo",
    "deficit neurologico focal",
    "sospecha de sepsis/shock",
    "sangrado activo",
    "trauma mayor",
]


def _parse_int_input(
    raw_value: str,
    label: str,
    min_value: int,
    max_value: int,
) -> tuple[int | None, str | None]:
    value = (raw_value or "").strip()
    if not value:
        return None, None

    normalized = value.replace(",", ".")
    try:
        number = float(normalized)
    except ValueError:
        return None, f"{label}: escriba un número válido."

    if not number.is_integer():
        return None, f"{label}: escriba un número entero."

    parsed = int(number)
    if parsed < min_value or parsed > max_value:
        return None, f"{label}: valor fuera de rango esperado ({min_value}-{max_value})."
    return parsed, None


def _parse_float_input(
    raw_value: str,
    label: str,
    min_value: float,
    max_value: float,
) -> tuple[float | None, str | None]:
    value = (raw_value or "").strip()
    if not value:
        return None, None

    normalized = value.replace(",", ".")
    try:
        parsed = float(normalized)
    except ValueError:
        return None, f"{label}: escriba un número válido."

    if parsed < min_value or parsed > max_value:
        return None, f"{label}: valor fuera de rango esperado ({min_value:g}-{max_value:g})."
    return parsed, None


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="trIAje — Apoyo al Triaje Clínico",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg-app:         #FAFBFC;
    --bg-card:        #FFFFFF;
    --bg-panel:       #F4F6F8;
    --border:         #D8E2EA;
    --text-primary:   #0B1F33;
    --text-secondary: #526579;
    --text-muted:     #8795A1;
    --btn-bg:         #005EB8;
    --btn-hover:      #004B93;
    --btn-disabled:   #D2DDE7;
    --focus-ring:     #005EB8;
    --focus-ring-a:   rgba(0,94,184,0.16);
    --shadow-sm:      0 1px 2px rgba(31,45,61,0.05);
    --shadow-md:      0 4px 12px rgba(31,45,61,0.08);
    --radius:         6px;
    --font-sans:      'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono:      ui-monospace, 'SF Mono', 'Cascadia Code', Consolas, monospace;
    --field-error:    #DC2626;
    --field-error-bg: #FEF2F2;
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

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in-up { animation: fadeInUp 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.68; }
}

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

/* ── Logo & Header ── */
.logo-wrap { display: flex; align-items: center; gap: 0.85rem; margin-bottom: 0.25rem; }
.logo-text {
    font-family: var(--font-sans);
    font-size: 1.85rem; font-weight: 700;
    color: #1F2D3D; letter-spacing: 0; line-height: 1;
}
.logo-text .brand-ia {
    color: var(--btn-bg);
    font-weight: 800;
}
.author-text { font-family: var(--font-sans); font-size: 0.82rem; font-weight: 500; color: var(--text-secondary); margin-bottom: 0.35rem; }
.tagline { font-size: 0.84rem; color: var(--text-secondary); font-weight: 400; letter-spacing: 0; margin-top: 0.2rem; line-height: 1.5; }
.header-divider { height: 1px; background: var(--border); margin: 1.5rem 0 2rem 0; }

/* ── Header con boton derecho ── */
.header-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 0.5rem;
}

/* ── Labels ── */
.sec-label {
    font-family: var(--font-sans); font-size: 0.82rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: #0F172A; font-weight: 700;
    margin-bottom: 1.2rem; display: block;
}

/* ── Cards Generales ── */
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 6px; box-shadow: var(--shadow-sm); padding: 1.4rem;
}

/* ── Form centrado en Fase 1 ── */
.form-container {
    max-width: 860px;
    margin: 0 auto;
}
.form-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow-sm);
    padding: 1.3rem 1.4rem 1.4rem;
    margin-bottom: 1rem;
}
.form-section-title {
    font-family: var(--font-sans);
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #0F172A;
    font-weight: 700;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.form-section-title .required-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--btn-bg);
    border-radius: 50%;
}
.field-error-msg {
    font-size: 0.72rem;
    color: var(--field-error);
    margin-top: 0.15rem;
    line-height: 1.4;
}

/* ── Widget styling ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stMultiSelect"] > div > div {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    min-height: 38px !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--focus-ring) !important;
    box-shadow: 0 0 0 3px var(--focus-ring-a) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stPills"] label,
[data-testid="stSegmentedControl"] label,
[data-testid="stSlider"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    display: block !important;
    font-size: 0.78rem !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    margin-bottom: 0.15rem !important;
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; font-family: var(--font-sans) !important;
    font-size: 0.9rem !important; color: var(--text-primary) !important;
    line-height: 1.6 !important; padding: 1.1rem 1.2rem !important;
    box-shadow: var(--shadow-sm) !important; resize: vertical !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--text-muted) !important; }

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

/* ── Button ── */
[data-testid="stButton"],
[data-testid="stFormSubmitButton"] {
    display: flex !important;
    justify-content: center !important;
}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button {
    background-color: var(--btn-bg) !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-family: var(--font-sans) !important; font-size: 1.05rem !important;
    font-weight: 600 !important; padding: 0.95rem 2rem !important;
    width: min(100%, 280px) !important; margin-top: 1rem !important;
    box-shadow: 0 2px 8px rgba(0,102,204,0.18) !important;
    transition: all 0.2s ease !important;
    min-height: 52px !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: var(--btn-hover) !important; transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(0,102,204,0.22) !important;
}
[data-testid="stButton"] > button[kind="primary"]:disabled,
[data-testid="stFormSubmitButton"] > button:disabled {
    background-color: var(--btn-disabled) !important; color: #94A3B8 !important;
    transform: none !important; box-shadow: none !important;
    cursor: not-allowed !important;
}
/* ── Botón secundario (Nuevo caso) ── */
[data-testid="stButton"] > button[kind="secondary"] {
    background-color: var(--bg-card) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    font-family: var(--font-sans) !important; font-size: 0.82rem !important;
    font-weight: 500 !important; padding: 0.55rem 1.2rem !important;
    width: min(100%, 220px) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    background-color: var(--bg-panel) !important;
    border-color: #C0CCDA !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { font-family: var(--font-mono) !important; font-size: 0.76rem !important; color: var(--focus-ring) !important; }

/* ── Vitals grid ── */
.vitals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.65rem; margin-top: 0.5rem; }

/* ── Probabilidades ── */
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

/* ── SHAP factor cards ── */
.shap-factor-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.shap-factor-card {
    border: 1px solid var(--border);
    border-left: 4px solid var(--btn-bg);
    border-radius: 6px;
    padding: 0.85rem 0.9rem;
}
.shap-factor-title {
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-secondary);
    font-weight: 700;
    margin-bottom: 0.55rem;
}
.shap-factor-row {
    display: flex;
    justify-content: space-between;
    gap: 0.7rem;
    align-items: baseline;
    border-top: 1px solid #E6EDF3;
    padding-top: 0.42rem;
    margin-top: 0.42rem;
    font-size: 0.82rem;
    color: var(--text-primary);
}
.shap-factor-row strong {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.shap-factor-empty {
    color: var(--text-muted);
    font-size: 0.8rem;
}

/* ── Alertas clínicas ── */
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
.triage-warning {
    background: #FFF4DE;
    border: 1px solid #F2D79B;
    border-left: 4px solid #B66A00;
    border-radius: 6px;
    padding: 0.55rem 0.75rem;
    margin: 0.45rem 0;
    color: #92400E;
    font-size: 0.78rem;
    line-height: 1.45;
}

/* ── Fase 2: Processing screen ── */
.processing-screen {
    min-height: 58vh;
    padding: 1.4rem;
    max-width: 720px;
    margin: 0 auto;
}
.processing-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-sm);
    padding: clamp(1.2rem, 3vw, 1.7rem);
}
.processing-title {
    font-family: var(--font-sans);
    font-size: 1.2rem;
    font-weight: 700;
    color: #1F2D3D;
    margin-bottom: 0.3rem;
    letter-spacing: 0;
}
.processing-subtitle {
    font-size: 0.84rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 1.1rem;
}
.processing-steps {
    text-align: left;
    width: 100%;
    margin-top: 0.8rem;
}
.processing-step {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.45rem 0;
    font-size: 0.84rem;
    line-height: 1.4;
    transition: opacity 0.3s ease, color 0.3s ease;
}
.processing-step.done {
    color: #16A34A;
    font-weight: 500;
}
.processing-step.active {
    color: var(--btn-bg);
    font-weight: 600;
    animation: pulse 1.5s infinite;
}
.processing-step.pending {
    color: #CBD5E1;
}
.step-icon {
    width: 18px;
    height: 18px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.processing-active-copy {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    color: #334155;
    font-size: 0.82rem;
    line-height: 1.55;
    padding: 0.75rem 0.85rem;
    margin-top: 0.8rem;
}
.processing-areas {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.45rem;
    margin-top: 0.9rem;
}
.processing-area {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.5rem 0.55rem;
    background: #F8FAFC;
    color: #64748B;
    font-size: 0.68rem;
    font-weight: 700;
    text-align: center;
}
.processing-area.done {
    background: #F0FDF4;
    border-color: #BBF7D0;
    color: #166534;
}
.processing-area.active {
    background: #EFF6FF;
    border-color: #93C5FD;
    color: #1D4ED8;
}
.screen-cover {
    position: fixed;
    inset: 0;
    z-index: 999999;
    background: var(--bg-app);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: clamp(1.2rem, 4vw, 3rem);
    overflow: auto;
}
.button-row-center {
    max-width: 360px;
    margin: 0.2rem auto 0;
}
.button-row-center.secondary {
    max-width: 260px;
    margin-top: 0.3rem;
}
.result-shell {
    animation: fadeInUp 0.45s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

/* ── Fase 3: Resumen vector ── */
.vector-summary-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow-sm);
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
}
.vector-section-title {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.vector-value {
    font-family: var(--font-sans);
    font-size: 0.88rem;
    color: var(--text-primary);
    font-weight: 500;
    line-height: 1.5;
}
.vector-vital-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
}
.vector-vital-item {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 0.5rem 0.6rem;
    min-width: 0;
    overflow: hidden;
}
.vector-vital-label {
    font-size: 0.58rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    font-weight: 700;
    white-space: nowrap;
}
.vector-vital-value {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1;
    margin-top: 0.15rem;
}

/* ── Quick result (hero ESI) ── */
.hero-esi {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    animation: fadeInUp 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}
.hero-esi-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.15rem;
    opacity: 0.8;
}
.hero-esi-title {
    font-family: var(--font-mono);
    font-size: 1.45rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: 0;
}
.hero-esi-confidence {
    text-align: right;
}
.hero-esi-confidence .label {
    font-size: 0.65rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 0.15rem;
    font-weight: 600;
}
.hero-esi-confidence .value {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    font-weight: 700;
    color: #0F172A;
}

/* ── Nuevo caso CTA ── */
.nuevo-caso-footer {
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

/* ── Scroll to top (injected via JS) ── */
.scroll-top-trigger {
    position: absolute;
    width: 0; height: 0;
    overflow: hidden;
}

/* ── Responsive ── */
@media (max-width: 960px) {
    .vitals-grid { grid-template-columns: repeat(2, 1fr); }
    .shap-factor-grid { grid-template-columns: 1fr; }
    .block-container { padding: 1.5rem 1.25rem 3rem !important; }
    .vector-vital-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-esi { flex-direction: column; align-items: flex-start; gap: 0.8rem; }
    .processing-areas { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .button-row-center,
    .button-row-center.secondary { max-width: none; }
    [data-testid="stButton"] > button,
    [data-testid="stFormSubmitButton"] > button { width: 100% !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
if "fase" not in st.session_state:
    st.session_state.fase = 1
if "ultimo_vector" not in st.session_state:
    st.session_state.ultimo_vector = None
if "ultima_narrativa" not in st.session_state:
    st.session_state.ultima_narrativa = ""
if "ultimo_resultado_ml" not in st.session_state:
    st.session_state.ultimo_resultado_ml = None
if "ultima_explicacion_shap" not in st.session_state:
    st.session_state.ultima_explicacion_shap = None
if "ultimo_umbral_alerta_a1" not in st.session_state:
    st.session_state.ultimo_umbral_alerta_a1 = 0.40
if "ultima_metrica_tiempos" not in st.session_state:
    st.session_state.ultima_metrica_tiempos = {}
if "vector_revisado_pendiente" not in st.session_state:
    st.session_state.vector_revisado_pendiente = None
if "datos_formulario" not in st.session_state:
    st.session_state.datos_formulario = None


# ─────────────────────────────────────────────────────────────
# HEADER (siempre visible)
# ─────────────────────────────────────────────────────────────
def _render_header():
    """Render the app header. Adds 'Nuevo caso' button in Fase 3."""
    if st.session_state.fase == 3:
        col_title, col_btn = st.columns([8, 3], gap="large")
        with col_title:
            st.markdown(
                '<div class="logo-wrap">'
                '<span class="logo-text">tr<span class="brand-ia">IA</span>je</span>'
                '</div>'
                '<div class="author-text">Carlos Rubio Martínez</div>'
                '<p class="tagline">Apoyo estructurado al triaje clínico &nbsp;&middot;&nbsp; TFG</p>',
                unsafe_allow_html=True,
            )
        with col_btn:
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            if st.button("Nuevo episodio", type="secondary", key="nuevo_caso_top"):
                _reset_estado()
    else:
        st.markdown(
            '<div class="logo-wrap">'
            '<span class="logo-text">tr<span class="brand-ia">IA</span>je</span>'
            '</div>'
            '<div class="author-text">Carlos Rubio Martínez</div>'
            '<p class="tagline">Apoyo estructurado al triaje clínico &nbsp;&middot;&nbsp; TFG</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)


def _reset_estado():
    """Limpia el estado completo y vuelve a Fase 1."""
    keys_to_clear = [
        "fase", "ultimo_vector", "ultima_narrativa",
        "ultimo_resultado_ml", "ultima_explicacion_shap",
        "ultimo_umbral_alerta_a1", "ultima_metrica_tiempos",
        "datos_formulario", "vector_revisado_pendiente",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ─────────────────────────────────────────────────────────────
# FASE 1 — FORMULARIO CLÍNICO GUIADO
# ─────────────────────────────────────────────────────────────
def _render_fase_1():
    st.markdown('<span class="sec-label">Registro de episodio</span>', unsafe_allow_html=True)

    with st.form("triage_intake_form_narrativo", border=False):
        form_col, side_col = st.columns([2, 1], gap="large")

        with form_col:
            st.markdown(
                '<div class="form-section-title"><span class="required-dot"></span> Datos mínimos</div>',
                unsafe_allow_html=True,
            )
            edad_col, sexo_col, llegada_col = st.columns([0.9, 1, 1.25], gap="small")
            with edad_col:
                edad_raw = st.text_input("Edad *", placeholder="años")
            with sexo_col:
                sexo = st.segmented_control("Sexo *", ["M", "F", "Otro"], width="stretch")
            with llegada_col:
                metodo_llegada = st.selectbox(
                    "Método de llegada",
                    ["desconocido", "autonomo", "ambulancia", "helicoptero", "otro"],
                    index=0,
                )

            st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="form-section-title"><span class="required-dot"></span> Relato clínico guiado</div>',
                unsafe_allow_html=True,
            )
            motivo_col, duracion_col = st.columns([1.45, 1], gap="small")
            with motivo_col:
                motivo_consulta = st.text_input(
                    "Motivo principal o síntoma guía *",
                    placeholder="Dolor abdominal desde la tarde",
                )
            with duracion_col:
                duracion_sintomas = st.text_input(
                    "Duración / evolución",
                    placeholder="Desde hace unas horas",
                )
            relato_clinico = st.text_area(
                "Relato clínico breve",
                placeholder=(
                    "Paciente consciente y orientado. Refiere dolor abdominal progresivo. "
                    "Niega dolor torácico."
                ),
                height=150,
            )

            st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="form-section-title"><span class="required-dot"></span> Constantes vitales y dolor</div>',
                unsafe_allow_html=True,
            )
            ta_s_col, ta_d_col, fc_col = st.columns(3, gap="small")
            with ta_s_col:
                presion_sistolica_raw = st.text_input("TA sistólica", placeholder="mmHg")
            with ta_d_col:
                presion_diastolica_raw = st.text_input("TA diastólica", placeholder="mmHg")
            with fc_col:
                frecuencia_cardiaca_raw = st.text_input("FC", placeholder="lpm")

            fr_col, spo2_col, temp_col = st.columns(3, gap="small")
            with fr_col:
                frecuencia_respiratoria_raw = st.text_input("FR", placeholder="rpm")
            with spo2_col:
                saturacion_oxigeno_raw = st.text_input("SpO₂", placeholder="%")
            with temp_col:
                temperatura_raw = st.text_input("Temperatura", placeholder="°C")

            dolor_valor = st.select_slider(
                "Dolor EVA/NRS",
                options=["No registrado", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                value="No registrado",
            )

            st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="form-section-title">Contexto clínico</div>',
                unsafe_allow_html=True,
            )
            antecedentes = st.text_area(
                "Antecedentes relevantes",
                placeholder="Hipertensión arterial",
                height=82,
            )
            medicacion = st.text_area(
                "Medicación habitual",
                placeholder="Enalapril",
                height=82,
            )
            observaciones = st.text_area(
                "Observaciones de triaje",
                placeholder="Acude acompañado. Buen estado general.",
                height=82,
            )

        with side_col:
            st.markdown(
                '<div class="form-section-title">Discriminadores de prioridad</div>',
                unsafe_allow_html=True,
            )
            signos_alarma = st.pills(
                "Marcar si está presente",
                SIGNOS_ALARMA,
                selection_mode="multi",
                default=[],
                width="stretch",
            )
            # Variable mapping for the academic report: narrative -> symptoms/BERT;
            # comorbidity/medication/vitals/arrival -> tabular model features.

        edad, edad_error = _parse_int_input(edad_raw, "Edad", 0, 120)
        presion_sistolica, ta_s_error = _parse_int_input(
            presion_sistolica_raw, "TA sistólica", 40, 300
        )
        presion_diastolica, ta_d_error = _parse_int_input(
            presion_diastolica_raw, "TA diastólica", 20, 200
        )
        frecuencia_cardiaca, fc_error = _parse_int_input(
            frecuencia_cardiaca_raw, "FC", 20, 300
        )
        frecuencia_respiratoria, fr_error = _parse_int_input(
            frecuencia_respiratoria_raw, "FR", 4, 60
        )
        saturacion_oxigeno, spo2_error = _parse_float_input(
            saturacion_oxigeno_raw, "SpO₂", 50, 100
        )
        temperatura, temp_error = _parse_float_input(
            temperatura_raw, "Temperatura", 30, 45
        )

        avisos_numericos = [
            aviso
            for aviso in (
                edad_error, ta_s_error, ta_d_error,
                fc_error, fr_error, spo2_error, temp_error,
            )
            if aviso
        ]

        datos_formulario = TriageFormData(
            edad=edad,
            sexo=sexo or "",
            metodo_llegada=metodo_llegada,
            motivo_consulta=motivo_consulta,
            sintomas_frecuentes=signos_alarma or [],
            sintomas_adicionales=relato_clinico,
            signos_alarma=signos_alarma or [],
            duracion_sintomas=duracion_sintomas,
            presion_sistolica=presion_sistolica,
            presion_diastolica=presion_diastolica,
            frecuencia_cardiaca=frecuencia_cardiaca,
            frecuencia_respiratoria=frecuencia_respiratoria,
            saturacion_oxigeno=saturacion_oxigeno,
            temperatura=temperatura,
            nivel_dolor=None if dolor_valor == "No registrado" else int(dolor_valor),
            antecedentes=antecedentes,
            medicacion=medicacion,
            observaciones=observaciones,
        )

        tiene_constante = any([
            presion_sistolica is not None,
            frecuencia_cardiaca is not None,
            frecuencia_respiratoria is not None,
            saturacion_oxigeno is not None,
            temperatura is not None,
        ])

        _, submit_col, _ = st.columns([1, 1.15, 1])
        with submit_col:
            analizar = st.form_submit_button(
                "Revisar episodio",
                type="primary",
                icon=":material/monitor_heart:",
            )

    if analizar:
        errores_bloqueo = []
        if edad is None:
            errores_bloqueo.append("Introduzca la **edad** del paciente.")
        if not sexo:
            errores_bloqueo.append("Seleccione el **sexo** del paciente.")
        if not (motivo_consulta or "").strip():
            errores_bloqueo.append("Introduzca el **motivo principal** de consulta.")
        if not tiene_constante:
            errores_bloqueo.append("Registre al menos **una constante vital** (TA, FC, FR, SpO₂ o temperatura).")
        errores_bloqueo.extend(avisos_numericos)

        if errores_bloqueo:
            for err in errores_bloqueo:
                st.error(err)
        else:
            for aviso in validar_entrada_triaje(datos_formulario):
                st.markdown(
                    f'<div class="triage-warning">{aviso}</div>',
                    unsafe_allow_html=True,
                )

            narrativa = generar_narrativa_triaje_texto_libre(datos_formulario)
            st.session_state.datos_formulario = datos_formulario
            st.session_state.ultima_narrativa = narrativa.strip()
            st.session_state.ultimo_vector = None
            st.session_state.ultimo_resultado_ml = None
            st.session_state.fase = 2
            st.rerun()

    st.markdown(
        """<p style="font-size:0.72rem; color:#94A3B8; margin-top:0.9rem; line-height:1.75; text-align:center;">El formulario guía la recogida mínima y prepara una revisión estructurada. No sustituye el criterio profesional.</p>""",
        unsafe_allow_html=True,
    )


PASOS_PROCESAMIENTO = [
    {
        "label": "Organizando relato clínico",
        "help": "Ordenando el motivo de consulta y el relato registrado.",
        "area": "Relato",
    },
    {
        "label": "Identificando datos relevantes",
        "help": "Separando síntomas, duración y contexto clínico.",
        "area": "Relato",
    },
    {
        "label": "Revisando constantes vitales",
        "help": "Comprobando constantes, temperatura y dolor.",
        "area": "Constantes",
    },
    {
        "label": "Comprobando antecedentes y medicación",
        "help": "Relacionando antecedentes, tratamiento habitual y observaciones.",
        "area": "Contexto",
    },
    {
        "label": "Buscando señales de prioridad",
        "help": "Revisando hallazgos que pueden requerir atención preferente.",
        "area": "Prioridad",
    },
    {
        "label": "Preparando revisión editable",
        "help": "Preparando los datos para que pueda corregirlos o confirmarlos.",
        "area": "Revisión",
    },
    {
        "label": "Revisión lista",
        "help": "Los datos están preparados para su comprobación.",
        "area": "Revisión",
    },
]
PROGRESO_PROCESAMIENTO = [0.10, 0.22, 0.36, 0.50, 0.64, 0.92, 1.0]
AREAS_PROCESAMIENTO = ["Relato", "Constantes", "Contexto", "Prioridad", "Revisión"]

_ICON_DONE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
_ICON_ACTIVE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0066CC" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/></svg>'
_ICON_PENDING = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#CBD5E1" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/></svg>'


def _render_processing_steps_html(current_step: int) -> str:
    """Genera el HTML de la lista de pasos con estado visual."""
    steps_html = ""
    for i, paso in enumerate(PASOS_PROCESAMIENTO):
        if i < current_step:
            css_class = "done"
            icon = _ICON_DONE
        elif i == current_step:
            css_class = "active"
            icon = _ICON_ACTIVE
        else:
            css_class = "pending"
            icon = _ICON_PENDING
        label = html.escape(paso["label"])
        steps_html += (
            f'<div class="processing-step {css_class}">'
            f'<span class="step-icon">{icon}</span>'
            f'<span>{label}</span>'
            f'</div>'
        )
    return steps_html


def _render_processing_areas_html(current_step: int) -> str:
    active_area = PASOS_PROCESAMIENTO[current_step]["area"]
    completed_areas = {
        paso["area"]
        for paso in PASOS_PROCESAMIENTO[:current_step]
        if paso["area"] != active_area
    }
    areas_html = ""
    for area in AREAS_PROCESAMIENTO:
        css_class = "done" if area in completed_areas else "active" if area == active_area else ""
        areas_html += f'<div class="processing-area {css_class}">{html.escape(area)}</div>'
    return areas_html


def _render_processing_screen(step_index: int, progress_pct: float, elapsed_s: float) -> str:
    step = PASOS_PROCESAMIENTO[step_index]
    return (
        '<div class="processing-screen">'
        '<div class="processing-card">'
        '<div class="processing-title">Preparando revisión clínica</div>'
        '<div class="processing-subtitle">'
        'Preparando los datos para su revisión antes de calcular el nivel de prioridad.'
        '</div>'
        '<div class="processing-active-copy">'
        f'<strong>{html.escape(step["label"])}</strong><br>'
        f'{html.escape(step["help"])}'
        '</div>'
        f'<div class="processing-areas">{_render_processing_areas_html(step_index)}</div>'
        f'<div class="processing-steps">{_render_processing_steps_html(step_index)}</div>'
        '<p style="font-size:0.74rem; color:#94A3B8; margin-top:0.8rem; line-height:1.5;">'
        'Esto tomará unos instantes.'
        '</p>'
        '</div>'
        '</div>'
    )


PASOS_RESULTADO = [
    {
        "label": "Confirmando datos revisados",
        "help": "Comprobando que los datos editados son válidos.",
        "area": "Revisión",
    },
    {
        "label": "Calculando nivel de prioridad",
        "help": "Preparando el nivel sugerido y la alerta de seguridad.",
        "area": "Prioridad",
    },
    {
        "label": "Preparando probabilidades",
        "help": "Ordenando la distribución por niveles ESI.",
        "area": "Probabilidades",
    },
    {
        "label": "Generando factores explicativos",
        "help": "Preparando los factores que influyen en el resultado.",
        "area": "Factores",
    },
    {
        "label": "Componiendo resumen final",
        "help": "Montando la pantalla final para mostrarla completa.",
        "area": "Resultado",
    },
]
PROGRESO_RESULTADO = [0.12, 0.34, 0.58, 0.82, 1.0]
AREAS_RESULTADO = ["Revisión", "Prioridad", "Probabilidades", "Factores", "Resultado"]


def _render_result_steps_html(current_step: int) -> str:
    steps_html = ""
    for i, paso in enumerate(PASOS_RESULTADO):
        if i < current_step:
            css_class = "done"
            icon = _ICON_DONE
        elif i == current_step:
            css_class = "active"
            icon = _ICON_ACTIVE
        else:
            css_class = "pending"
            icon = _ICON_PENDING
        steps_html += (
            f'<div class="processing-step {css_class}">'
            f'<span class="step-icon">{icon}</span>'
            f'<span>{html.escape(paso["label"])}</span>'
            f'</div>'
        )
    return steps_html


def _render_result_areas_html(current_step: int) -> str:
    active_area = PASOS_RESULTADO[current_step]["area"]
    completed_areas = {
        paso["area"]
        for paso in PASOS_RESULTADO[:current_step]
        if paso["area"] != active_area
    }
    areas_html = ""
    for area in AREAS_RESULTADO:
        css_class = "done" if area in completed_areas else "active" if area == active_area else ""
        areas_html += f'<div class="processing-area {css_class}">{html.escape(area)}</div>'
    return areas_html


def _render_result_preparation_screen(step_index: int, progress_pct: float) -> str:
    step = PASOS_RESULTADO[step_index]
    return (
        '<div class="processing-screen">'
        '<div class="processing-card">'
        '<div class="processing-title">Preparando resultado clínico</div>'
        '<div class="processing-subtitle">'
        'Preparando resumen, prioridad y factores explicativos.'
        '</div>'
        '<div class="processing-active-copy">'
        f'<strong>{html.escape(step["label"])}</strong><br>'
        f'{html.escape(step["help"])}'
        '</div>'
        f'<div class="processing-areas">{_render_result_areas_html(step_index)}</div>'
        f'<div class="processing-steps">{_render_result_steps_html(step_index)}</div>'
        '</div>'
        '</div>'
    )


def _render_result_transition_cover() -> str:
    return (
        '<div class="screen-cover">'
        f'{_render_result_preparation_screen(0, PROGRESO_RESULTADO[0])}'
        '</div>'
    )


def _preparar_explicacion_resultado(predictor, result, clase_predicha: int) -> dict:
    try:
        import plotly.graph_objects as go

        explainer = crear_explainer(predictor._clf, clase=clase_predicha)
        shap_explanation = generar_shap_explanation_object(
            explainer, result.X, clase=clase_predicha, modelo=predictor._clf,
        )

        shap_vals = np.array(shap_explanation.values).flatten()
        feat_names = np.array(shap_explanation.feature_names)
        feat_names_clean = [clean_feature_name(str(f)) for f in feat_names]
        shap_a_favor, shap_en_contra = split_shap_factors(
            [str(f) for f in feat_names],
            [float(v) for v in shap_vals],
            top_n=3,
        )

        df_shap = pd.DataFrame({"Feature": feat_names_clean, "SHAP": shap_vals})
        df_shap["Abs_SHAP"] = df_shap["SHAP"].abs()
        df_shap = df_shap.sort_values(by="Abs_SHAP", ascending=True).tail(10)
        df_shap["Feature"] = [f + (" " * i) for i, f in enumerate(df_shap["Feature"])]

        fill_colores = [
            "rgba(220, 38, 38, 0.85)" if v > 0 else "rgba(14, 165, 233, 0.85)"
            for v in df_shap["SHAP"]
        ]
        line_colores = ["#B91C1C" if v > 0 else "#0284C7" for v in df_shap["SHAP"]]

        fig = go.Figure(go.Bar(
            x=df_shap["SHAP"],
            y=df_shap["Feature"],
            orientation="h",
            marker=dict(color=fill_colores, line=dict(color=line_colores, width=1.5)),
            text=[f"{v:+.3f}" for v in df_shap["SHAP"]],
            textposition="outside",
            textfont=dict(
                family="IBM Plex Sans, sans-serif",
                size=11,
                color="#1F2D3D",
                weight="bold",
            ),
            hovertemplate="<b>%{y}</b><br>Impacto en el resultado: %{x:+.3f}<extra></extra>",
        ))
        fig.update_layout(
            margin=dict(l=0, r=45, t=15, b=10),
            height=340,
            bargap=0.3,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#253242", size=12),
            xaxis=dict(
                showgrid=True,
                gridcolor="#E2E8F0",
                zeroline=True,
                zerolinecolor="#94A3B8",
                zerolinewidth=1.5,
                showticklabels=True,
                tickfont=dict(color="#64748B", size=10),
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color="#0F172A", size=11, weight="bold"),
            ),
            showlegend=False,
            dragmode=False,
        )

        resultado_shap = explicar_prediccion(
            explainer, result.X, clase_predicha=clase_predicha, modelo=predictor._clf,
        )
        return {
            "ok": True,
            "fig": fig,
            "shap_a_favor": shap_a_favor,
            "shap_en_contra": shap_en_contra,
            "top_positivas": resultado_shap.top_positivas[:3],
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "fig": None,
            "shap_a_favor": [],
            "shap_en_contra": [],
            "top_positivas": [],
            "error": str(e),
        }

def _vector_review_text(value):
    return "" if value is None else str(value)


def _render_fase_2():
    st.markdown('<script>window.parent.document.querySelector("section.main").scrollTo(0, 0);</script>', unsafe_allow_html=True)

    narrativa = st.session_state.ultima_narrativa
    if not narrativa:
        st.session_state.fase = 1
        st.rerun()
        return

    if st.session_state.ultimo_vector is None:
        container = st.empty()
        progress_bar = st.progress(0)

        def _update_display(step_index: int, progress_pct: float, elapsed_s: float = 0.0):
            container.markdown(
                _render_processing_screen(step_index, progress_pct, elapsed_s),
                unsafe_allow_html=True,
            )
            progress_bar.progress(progress_pct)

        try:
            start_time = time.monotonic()
            _update_display(0, PROGRESO_PROCESAMIENTO[0], 0.0)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_extraer_fn, narrativa, modelo=MODELO_LLM_FIJO)
                while not future.done():
                    elapsed = time.monotonic() - start_time
                    if elapsed < 1.2:
                        step_index = 0
                        progress_pct = PROGRESO_PROCESAMIENTO[0]
                    elif elapsed < 2.4:
                        step_index = 1
                        progress_pct = PROGRESO_PROCESAMIENTO[1]
                    elif elapsed < 3.6:
                        step_index = 2
                        progress_pct = PROGRESO_PROCESAMIENTO[2]
                    elif elapsed < 5.0:
                        step_index = 3
                        progress_pct = PROGRESO_PROCESAMIENTO[3]
                    elif elapsed < 8.0:
                        step_index = 4
                        progress_pct = PROGRESO_PROCESAMIENTO[4]
                    else:
                        step_index = 5
                        progress_pct = min(0.94, 0.78 + ((elapsed - 8.0) * 0.01))
                    _update_display(step_index, progress_pct, elapsed)
                    time.sleep(0.25)
                vector = future.result()
            elapsed = time.monotonic() - start_time
            _update_display(5, PROGRESO_PROCESAMIENTO[5], elapsed)
            st.session_state.ultimo_vector = vector
            st.session_state.ultima_metrica_tiempos = {
                **st.session_state.get("ultima_metrica_tiempos", {}),
                "extraccion_llm_s": elapsed,
            }
            st.session_state.ultimo_resultado_ml = None
            st.session_state.ultima_explicacion_shap = None
            st.session_state.vector_revisado_pendiente = None
            _update_display(6, PROGRESO_PROCESAMIENTO[6], elapsed)
            time.sleep(0.2)
            st.rerun()
        except ConnectionError:
            container.empty()
            progress_bar.empty()
            st.error("No se ha podido conectar con el procesamiento local. Compruebe que está iniciado y vuelva a intentarlo.")
            if st.button("Volver al registro", type="secondary"):
                st.session_state.fase = 1
                st.rerun()
        except Exception as e:
            container.empty()
            progress_bar.empty()
            st.error(f"No se ha podido preparar la revisión. Detalle técnico: {e}")
            if st.button("Volver al registro", type="secondary"):
                st.session_state.fase = 1
                st.rerun()
        return

    vector = st.session_state.ultimo_vector
    st.markdown('<span class="sec-label">Revisión del episodio</span>', unsafe_allow_html=True)

    alertas = validar_vector_clinico(vector, narrativa)
    for alerta in [a for a in alertas if a.nivel in (NivelAlerta.WARNING, NivelAlerta.ERROR)]:
        css_class = "warning-validation" if alerta.nivel == NivelAlerta.WARNING else "alert-clinical"
        st.markdown(
            f'<div class="{css_class}"><strong>[{html.escape(alerta.campo)}]</strong> '
            f'{html.escape(alerta.mensaje)}</div>',
            unsafe_allow_html=True,
        )

    with st.form("review_vector_form", border=False):
        col_left, col_right = st.columns([1.1, 1], gap="large")

        with col_left:
            st.markdown(
                '<div class="form-section-title"><span class="required-dot"></span> Datos clínicos revisables</div>',
                unsafe_allow_html=True,
            )
            edad_col, sexo_col, llegada_col = st.columns([0.8, 0.9, 1.2], gap="small")
            with edad_col:
                edad_raw = st.text_input("Edad", value=_vector_review_text(vector.edad))
            with sexo_col:
                sexo_opts = ["M", "F", "Otro"]
                sexo = st.selectbox(
                    "Sexo",
                    sexo_opts,
                    index=sexo_opts.index(vector.sexo) if vector.sexo in sexo_opts else 2,
                )
            with llegada_col:
                llegada_opts = ["desconocido", "autonomo", "ambulancia", "helicoptero", "otro"]
                metodo_llegada = st.selectbox(
                    "Método de llegada",
                    llegada_opts,
                    index=(
                        llegada_opts.index(vector.metodo_llegada)
                        if vector.metodo_llegada in llegada_opts
                        else 0
                    ),
                )

            sintomas_presentes = st.text_area(
                "Síntomas / motivo",
                value=join_review_terms_display(vector.sintomas_presentes),
                height=92,
            )
            duracion_sintomas = st.text_input(
                "Duración o evolución",
                value=vector.duracion_sintomas or "",
            )
            patologias_previas = st.text_area(
                "Antecedentes relevantes",
                value=join_review_terms_display(vector.patologias_previas),
                height=82,
            )
            medicacion_habitual = st.text_area(
                "Medicación habitual",
                value=join_review_terms_display(vector.medicacion_habitual),
                height=82,
            )

        with col_right:
            st.markdown(
                '<div class="form-section-title">Constantes y dolor</div>',
                unsafe_allow_html=True,
            )
            ta_s_col, ta_d_col = st.columns(2, gap="small")
            with ta_s_col:
                presion_sistolica_raw = st.text_input(
                    "TA sistólica", value=_vector_review_text(vector.presion_sistolica)
                )
            with ta_d_col:
                presion_diastolica_raw = st.text_input(
                    "TA diastólica", value=_vector_review_text(vector.presion_diastolica)
                )
            fc_col, fr_col = st.columns(2, gap="small")
            with fc_col:
                frecuencia_cardiaca_raw = st.text_input(
                    "FC", value=_vector_review_text(vector.frecuencia_cardiaca)
                )
            with fr_col:
                frecuencia_respiratoria_raw = st.text_input(
                    "FR", value=_vector_review_text(vector.frecuencia_respiratoria)
                )
            spo2_col, temp_col = st.columns(2, gap="small")
            with spo2_col:
                saturacion_oxigeno_raw = st.text_input(
                    "SpO₂", value=_vector_review_text(vector.saturacion_oxigeno)
                )
            with temp_col:
                temperatura_raw = st.text_input(
                    "Temperatura", value=_vector_review_text(vector.temperatura)
                )
            dolor_valor = st.select_slider(
                "Dolor EVA/NRS",
                options=["No registrado", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                value="No registrado" if vector.nivel_dolor is None else int(vector.nivel_dolor),
            )

            st.markdown(
                '<div class="card" style="margin-top:1rem;">'
                '<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase;'
                ' color:#475569; font-weight:700; margin-bottom:0.55rem;">Registro estructurado</div>'
                f'<p style="font-size:0.76rem; color:#64748B; line-height:1.65; margin:0;">{html.escape(narrativa)}</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        _, confirm_col, _ = st.columns([1, 1.2, 1])
        with confirm_col:
            confirm = st.form_submit_button(
                "Confirmar y calcular nivel",
                type="primary",
                icon=":material/check_circle:",
            )

    _, back_col, _ = st.columns([1.35, 1, 1.35])
    with back_col:
        if st.button("Volver al registro", type="secondary"):
            st.session_state.ultimo_vector = None
            st.session_state.ultimo_resultado_ml = None
            st.session_state.ultima_explicacion_shap = None
            st.session_state.vector_revisado_pendiente = None
            st.session_state.fase = 1
            st.rerun()

    if confirm:
        edad, edad_error = _parse_int_input(edad_raw, "Edad", 0, 120)
        presion_sistolica, ta_s_error = _parse_int_input(
            presion_sistolica_raw, "TA sistólica", 40, 300
        )
        presion_diastolica, ta_d_error = _parse_int_input(
            presion_diastolica_raw, "TA diastólica", 20, 200
        )
        frecuencia_cardiaca, fc_error = _parse_int_input(
            frecuencia_cardiaca_raw, "FC", 20, 300
        )
        frecuencia_respiratoria, fr_error = _parse_int_input(
            frecuencia_respiratoria_raw, "FR", 4, 60
        )
        saturacion_oxigeno, spo2_error = _parse_float_input(
            saturacion_oxigeno_raw, "SpO₂", 50, 100
        )
        temperatura, temp_error = _parse_float_input(
            temperatura_raw, "Temperatura", 30, 45
        )
        errores = [
            err
            for err in (
                edad_error, ta_s_error, ta_d_error, fc_error,
                fr_error, spo2_error, temp_error,
            )
            if err
        ]
        if edad is None:
            errores.append("Edad: campo obligatorio.")

        if errores:
            for err in errores:
                st.error(err)
            return

        try:
            vector_revisado = construir_vector_desde_revision(
                edad=edad,
                sexo=sexo,
                sintomas_presentes=sintomas_presentes,
                patologias_previas=patologias_previas,
                medicacion_habitual=medicacion_habitual,
                presion_sistolica=presion_sistolica,
                presion_diastolica=presion_diastolica,
                frecuencia_cardiaca=frecuencia_cardiaca,
                frecuencia_respiratoria=frecuencia_respiratoria,
                saturacion_oxigeno=saturacion_oxigeno,
                temperatura=temperatura,
                nivel_dolor=None if dolor_valor == "No registrado" else int(dolor_valor),
                duracion_sintomas=duracion_sintomas,
                metodo_llegada=metodo_llegada,
            )
        except Exception as e:
            st.error(f"No se pudo validar la revisión del episodio: {e}")
            return

        st.session_state.ultimo_vector = vector_revisado
        st.session_state.vector_revisado_pendiente = vector_revisado
        st.session_state.ultimo_resultado_ml = None
        st.session_state.ultima_explicacion_shap = None
        st.session_state.fase = 25
        st.markdown(_render_result_transition_cover(), unsafe_allow_html=True)
        st.rerun()


def _render_fase_resultado_preparacion():
    st.markdown('<script>window.parent.document.querySelector("section.main").scrollTo(0, 0);</script>', unsafe_allow_html=True)

    vector_revisado = st.session_state.vector_revisado_pendiente
    narrativa = st.session_state.ultima_narrativa
    if vector_revisado is None or not narrativa:
        st.session_state.fase = 1
        st.rerun()
        return

    container = st.empty()
    progress_bar = st.progress(0)

    def _update_display(step_index: int):
        container.markdown(
            _render_result_preparation_screen(step_index, PROGRESO_RESULTADO[step_index]),
            unsafe_allow_html=True,
        )
        progress_bar.progress(PROGRESO_RESULTADO[step_index])

    total_start = time.monotonic()
    metricas = dict(st.session_state.get("ultima_metrica_tiempos", {}))

    try:
        _update_display(0)
        time.sleep(0.15)

        _update_display(1)
        pred_start = time.monotonic()
        predictor = _cargar_predictor()
        result = predictor.predict(vector_revisado, narrativa)
        metricas["prediccion_s"] = time.monotonic() - pred_start
        st.session_state.ultimo_umbral_alerta_a1 = getattr(
            predictor, "_warning_threshold_a1", 0.40
        )

        _update_display(2)
        time.sleep(0.12)

        _update_display(3)
        shap_start = time.monotonic()
        explicacion = _preparar_explicacion_resultado(
            predictor, result, result.clase_predicha
        )
        metricas["shap_s"] = time.monotonic() - shap_start

        _update_display(4)
        metricas["preparacion_resultado_s"] = time.monotonic() - total_start
        if "extraccion_llm_s" in metricas:
            metricas["total_hasta_resultado_s"] = (
                metricas["extraccion_llm_s"] + metricas["preparacion_resultado_s"]
            )

        st.session_state.ultimo_vector = vector_revisado
        st.session_state.ultimo_resultado_ml = result
        st.session_state.ultima_explicacion_shap = explicacion
        st.session_state.ultima_metrica_tiempos = metricas
        st.session_state.vector_revisado_pendiente = None
        time.sleep(0.2)
        st.session_state.fase = 3
        st.rerun()
    except Exception as e:
        container.empty()
        progress_bar.empty()
        st.error(f"Error preparando el resultado clínico: {e}")
        if st.button("Volver a la revisión", type="secondary"):
            st.session_state.vector_revisado_pendiente = None
            st.session_state.fase = 2
            st.rerun()


def _render_fase_3():
    # Scroll to top al entrar en resultados
    st.markdown('<script>window.parent.document.querySelector("section.main").scrollTo(0, 0);</script>', unsafe_allow_html=True)

    vector = st.session_state.ultimo_vector
    result = st.session_state.ultimo_resultado_ml
    narrativa = st.session_state.ultima_narrativa

    if vector is None or result is None:
        st.session_state.fase = 1
        st.rerun()
        return

    probas_fila = result.probas
    clase_predicha = result.clase_predicha
    confianza = result.confianza
    alerta_a1_activada = result.alerta_a1_activada
    esi = ESI_CONFIG[clase_predicha]
    umbral_alerta_a1 = st.session_state.get("ultimo_umbral_alerta_a1", 0.40)
    explicacion_shap = st.session_state.get("ultima_explicacion_shap")

    # ── Layout 2 columnas ──
    st.markdown('<div class="result-shell">', unsafe_allow_html=True)
    col_vector, col_resultado = st.columns([5, 7], gap="large")

    # ── Columna izquierda: Resumen del vector clínico ──
    with col_vector:
        st.markdown('<span class="sec-label">Resumen del episodio</span>', unsafe_allow_html=True)

        # Paciente + Banner éxito combinados
        edad_v = getattr(vector, "edad", None)
        sexo_v = getattr(vector, "sexo", None)
        paciente_parts = []
        if edad_v is not None:
            paciente_parts.append(f"{edad_v} años")
        if sexo_v:
            paciente_parts.append(str(sexo_v))
        paciente_str = " · ".join(paciente_parts) if paciente_parts else "No registrado"

        st.markdown(
            '<div style="background:#FFF; border:1px solid #DDE5EC; border-radius:8px;'
            ' box-shadow:0 1px 2px rgba(31,45,61,0.05); padding:1rem 1.1rem; margin-bottom:0.8rem;">'
            '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">'
            '<span style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase;'
            ' color:#475569; font-weight:700;">Paciente</span>'
            '<span style="display:inline-flex; align-items:center; gap:0.35rem; background:#F0FDF4;'
            ' border:1px solid #BBF7D0; border-radius:12px; padding:0.15rem 0.55rem;">'
            '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#16A34A"'
            ' stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
            '<polyline points="20 6 9 17 4 12"/></svg>'
            '<span style="font-size:0.6rem; color:#166534; font-weight:600; text-transform:uppercase;'
            ' letter-spacing:0.06em;">Revisado</span></span></div>'
            f'<div style="font-size:0.95rem; color:#1F2D3D; font-weight:600;">{html.escape(paciente_str)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Constantes vitales — todo en una sola st.markdown, sin indentación
        vitales = [
            ("TA Sist.", getattr(vector, "presion_sistolica", None)),
            ("TA Diast.", getattr(vector, "presion_diastolica", None)),
            ("FC", getattr(vector, "frecuencia_cardiaca", None)),
            ("FR", getattr(vector, "frecuencia_respiratoria", None)),
            ("SpO\u2082", getattr(vector, "saturacion_oxigeno", None)),
            ("Temp.", getattr(vector, "temperatura", None)),
        ]

        items_html = ""
        for nombre, valor in vitales:
            v_str = str(valor) if valor is not None else "\u2014"
            v_col = "#0F172A" if valor is not None else "#CBD5E1"
            items_html += (
                f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px;'
                f' padding:0.45rem 0.55rem; min-width:0; overflow:hidden;">'
                f'<div style="font-size:0.58rem; letter-spacing:0.08em; text-transform:uppercase;'
                f' color:#64748B; font-weight:700; white-space:nowrap;">{nombre}</div>'
                f'<div style="font-family:var(--font-mono); font-size:1rem; font-weight:700;'
                f' color:{v_col}; line-height:1; margin-top:0.15rem;">{v_str}</div></div>'
            )

        st.markdown(
            '<div style="background:#FFF; border:1px solid #DDE5EC; border-radius:8px;'
            ' box-shadow:0 1px 2px rgba(31,45,61,0.05); padding:1rem 1.1rem; margin-bottom:0.8rem;">'
            '<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase;'
            ' color:#475569; font-weight:700; margin-bottom:0.5rem;">Constantes vitales</div>'
            f'<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.45rem;">{items_html}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Tags: síntomas, antecedentes, medicación
        sintomas = display_clinical_terms(getattr(vector, "sintomas_presentes", []) or [])
        patologias = display_clinical_terms(getattr(vector, "patologias_previas", []) or [])
        medicacion_v = display_clinical_terms(getattr(vector, "medicacion_habitual", []) or [])

        def _render_tag(text, bg, tc, bc):
            return (
                f'<span style="display:inline-flex; background:{bg}; color:{tc};'
                f' border:1px solid {bc}; font-size:0.76rem; font-weight:500;'
                f' padding:3px 10px; border-radius:16px; margin:0 4px 5px 0;'
                f' line-height:1.4;">{html.escape(text)}</span>'
            )

        secciones_tags = [
            ("Síntomas presentes", sintomas, "#EFF6FF", "#1D4ED8", "#93C5FD"),
            ("Antecedentes", patologias, "#FFF7ED", "#C2410C", "#FDBA74"),
            ("Medicación", medicacion_v, "#F0FDF4", "#15803D", "#86EFAC"),
        ]

        any_tags = False
        for titulo, items, bg, tc, bc in secciones_tags:
            if items:
                any_tags = True
                tags_joined = "".join(_render_tag(i, bg, tc, bc) for i in items)
                st.markdown(
                    '<div style="background:#FFF; border:1px solid #DDE5EC; border-radius:8px;'
                    ' box-shadow:0 1px 2px rgba(31,45,61,0.05); padding:1rem 1.1rem; margin-bottom:0.8rem;">'
                    f'<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase;'
                    f' color:#475569; font-weight:700; margin-bottom:0.5rem;">{titulo}</div>'
                    f'<div style="display:flex; flex-wrap:wrap;">{tags_joined}</div></div>',
                    unsafe_allow_html=True,
                )

        if not any_tags:
            st.markdown(
                '<div style="background:#FFF; border:1px solid #DDE5EC; border-radius:8px;'
                ' box-shadow:0 1px 2px rgba(31,45,61,0.05); padding:1rem 1.1rem; margin-bottom:0.8rem;">'
                '<div style="font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase;'
                ' color:#475569; font-weight:700; margin-bottom:0.5rem;">Datos clínicos</div>'
                '<p style="font-size:0.8rem; color:#94A3B8; margin:0;">'
                'No se detectaron síntomas, antecedentes ni medicación.</p></div>',
                unsafe_allow_html=True,
            )

    # ── Columna derecha: Predicción + SHAP ──
    with col_resultado:
        st.markdown('<span class="sec-label">Nivel de prioridad sugerido</span>', unsafe_allow_html=True)

        # Alerta A1
        if alerta_a1_activada:
            st.markdown(f"""
            <div class="alert-clinical" style="display:flex; align-items:flex-start; gap:0.6rem;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; margin-top:2px;">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <div>
                    <strong>Alerta de seguridad clínica:</strong>
                    P(acuity=1) = {probas_fila[0]:.1%} supera el umbral de aviso
                    ({umbral_alerta_a1:.0%}).
                    Revisar posible criticidad A1 antes de confirmar el nivel sugerido.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Validación semántica warnings
        alertas = validar_vector_clinico(vector, narrativa)
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

        # Hero ESI
        st.markdown(f"""
        <div class="hero-esi" style="background:{esi['bg']}; border:2px solid {esi['border']};">
            <div style="display:flex; align-items:center; gap:1.1rem;">
                <div style="width:46px; height:46px; border-radius:50%; background:#FFF;
                            display:flex; align-items:center; justify-content:center;
                            box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{esi['color']}"
                         stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                    </svg>
                </div>
                <div>
                    <div class="hero-esi-label" style="color:{esi['color']};">Nivel sugerido</div>
                    <div class="hero-esi-title" style="color:{esi['color']};">{esi['label']}</div>
                </div>
            </div>
            <div class="hero-esi-confidence">
                <div class="label">Confianza</div>
                <div class="value">{confianza:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Barras de probabilidad
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
                f"<div style='font-family:var(--font-mono); font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; color:#0F172A; margin-bottom:1.1rem; font-weight:700;'>Probabilidades por nivel ESI</div>"
                f"{barras_html}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("Probabilidades por nivel ESI no disponibles para este resultado.")

    # ── Sección SHAP — Ancho completo, debajo de las dos columnas ──
    st.markdown('<div style="height:1px; background:#DDE5EC; margin:1.5rem 0 2rem 0;"></div>', unsafe_allow_html=True)

    if explicacion_shap and explicacion_shap.get("ok"):
        st.markdown(
            '<span class="sec-label">Factores que influyen en el resultado</span>',
            unsafe_allow_html=True,
        )

        def _render_shap_factors(title, factors, border_color, bg_color):
            if not factors:
                rows = "<span class='shap-factor-empty'>Sin factores destacados.</span>"
            else:
                rows = "".join(
                    "<div class='shap-factor-row'>"
                    f"<span>{html.escape(factor.nombre)}</span>"
                    f"<strong>{factor.valor:+.3f}</strong>"
                    "</div>"
                    for factor in factors
                )
            return (
                f"<div class='shap-factor-card' style='border-left-color:{border_color}; background:{bg_color};'>"
                f"<div class='shap-factor-title'>{title}</div>"
                f"{rows}"
                "</div>"
            )

        st.markdown(
            "<div class='shap-factor-grid'>"
            + _render_shap_factors(
                "Peso a favor del nivel sugerido",
                explicacion_shap.get("shap_a_favor", []),
                "#0066CC",
                "#F7FBFF",
            )
            + _render_shap_factors(
                "Peso en contra",
                explicacion_shap.get("shap_en_contra", []),
                "#607080",
                "#F7F9FA",
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            explicacion_shap["fig"],
            use_container_width=True,
            config={"displayModeBar": False},
        )

        top_positivas = explicacion_shap.get("top_positivas", [])
        if top_positivas:
            tops = ", ".join(
                f"<strong>{clean_feature_name(f.nombre)}</strong> ({f.shap_value:+.3f})"
                for f in top_positivas
            )
            st.markdown(
                f'<p style="font-size:0.73rem; color:#64748B; line-height:1.7; margin-top:0.3rem;">Factores principales ↑: {tops}</p>',
                unsafe_allow_html=True,
            )
    else:
        detalle = ""
        if explicacion_shap and explicacion_shap.get("error"):
            detalle = f': {html.escape(explicacion_shap["error"])}'
        st.markdown(
            f'<p style="font-size:0.75rem; color:#94A3B8; margin-top:0.5rem;">Explicación no disponible{detalle}</p>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


    # ── CTA: Nuevo caso (fondo) ──
    st.markdown('<div style="height:1px; background:#DDE5EC; margin:2rem 0 1.5rem 0;"></div>', unsafe_allow_html=True)
    col_spacer_l, col_btn_bottom, col_spacer_r = st.columns([3, 4, 3])
    with col_btn_bottom:
        if st.button("Nuevo episodio", type="primary", key="nuevo_caso_bottom",
                      icon=":material/refresh:"):
            _reset_estado()

    st.markdown('<div style="text-align:center; margin-top:2rem; padding-top:1rem; font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);">Carlos Rubio Mart\u00ednez &middot; TFG</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────
_render_header()

fase_actual = st.session_state.get("fase", 1)

if fase_actual == 1:
    _render_fase_1()
elif fase_actual == 2:
    _render_fase_2()
elif fase_actual == 25:
    _render_fase_resultado_preparacion()
elif fase_actual == 3:
    _render_fase_3()
else:
    st.session_state.fase = 1
    st.rerun()

# Footer (solo en Fase 1)
if fase_actual == 1:
    st.markdown("""
    <div style="text-align:center; margin-top:4rem; padding-top:1.5rem; border-top:1px solid var(--border);
                font-size:0.75rem; color:var(--text-muted); font-family:var(--font-mono);">
        Carlos Rubio Martínez &middot; TFG
    </div>
    """, unsafe_allow_html=True)
