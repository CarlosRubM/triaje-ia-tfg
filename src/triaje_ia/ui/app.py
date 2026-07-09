"""
src/triaje_ia/ui/app.py
──────────────────────────
Interfaz principal — Sistema de apoyo al triaje clínico.
Ejecutar: uv run streamlit run src/triaje_ia/ui/app.py
"""

from concurrent.futures import ThreadPoolExecutor
import html
import time
import uuid
import streamlit as st
import streamlit.components.v1 as components

from triaje_ia.llm.validator import validar_vector_clinico, NivelAlerta
from triaje_ia.ui.clinical_form import (
    TriageFormData,
    generar_narrativa_triaje_texto_libre,
    validar_entrada_triaje,
)
from triaje_ia.ui.vector_review import construir_vector_desde_revision
from triaje_ia.ui.clinical_display import (
    display_clinical_terms,
    join_review_terms_display,
)
from triaje_ia.ui.shap_presenter import clean_feature_name, select_balanced_shap_factors


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _cargar_predictor():
    from triaje_ia.inference.predictor import TriajePredictor
    return TriajePredictor()


@st.cache_resource(show_spinner=False)
def _cargar_extractor():
    from triaje_ia.llm.factory import crear_extractor
    return crear_extractor("ollama")


@st.cache_resource(show_spinner=False)
def _cargar_explainer(_modelo):
    from triaje_ia.ml.explicabilidad import crear_explainer
    return crear_explainer(_modelo)


MODELO_LLM_FIJO = "llama3.1:8b-instruct-q4_K_M"

ESI_CONFIG = {
    1: {"label": "ESI 1 — Resucitación",    "color": "#DC2626", "bg": "#FEF2F2", "border": "#FECACA"},
    2: {"label": "ESI 2 — Emergencia",       "color": "#EA580C", "bg": "#FFF7ED", "border": "#FED7AA"},
    3: {"label": "ESI 3 — Urgente",          "color": "#CA8A04", "bg": "#FEFCE8", "border": "#FEF08A"},
    4: {"label": "ESI 4 — Menos urgente",    "color": "#16A34A", "bg": "#F0FDF4", "border": "#BBF7D0"},
    5: {"label": "ESI 5 — No urgente",       "color": "#2563EB", "bg": "#EFF6FF", "border": "#BFDBFE"},
}

SINTOMAS_FRECUENTES = [
    "dolor toracico", "disnea", "fiebre", "dolor abdominal",
    "mareo/sincope", "traumatismo", "vomitos", "alteracion neurologica",
]

SIGNOS_ALARMA = [
    "mal estado general", "compromiso respiratorio",
    "alteracion del nivel de consciencia", "dolor toracico activo",
    "deficit neurologico focal", "sospecha de sepsis/shock",
    "sangrado activo", "trauma mayor",
]


def _parse_int_input(raw_value, label, min_value, max_value):
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


def _parse_float_input(raw_value, label, min_value, max_value):
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
# ────────────────────────────────────────────────────────────
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
    font-size: 16px;
    --bg-app:         #FAFBFC;
    --bg-card:        #FFFFFF;
    --bg-panel:       #F2F7F8;
    --border:         #CBD5E0;
    --text-primary:   #0F172A;
    --text-secondary: #486581;
    --text-muted:     #5C7080;
    --btn-bg:         #0E7490;
    --btn-hover:      #0C6A80;
    --btn-disabled:   #C5D5DF;
    --focus-ring:     #0E7490;
    --focus-ring-a:   rgba(14,116,144,0.18);
    --shadow-sm:      0 1px 2px rgba(16,42,67,0.05);
    --shadow-md:      0 4px 12px rgba(16,42,67,0.08);
    --radius:         6px;
    --font-sans:      'Figtree', 'Noto Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono:      ui-monospace, 'SF Mono', 'Cascadia Code', Consolas, monospace;
    --field-error:    #DC2626;
    --ease-out-expo:  cubic-bezier(0.16, 1, 0.3, 1);
    --ease-standard:  cubic-bezier(0.4, 0, 0.2, 1);
    --dur-fast:       160ms;
    --dur-base:       260ms;
    --dur-slow:       380ms;
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
.block-container { padding: clamp(1.5rem, 3vw, 2.5rem) clamp(1.5rem, 4vw, 3.5rem) 5rem !important; max-width: 1240px !important; }
[data-testid="column"] { padding: 0 !important; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn var(--dur-base) var(--ease-out-expo) both; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in-up { animation: fadeInUp var(--dur-slow) var(--ease-out-expo) both; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

@keyframes welcomeStatusPulse {
    0%, 100% { opacity: 0.35; transform: scale(0.92); }
    50% { opacity: 1; transform: scale(1); }
}

.initial-welcome {
    position: fixed; inset: 0; z-index: 2147483600;
    background: var(--bg-app);
    display: flex; align-items: center; justify-content: center;
    padding: 2rem; opacity: 1; visibility: visible;
    transition: opacity 320ms ease, visibility 320ms ease;
}
.initial-welcome.is-leaving {
    opacity: 0; visibility: hidden; pointer-events: none;
}
.initial-welcome-content {
    width: min(680px, 100%); text-align: center;
    animation: fadeInUp 520ms var(--ease-out-expo) both;
}
.initial-welcome-brand {
    font-size: clamp(2.8rem, 7vw, 4.4rem); line-height: 1;
    font-weight: 750; letter-spacing: -0.05em; color: #1F2D3D;
}
.initial-welcome-brand .brand-ia { color: var(--btn-bg); font-weight: 800; }
.initial-welcome-title {
    margin-top: 1.6rem; font-size: clamp(1.25rem, 3vw, 1.65rem);
    line-height: 1.3; font-weight: 700; color: var(--text-primary);
}
.initial-welcome-copy {
    max-width: 620px; margin: 0.8rem auto 0; color: var(--text-secondary);
    font-size: clamp(1.15rem, 2.5vw, 1.35rem); line-height: 1.55;
    font-weight: 500;
}
.initial-welcome-status {
    display: inline-flex; align-items: center; gap: 0.65rem;
    margin-top: 1.7rem; color: var(--text-muted); font-size: 0.9rem;
}
.initial-welcome-dot {
    width: 8px; height: 8px; border-radius: 999px; background: var(--btn-bg);
    animation: welcomeStatusPulse 1.35s ease-in-out infinite;
}

.app-header {
    display: flex; align-items: center; gap: 1.25rem; min-height: 72px;
}
.logo-wrap { display: flex; align-items: center; flex-shrink: 0; }
.logo-text {
    font-family: var(--font-sans);
    font-size: 2.1rem; font-weight: 700;
    color: #1F2D3D; letter-spacing: -0.035em; line-height: 1;
}
.logo-text .brand-ia { color: var(--btn-bg); font-weight: 800; }
.header-copy { border-left: 1px solid var(--border); padding-left: 1.25rem; }
.header-title {
    color: var(--text-primary); font-size: 1.18rem;
    font-weight: 650; line-height: 1.35;
}
.header-meta {
    color: var(--text-secondary); font-size: 1rem;
    font-weight: 500; line-height: 1.5; margin-top: 0.2rem;
}
.header-meta-separator { color: var(--btn-bg); padding: 0 0.35rem; }
.header-divider { height: 1px; background: var(--border); margin: 1.25rem 0 2rem 0; }

.app-footer {
    margin-top: 3rem; padding: 1.5rem 1rem 0;
    border-top: 1px solid var(--border); text-align: center;
}
.app-footer-brand {
    color: var(--text-primary); font-size: 1.22rem;
    font-weight: 700; letter-spacing: -0.02em;
}
.app-footer-brand .brand-ia { color: var(--btn-bg); font-weight: 800; }
.app-footer-title {
    color: var(--text-primary); font-size: 1.05rem;
    font-weight: 600; margin-top: 0.35rem;
}
.app-footer-meta {
    color: var(--text-secondary); font-size: 0.95rem;
    line-height: 1.5; margin-top: 0.2rem;
}
.text-support { color: var(--text-secondary); font-size: 0.875rem; line-height: 1.55; }
.text-note { color: var(--text-secondary); font-size: 0.8125rem; line-height: 1.5; }
.review-narrative { color: #64748B; font-size: 0.875rem; line-height: 1.5; margin: 0; }
.result-intro { color: #64748B; font-size: 0.9rem; line-height: 1.5; margin: 0.35rem 0 0; }
.empty-clinical-state { color: var(--text-muted); font-size: 0.875rem; line-height: 1.5; margin: 0; }
.explanation-unavailable { color: var(--text-secondary); font-size: 0.8125rem; line-height: 1.5; margin-top: 0.5rem; }
.st-key-nuevo_caso_top [data-testid="stButton"] { justify-content: flex-end !important; }
.st-key-nuevo_caso_top [data-testid="stButton"] > button[kind="secondary"] {
    min-height: 44px !important; max-width: 168px !important; margin-top: 0 !important;
    border-color: var(--border) !important; box-shadow: none !important;
    color: var(--text-secondary) !important; font-size: 0.8125rem !important;
}
.st-key-nuevo_caso_top [data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--btn-bg) !important; color: var(--btn-hover) !important;
}

.sec-label {
    font-family: var(--font-sans); font-size: 0.875rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: #0F172A; font-weight: 700;
    margin-bottom: 1.2rem; display: block;
}

.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; box-shadow: var(--shadow-sm); padding: 1.4rem;
}

[data-testid="stForm"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; box-shadow: var(--shadow-md) !important;
    padding: clamp(1.15rem, 2.5vw, 1.75rem) !important;
}
.form-spacer { height: 1rem; }

.form-section-title {
    font-family: var(--font-sans); font-size: 0.875rem;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: #0F172A; font-weight: 700; margin-bottom: 0.9rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.form-section-title .required-dot {
    display: inline-block; width: 6px; height: 6px;
    background: var(--btn-bg); border-radius: 50%;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stMultiSelect"] > div > div {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; min-height: 44px !important;
    box-shadow: none !important; color: var(--text-primary) !important;
    font-size: 1rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--focus-ring) !important;
    box-shadow: 0 0 0 3px var(--focus-ring-a) !important; outline: none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stPills"] label,
[data-testid="stSegmentedControl"] label,
[data-testid="stSlider"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label {
    display: block !important; font-size: 0.875rem !important;
    color: var(--text-primary) !important; font-weight: 600 !important;
    margin-bottom: 0.25rem !important;
}

[data-testid="stTextArea"] textarea {
    background-color: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; font-family: var(--font-sans) !important;
    font-size: 1rem !important; color: var(--text-primary) !important;
    line-height: 1.6 !important; padding: 1.1rem 1.2rem !important;
    box-shadow: var(--shadow-sm) !important; resize: none !important; overflow-y: auto !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--text-secondary) !important; font-size: 0.8125rem !important; line-height: 1.5 !important;
}

[data-testid="stBaseButton-pills"],
[data-testid="stBaseButton-pillsActive"],
[data-testid="stBaseButton-segmented_controlActive"],
[data-testid="stBaseButton-segmented_control"] {
    min-height: 44px !important; padding: 0.6rem 0.75rem !important;
    color: var(--text-primary) !important; font-size: 0.875rem !important;
    box-sizing: border-box !important; line-height: 1.25 !important;
}
[data-testid="stBaseButton-pillsActive"],
[data-testid="stBaseButton-segmented_controlActive"] {
    background: var(--active-bg) !important; border-color: var(--btn-bg) !important;
    color: var(--btn-hover) !important;
}
[data-testid="stBaseButton-pills"] p,
[data-testid="stBaseButton-pillsActive"] p,
[data-testid="stBaseButton-segmented_control"] p,
[data-testid="stBaseButton-segmented_controlActive"] p {
    font-weight: 500 !important; letter-spacing: normal !important; width: 100% !important;
}
.st-key-minimum_data_controls [data-testid="stTextInput"] input,
.st-key-minimum_data_controls [data-testid="stSelectbox"] > div > div,
.st-key-minimum_data_controls [data-testid="stBaseButton-segmented_control"],
.st-key-minimum_data_controls [data-testid="stBaseButton-segmented_controlActive"] {
    height: 44px !important; min-height: 44px !important; box-sizing: border-box !important;
}
[data-testid="stButtonGroup"]:has([data-testid="stBaseButton-pills"]) [role="group"] {
    display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 0.5rem !important;
}
[data-testid="stButtonGroup"]:has([data-testid="stBaseButton-pills"]) [data-testid="stBaseButton-pills"],
[data-testid="stButtonGroup"]:has([data-testid="stBaseButton-pills"]) [data-testid="stBaseButton-pillsActive"] {
    width: 100% !important; height: auto !important; padding: 0.6rem 0.5rem !important;
}
[data-testid="stBaseButton-pills"] p,
[data-testid="stBaseButton-pillsActive"] p {
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; line-height: 1.25 !important;
}
[data-testid="stBaseButton-pills"]:focus-visible,
[data-testid="stBaseButton-segmented_control"]:focus-visible,
[data-testid="stButton"] button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible {
    outline: 3px solid var(--focus-ring-a) !important; outline-offset: 2px !important;
}

[data-testid="stButton"],
[data-testid="stFormSubmitButton"] {
    display: flex !important; justify-content: center !important; width: 100% !important;
}
[data-testid="stElementContainer"]:has([data-testid="stButton"]),
[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]),
[data-testid="stElementContainer"]:has([data-testid="stButton"]) > div,
[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) > div {
    width: 100% !important;
}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button {
    background-color: var(--btn-bg) !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-family: var(--font-sans) !important; font-size: 1.05rem !important;
    font-weight: 600 !important; padding: 0.95rem 2rem !important;
    width: min(100%, 280px) !important; margin-top: 1rem !important;
    max-width: 280px !important; box-shadow: 0 2px 8px rgba(0,102,204,0.18) !important;
    transition: background-color 180ms ease, box-shadow 180ms ease, border-color 180ms ease !important;
    min-height: 52px !important; letter-spacing: 0.01em !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: var(--btn-hover) !important; transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(0,102,204,0.22) !important;
}
[data-testid="stButton"] > button[kind="primary"]:disabled,
[data-testid="stFormSubmitButton"] > button:disabled {
    background-color: var(--btn-disabled) !important; color: #94A3B8 !important;
    transform: none !important; box-shadow: none !important; cursor: not-allowed !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background-color: var(--bg-card) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: var(--font-sans) !important; font-size: 0.875rem !important;
    font-weight: 500 !important; padding: 0.55rem 1.2rem !important;
    width: min(100%, 220px) !important; max-width: 220px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: background-color 180ms ease, box-shadow 180ms ease, border-color 180ms ease !important;
    min-height: 44px !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    background-color: var(--bg-panel) !important; border-color: #C0CCDA !important;
}

.prob-label {
    font-family: var(--font-mono); font-size: 0.8125rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-secondary); margin-bottom: 0.2rem;
}
.prob-bar-bg {
    height: 12px; background: #F8FAFC; border-radius: 10px;
    overflow: hidden; position: relative; width: 100%;
}
@keyframes growBar { from { width: 0%; } to { width: var(--target-width); } }
.prob-bar-fill {
    height: 100%; border-radius: 10px; min-width: 1.5%;
    width: var(--target-width);
    animation: growBar 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.shap-factor-grid {
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem; margin-bottom: 1rem;
}
.shap-factor-card {
    border: 1px solid var(--border); border-left: 4px solid var(--btn-bg);
    border-radius: 6px; padding: 0.85rem 0.9rem;
}
.shap-factor-title {
    font-size: 0.8125rem; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-secondary); font-weight: 700; margin-bottom: 0.55rem;
}
.shap-factor-row {
    display: flex; justify-content: space-between; gap: 0.7rem;
    align-items: baseline; border-top: 1px solid var(--border);
    padding-top: 0.42rem; margin-top: 0.42rem; font-size: 0.875rem; color: var(--text-primary);
}
.shap-factor-row strong {
    font-family: var(--font-mono); font-size: 0.8125rem; color: var(--text-secondary);
}
.shap-factor-empty { color: var(--text-muted); font-size: 0.8125rem; }

.shap-table-wrap { overflow-x: auto; margin: 0.8rem 0 1.2rem; }
.shap-table { width: 100%; border-collapse: collapse; background: #FFF; border: 1px solid var(--border); }
.shap-table th { background: #F8FAFC; color: #475569; font-size: 0.8125rem; text-transform: uppercase; letter-spacing: 0.045em; }
.shap-table th, .shap-table td { padding: 0.72rem 0.9rem; border-bottom: 1px solid var(--border); text-align: left; }
.shap-table td { color: var(--text-primary); font-size: 0.9375rem; }
.shap-table tr:last-child td { border-bottom: 0; }
.shap-table .effect-up { color: #0891B2; font-weight: 650; }
.shap-table .effect-down { color: #64748B; font-weight: 650; }
.shap-table .contribution { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-weight: 650; white-space: nowrap; }
.shap-table th:last-child { text-align: right; }

.st-key-review_vector_form [data-testid="stForm"] { padding: 0.85rem 1rem !important; }
.st-key-review_vector_form [data-testid="stVerticalBlock"] { gap: 0.42rem !important; }
.st-key-review_vector_form [data-testid="stTextArea"] textarea { min-height: 58px !important; height: 58px !important; padding: 0.55rem 0.7rem !important; line-height: 1.35 !important; }
.st-key-review_vector_form [data-testid="stExpander"] { margin-top: 0.15rem; }
.st-key-review_vector_form .form-section-title { margin-bottom: 0.2rem; }

.alert-clinical {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-left: 4px solid #DC2626; border-radius: 8px;
    padding: 0.75rem 1.1rem; margin-bottom: 0.8rem;
    font-size: 0.9rem; color: #991B1B; animation: fadeIn 0.4s ease-out;
}
.warning-validation {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-left: 4px solid #D97706; border-radius: 8px;
    padding: 0.6rem 1rem; margin-bottom: 0.6rem;
    font-size: 0.875rem; color: #92400E; animation: fadeIn 0.4s ease-out;
}
.triage-warning {
    background: #FFF4DE; border: 1px solid #F2D79B;
    border-left: 4px solid #B66A00; border-radius: 6px;
    padding: 0.55rem 0.75rem; margin: 0.45rem 0;
    color: #92400E; font-size: 0.8125rem; line-height: 1.45;
}

/* ── Fase 2: Processing screen ─ */
@keyframes phaseTransitionOut {
    from { opacity: 1; }
    to   { opacity: 0; }
}
.processing-screen.is-leaving {
    pointer-events: none;
    animation: phaseTransitionOut 0.5s ease-out 0.22s forwards;
}
.processing-screen {
    position: fixed; inset: 0; z-index: 99999; min-height: 100vh;
    padding: clamp(1.5rem, 4vh, 2.5rem) 1.25rem 2rem;
    background: var(--bg-app) !important;
    display: flex; justify-content: center; align-items: center; overflow-y: auto;
}
.processing-card {
    background: #FFFFFF; border: 1px solid var(--border);
    border-radius: 10px; box-shadow: var(--shadow-sm);
    padding: clamp(1.2rem, 3vw, 1.7rem); width: min(720px, 100%);
}
.processing-card.result-compact {
    width: min(720px, 100%); min-height: 410px;
    padding: clamp(2rem, 5vw, 3rem); text-align: center;
    display: flex; flex-direction: column; justify-content: center;
}
.result-processing-mark {
    width: 218px; height: 218px; margin: 0 auto 1.6rem;
    border: 2px solid #CFF3F8; border-radius: 50%;
    background: linear-gradient(145deg, #F8FEFF 0%, #ECFEFF 100%);
    box-shadow: 0 12px 30px rgba(8, 145, 178, 0.12);
    display: flex; align-items: center; justify-content: center;
}
.result-processing-mark svg { width: 164px; height: 112px; overflow: visible; }
.result-pulse-base, .result-pulse-progress {
    fill: none; stroke-width: 7; stroke-linecap: round; stroke-linejoin: round;
}
.result-pulse-base { stroke: #CBD5E1; }
.result-pulse-progress {
    stroke: #0891B2; filter: drop-shadow(0 0 4px rgba(8, 145, 178, 0.28));
}
.result-pulse-clip rect { transition: width 0.35s ease-out; }
.result-compact .processing-title {
    color: #64748B; font-size: clamp(0.95rem, 1.8vw, 1.12rem);
    font-weight: 550; margin: 0;
}
.processing-title {
    font-family: var(--font-sans); font-size: 1.2rem;
    font-weight: 700; color: #1F2D3D; margin-bottom: 0.3rem; letter-spacing: 0;
}
.processing-subtitle {
    font-size: 0.9rem; color: var(--text-secondary);
    line-height: 1.55; margin-bottom: 1.1rem;
}
.processing-progress-row { display: flex; align-items: center; margin: 1.1rem 0 0.25rem; }
.processing-progress-track {
    flex: 1; height: 9px; overflow: hidden;
    border-radius: 999px; background: #E2E8F0;
}
.processing-progress-fill {
    height: 100%; border-radius: inherit; background: #0891B2; transition: width 0.3s ease;
}
.processing-steps { text-align: left; width: 100%; margin-top: 0.8rem; }
.processing-step {
    display: flex; align-items: center; gap: 0.65rem;
    padding: 0.45rem 0; font-size: 0.9375rem; line-height: 1.4;
    transition: opacity 0.3s ease, color 0.3s ease;
}
.processing-step.done { color: #16A34A; font-weight: 500; }
.processing-step.active { color: var(--btn-bg); font-weight: 600; animation: pulse 1.5s infinite; }
.processing-step.pending { color: var(--text-muted); }
.step-icon { width: 18px; height: 18px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
.processing-active-copy {
    background: #F8FAFC; border: 1px solid var(--border);
    border-radius: 8px; color: #334155; font-size: 0.875rem;
    line-height: 1.55; padding: 0.75rem 0.85rem; margin-top: 0.8rem;
}
.processing-areas {
    display: grid; grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.45rem; margin-top: 0.9rem;
}
.processing-area {
    border: 1px solid var(--border); border-radius: 8px;
    padding: 0.5rem 0.55rem; background: #F8FAFC; color: #64748B;
    min-height: 44px; display: flex; align-items: center; justify-content: center;
    font-size: 0.8125rem; font-weight: 700; text-align: center;
}
.processing-area.done { background: #F0FDF4; border-color: #BBF7D0; color: #166534; }
.processing-area.active { background: #ECFEFF; border-color: #67D4E5; color: var(--btn-hover); }

.button-row-center { max-width: 360px; margin: 0.2rem auto 0; }
.button-row-center.secondary { max-width: 260px; margin-top: 0.3rem; }

/* ─ Fase 3: Resumen vector ── */
.vector-summary-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; box-shadow: var(--shadow-sm);
    padding: 1.2rem 1.3rem; margin-bottom: 1rem;
    animation: fadeInUp var(--dur-slow) var(--ease-out-expo) both;
}
.vector-section-title {
    font-size: 0.8125rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #475569; font-weight: 700; margin-bottom: 0.6rem;
}
.vector-value {
    font-family: var(--font-sans); font-size: 0.9375rem;
    color: var(--text-primary); font-weight: 500; line-height: 1.5;
}
.vector-vital-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; }
.vector-vital-item {
    background: #F8FAFC; border: 1px solid var(--border);
    border-radius: 6px; padding: 0.5rem 0.6rem; min-width: 0; overflow: hidden;
}
.vector-vital-label {
    font-size: 0.8125rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #64748B; font-weight: 700; white-space: nowrap;
}
.vector-vital-value {
    font-family: var(--font-mono); font-size: 1rem;
    font-weight: 700; color: #0F172A; line-height: 1; margin-top: 0.15rem;
}

.hero-esi {
    border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: var(--shadow-md);
    animation: fadeInUp var(--dur-slow) var(--ease-out-expo) both;
    animation-delay: 40ms;
}
.hero-esi-label {
    font-size: 0.8125rem; letter-spacing: 0.12em; text-transform: uppercase;
    font-weight: 700; margin-bottom: 0.15rem; opacity: 0.8;
}
.hero-esi-title {
    font-family: var(--font-mono); font-size: 1.6rem;
    font-weight: 700; line-height: 1; letter-spacing: 0;
}
.hero-esi-confidence { text-align: right; }
.hero-esi-confidence .label {
    font-size: 0.8125rem; letter-spacing: 0.05em; text-transform: uppercase;
    color: var(--text-secondary); margin-bottom: 0.15rem; font-weight: 600;
}
.hero-esi-confidence .value {
    font-family: var(--font-mono); font-size: 1.4rem;
    font-weight: 700; color: #0F172A;
}

.st-key-review_identity_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important;
}

@media (max-width: 1100px) {
    [data-testid="stHorizontalBlock"]:has(.st-key-review_identity_controls):has([data-testid="stExpander"]) {
        flex-direction: column !important;
        gap: 1.25rem !important;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-review_identity_controls):has([data-testid="stExpander"]) > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
}

@media (max-width: 960px) {
    .shap-factor-grid { grid-template-columns: 1fr; }
    .block-container { padding: 1.5rem 1.25rem 3rem !important; }
    .vector-vital-grid { grid-template-columns: repeat(2, 1fr); }
    .hero-esi { flex-direction: column; align-items: flex-start; gap: 0.8rem; }
    .processing-areas { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .button-row-center, .button-row-center.secondary { max-width: none; }
    [data-testid="stHorizontalBlock"]:has(.st-key-minimum_data_controls),
    [data-testid="stHorizontalBlock"]:has(.st-key-review_identity_controls):has([data-testid="stExpander"]),
    [data-testid="stHorizontalBlock"]:has(.hero-esi):has(.vector-summary-card),
    [data-testid="stHorizontalBlock"]:has(.app-header):has(.st-key-top_new_episode_action) {
        flex-direction: column !important;
        gap: 1.25rem !important;
    }
    [data-testid="stHorizontalBlock"]:has(.st-key-minimum_data_controls) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(.st-key-review_identity_controls):has([data-testid="stExpander"]) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(.hero-esi):has(.vector-summary-card) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(.app-header):has(.st-key-top_new_episode_action) > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    [data-testid="stHorizontalBlock"]:has(.app-header):has(.st-key-top_new_episode_action) .st-key-top_new_episode_action [data-testid="stButton"] {
        justify-content: flex-start !important;
    }
    [data-testid="stButton"] > button,
    [data-testid="stFormSubmitButton"] > button { width: 100% !important; }
    [data-testid="stForm"] { padding: 1rem !important; border-radius: 10px !important; box-shadow: var(--shadow-sm) !important; }
}
@media (max-width: 599px) {
    .processing-screen { inset: 0; min-height: 100vh; padding-top: 4.5rem; }
    .app-header { align-items: flex-start; flex-direction: column; gap: 0.65rem; }
    .logo-text { font-size: 1.95rem; }
    .header-copy { border-left: 0; padding-left: 0; }
    .header-title { font-size: 1rem; }
    .header-meta { font-size: 0.875rem; }
    .block-container { padding: 1.25rem 1rem 2.5rem !important; }
    .card { padding: 1.1rem; }
    .st-key-minimum_data_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
    .st-key-intake_narrative_row > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
    .st-key-intake_vitals_primary > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
    .st-key-intake_vitals_secondary > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
    .st-key-review_identity_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
    .st-key-review_action_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.75rem !important;
    }
    .st-key-minimum_data_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-intake_narrative_row > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-intake_vitals_primary > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-intake_vitals_secondary > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-review_identity_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-review_action_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    .sec-label, .form-section-title { line-height: 1.4; }
    .app-footer { margin-top: 2rem; padding-inline: 0; }
    .shap-table { min-width: 520px; }
    [data-testid="stPlotlyChart"] { min-height: 420px; }
    [data-testid="stButton"] > button,
    [data-testid="stFormSubmitButton"] > button { max-width: none !important; }
    [data-testid="stButtonGroup"]:has([data-testid="stBaseButton-pills"]) [role="group"] { grid-template-columns: 1fr !important; }
    .shap-table th, .shap-table td { padding: 0.65rem 0.6rem; }
    .shap-table td:first-child { white-space: normal; min-width: 145px; line-height: 1.25; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
        scroll-behavior: auto !important; transition-duration: 0.01ms !important;
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
if "vector_revisado_pendiente" not in st.session_state:
    st.session_state.vector_revisado_pendiente = None
if "datos_formulario" not in st.session_state:
    st.session_state.datos_formulario = None
if "scroll_to_top_pending" not in st.session_state:
    st.session_state.scroll_to_top_pending = False
if "new_episode_scroll_token" not in st.session_state:
    st.session_state.new_episode_scroll_token = ""
if "initial_welcome_pending" not in st.session_state:
    st.session_state.initial_welcome_pending = True
if "review_transition_token" not in st.session_state:
    st.session_state.review_transition_token = ""
if "result_reveal_pending" not in st.session_state:
    st.session_state.result_reveal_pending = False
if "result_transition_token" not in st.session_state:
    st.session_state.result_transition_token = ""


# ─────────────────────────────────────────────────────────────
# OVERLAY JS PERSISTENTE (fase 25 → fase 3)
# ─────────────────────────────────────────────────────────────
# El parpadeo del "vector de revisión" al llegar a los resultados finales
# viene de que el overlay de la fase 25 vive DENTRO del árbol que Streamlit
# reconcilia en cada rerun: si el rerun que activa la fase 3 tarda en asentar
# del todo su propio árbol (columnas, tabla SHAP, gráfico Plotly...), puede
# quedar una rendija por la que se cuela contenido de un estado anterior
# mientras Streamlit termina de resolverlo.
#
# La solución de fondo: sacar el overlay del árbol de Streamlit por completo.
# Se inyecta directamente en el documento padre vía JavaScript (mismo patrón
# que _scroll_to_top), como un <div> normal y corriente del DOM que Streamlit
# ni gestiona ni puede tocar. Como no es un elemento de Streamlit, ningún
# rerun puede hacerlo parpadear ni reemplazarlo por sorpresa: solo lo
# cambiamos nosotros, con JS, cuando nosotros decidimos.
#
# La revelación final tampoco se dispara "cuando el script de Python
# termina" (que es lo que fallaba antes), sino cuando de verdad aparece en
# el DOM real un marcador invisible que pintamos como último elemento de la
# fase 3. Es decir: esperamos una prueba objetiva de que el contenido final
# ya está montado, no una suposición de tiempo.
_JS_OVERLAY_ID = "triaje-transition-overlay"
_JS_READY_MARKER_PREFIX = "triaje-fase3-ready-"
_REVIEW_JS_OVERLAY_ID = "triaje-review-transition-overlay"
_REVIEW_READY_MARKER_PREFIX = "triaje-fase2-ready-"


def _ready_marker_id(transition_token: str) -> str:
    return f"{_JS_READY_MARKER_PREFIX}{transition_token}"


def _review_ready_marker_id(transition_token: str) -> str:
    return f"{_REVIEW_READY_MARKER_PREFIX}{transition_token}"


def _render_review_preparation_card_inner(step_index, progress_pct):
    step = PASOS_PROCESAMIENTO[step_index]
    progress_value = max(0, min(100, round(progress_pct * 100)))
    return (
        '<div class="processing-card">'
        '<div class="processing-title">Preparando revisión clínica</div>'
        '<div class="processing-subtitle">Preparando los datos para su revisión antes de calcular el nivel de prioridad.</div>'
        '<div class="processing-active-copy">'
        f'<strong>{html.escape(step["label"])}</strong><br>{html.escape(step["help"])}</div>'
        f'<div class="processing-areas">{_render_processing_areas_html(step_index)}</div>'
        f'<div class="processing-steps">{_render_processing_steps_html(step_index)}</div>'
        '<div class="processing-progress-row">'
        f'<div class="processing-progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{progress_value}">'
        f'<div class="processing-progress-fill" style="width:{progress_value}%"></div></div></div>'
        '<p style="font-size:0.8125rem; color:var(--text-secondary); margin-top:0.8rem; line-height:1.5;">Esto tomará unos instantes.</p>'
        '</div>'
    )


def _render_result_preparation_card_inner(step_index, progress_pct):
    """Igual que _render_result_preparation_screen pero sin el <div> exterior
    'processing-screen': el propio overlay JS hace de contenedor."""
    progress_value = max(0, min(100, round(progress_pct * 100)))
    return (
        '<div class="processing-card result-compact">'
        f'<div class="result-processing-mark" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{progress_value}">'
        '<svg viewBox="0 0 120 80" aria-hidden="true">'
        '<defs><clipPath id="result-pulse-clip" class="result-pulse-clip">'
        f'<rect x="0" y="0" height="80" style="width:{progress_value}%"/>'
        '</clipPath></defs>'
        '<path class="result-pulse-base" d="M4 42h20l9-19 15 38 15-53 16 45 10-23 9 12h18"/>'
        '<path class="result-pulse-progress" clip-path="url(#result-pulse-clip)" d="M4 42h20l9-19 15 38 15-53 16 45 10-23 9 12h18"/>'
        '</svg></div>'
        '<div class="processing-title">Calculando prioridad y explicación</div>'
        '</div>'
    )


def _cleanup_js_transition_artifacts():
    """Retira restos de transiciones anteriores antes de iniciar otro episodio."""
    components.html(
        f"""
        <script>
        (function() {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            if (parentWindow.__triajeTransitionController) {{
                parentWindow.__triajeTransitionController.cancelled = true;
                parentWindow.__triajeTransitionController = null;
            }}
            const overlay = doc.getElementById('{_JS_OVERLAY_ID}');
            if (overlay) overlay.remove();
            if (parentWindow.__triajeReviewTransitionController) {{
                parentWindow.__triajeReviewTransitionController.cancelled = true;
                parentWindow.__triajeReviewTransitionController = null;
            }}
            const reviewOverlay = doc.getElementById('{_REVIEW_JS_OVERLAY_ID}');
            if (reviewOverlay) reviewOverlay.remove();
            doc.querySelectorAll('[id^="{_JS_READY_MARKER_PREFIX}"]').forEach(
                marker => marker.remove()
            );
            doc.querySelectorAll('[id^="{_REVIEW_READY_MARKER_PREFIX}"]').forEach(
                marker => marker.remove()
            );
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _push_review_js_overlay(inner_html: str, transition_token: str):
    """Mantiene la carga de revisión fuera del árbol que Streamlit reconcilia."""
    safe_html = inner_html.replace("\\", "\\\\").replace("`", "\\`").replace("</script", "<\\/script")
    components.html(
        f"""
        <script>
        (function() {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const token = '{transition_token}';
            let overlay = doc.getElementById('{_REVIEW_JS_OVERLAY_ID}');
            if (overlay && overlay.dataset.transitionToken !== token) {{
                overlay.remove();
                overlay = null;
            }}
            if (!overlay) {{
                overlay = doc.createElement('div');
                overlay.id = '{_REVIEW_JS_OVERLAY_ID}';
                overlay.className = 'processing-screen';
                overlay.dataset.mountedAt = String(Date.now());
                doc.body.appendChild(overlay);
                const log = parentWindow.__triajeReviewTransitionLog ||= [];
                log.push({{token, event:'overlay-mounted', at:Date.now()}});
                if (log.length > 200) log.splice(0, log.length - 200);
            }}
            overlay.dataset.transitionToken = token;
            overlay.style.zIndex = '2147483647';
            overlay.style.opacity = '1';
            overlay.style.visibility = 'visible';
            overlay.style.transition = 'none';
            overlay.style.pointerEvents = 'auto';
            overlay.innerHTML = `{safe_html}`;
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _cleanup_review_js_overlay():
    components.html(
        f"""
        <script>
        (function() {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            if (parentWindow.__triajeReviewTransitionController) {{
                parentWindow.__triajeReviewTransitionController.cancelled = true;
                parentWindow.__triajeReviewTransitionController = null;
            }}
            const overlay = doc.getElementById('{_REVIEW_JS_OVERLAY_ID}');
            if (overlay) overlay.remove();
            doc.querySelectorAll('[id^="{_REVIEW_READY_MARKER_PREFIX}"]').forEach(
                marker => marker.remove()
            );
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _reveal_review_behind_js_overlay(transition_token: str):
    marker_id = _review_ready_marker_id(transition_token)
    components.html(
        f"""
        <script>
        (function() {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const token = '{transition_token}';
            const overlay = doc.getElementById('{_REVIEW_JS_OVERLAY_ID}');
            if (!overlay || overlay.dataset.transitionToken !== token) return;

            if (parentWindow.__triajeReviewTransitionController) {{
                parentWindow.__triajeReviewTransitionController.cancelled = true;
            }}
            const controller = {{token, cancelled:false}};
            parentWindow.__triajeReviewTransitionController = controller;
            const startedAt = Date.now();
            const log = parentWindow.__triajeReviewTransitionLog ||= [];
            let previousSignature = '';
            let stableFrames = 0;
            let revealScheduled = false;

            const record = (event, extra={{}}) => {{
                log.push({{token, event, at:Date.now(), ...extra}});
                if (log.length > 200) log.splice(0, log.length - 200);
            }};
            const visible = element => {{
                if (!element) return false;
                const style = parentWindow.getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden'
                    && Number(style.opacity) > 0 && rect.width > 0 && rect.height > 0;
            }};
            const reviewIsStable = () => {{
                const marker = doc.getElementById('{marker_id}');
                if (!marker) {{ stableFrames = 0; return false; }}
                const identityHeader = doc.querySelector('.app-header');
                if (!visible(identityHeader)) {{ stableFrames = 0; return false; }}
                const buttons = Array.from(doc.querySelectorAll('button'));
                const hasConfirm = buttons.some(button =>
                    button.textContent.includes('Confirmar y calcular nivel') && visible(button)
                );
                const requiredLabels = ['Edad', 'Sexo', 'TA sistólica', 'SpO₂', 'Síntomas / motivo'];
                const bodyText = doc.body.innerText || '';
                const requiredFields = requiredLabels.every(label => bodyText.includes(label));
                const busy = Array.from(doc.querySelectorAll(
                    '[data-testid="stSkeleton"], [data-testid="stSpinner"], [data-testid="stStatusWidget"]'
                )).some(visible);
                if (!hasConfirm || !requiredFields || busy) {{ stableFrames = 0; return false; }}

                const main = doc.querySelector('[data-testid="stMain"]') || doc.scrollingElement;
                if (!main) {{ stableFrames = 0; return false; }}
                const signature = [
                    main.scrollHeight,
                    main.clientHeight,
                    buttons.length,
                    doc.querySelectorAll('[data-testid="stElementContainer"]').length,
                ].join(':');
                if (signature === previousSignature) stableFrames += 1;
                else {{ previousSignature = signature; stableFrames = 0; }}
                return stableFrames >= 5;
            }};
            const finish = reason => {{
                if (revealScheduled || controller.cancelled) return;
                if (overlay.dataset.transitionToken !== token) return;
                revealScheduled = true;
                const main = doc.querySelector('[data-testid="stMain"]') || doc.scrollingElement;
                const placeReviewAtTop = () => {{
                    if (main && typeof main.scrollTo === 'function') {{
                        main.scrollTo({{top:0, left:0, behavior:'auto'}});
                    }}
                    doc.documentElement.scrollTop = 0;
                    doc.body.scrollTop = 0;
                }};
                placeReviewAtTop();
                record('review-ready', {{reason, waitMs:Date.now() - startedAt}});
                const card = overlay.querySelector('.processing-card');
                if (card) {{
                    card.style.transition = 'opacity 180ms ease-in, transform 180ms ease-in';
                    card.style.opacity = '0';
                    card.style.transform = 'scale(0.985)';
                }}
                parentWindow.setTimeout(() => {{
                    if (controller.cancelled || overlay.dataset.transitionToken !== token) return;
                    record('card-hidden');
                    parentWindow.setTimeout(() => {{
                        if (controller.cancelled || overlay.dataset.transitionToken !== token) return;
                        overlay.style.pointerEvents = 'none';
                        overlay.style.transition = 'opacity 220ms ease-out';
                        overlay.style.opacity = '0';
                        record('overlay-fade-start');
                        parentWindow.setTimeout(() => {{
                            if (overlay.parentNode && overlay.dataset.transitionToken === token) overlay.remove();
                            parentWindow.requestAnimationFrame(() =>
                                parentWindow.requestAnimationFrame(placeReviewAtTop)
                            );
                            record('overlay-removed');
                            if (parentWindow.__triajeReviewTransitionController === controller) {{
                                parentWindow.__triajeReviewTransitionController = null;
                            }}
                        }}, 240);
                    }}, 100);
                }}, 180);
            }};
            const check = () => {{
                if (controller.cancelled) return;
                if (reviewIsStable()) return finish('stable');
                if (Date.now() - startedAt >= 8000) return finish('timeout');
                parentWindow.requestAnimationFrame(check);
            }};
            record('readiness-wait-start');
            parentWindow.requestAnimationFrame(check);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _push_js_overlay(inner_html: str, transition_token: str):
    """Crea (o actualiza si ya existe) un overlay a pantalla completa
    directamente en el DOM del documento padre, fuera del árbol de
    Streamlit. Bloquea el clic (pointer-events) mientras está activo."""
    safe_html = inner_html.replace("\\", "\\\\").replace("`", "\\`").replace("</script", "<\\/script")
    components.html(
        f"""
        <script>
        (function() {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            if (parentWindow.__triajeTransitionController) {{
                parentWindow.__triajeTransitionController.cancelled = true;
            }}
            parentWindow.__triajeTransitionController = null;
            doc.querySelectorAll('[id^="{_JS_READY_MARKER_PREFIX}"]').forEach(
                marker => marker.remove()
            );
            let overlay = doc.getElementById('{_JS_OVERLAY_ID}');
            if (overlay && overlay.dataset.transitionToken !== '{transition_token}') {{
                overlay.remove();
                overlay = null;
            }}
            if (!overlay) {{
                overlay = doc.createElement('div');
                overlay.id = '{_JS_OVERLAY_ID}';
                overlay.className = 'processing-screen';
                doc.body.appendChild(overlay);
            }}
            overlay.dataset.transitionToken = '{transition_token}';
            overlay.dataset.mountedAt = String(Date.now());
            overlay.style.zIndex = '2147483647';
            overlay.style.opacity = '1';
            overlay.style.visibility = 'visible';
            overlay.style.transition = 'none';
            overlay.style.pointerEvents = 'auto';
            overlay.innerHTML = `{safe_html}`;
            const log = parentWindow.__triajeTransitionLog ||= [];
            log.push({{token:'{transition_token}', event:'overlay-mounted', at:Date.now()}});
            if (log.length > 200) log.splice(0, log.length - 200);
        }})();
        </script>
        """,
        height=0, width=0,
    )


def _reveal_behind_js_overlay(
    transition_token: str,
    *,
    require_plot: bool,
    stability_margin_ms: int = 100,
):
    """Retira el overlay solo cuando el resultado actual está montado y estable."""
    marker_id = _ready_marker_id(transition_token)
    require_plot_js = "true" if require_plot else "false"
    components.html(
        f"""
        <script>
        (function() {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const overlay = doc.getElementById('{_JS_OVERLAY_ID}');
            if (!overlay || overlay.dataset.transitionToken !== '{transition_token}') return;

            if (parentWindow.__triajeTransitionController) {{
                parentWindow.__triajeTransitionController.cancelled = true;
            }}
            const controller = {{token:'{transition_token}', cancelled:false}};
            parentWindow.__triajeTransitionController = controller;
            const startedAt = Date.now();
            const log = parentWindow.__triajeTransitionLog ||= [];
            let previousSignature = '';
            let stableFrames = 0;
            let revealScheduled = false;

            const record = (event, extra={{}}) => {{
                log.push({{token:'{transition_token}', event, at:Date.now(), ...extra}});
                if (log.length > 200) log.splice(0, log.length - 200);
                const marker = doc.getElementById('{marker_id}');
                if (marker) {{
                    marker.dataset.lastTransitionEvent = event;
                    if (extra.reason) marker.dataset.readyReason = extra.reason;
                    if (extra.waitMs !== undefined) marker.dataset.readyWaitMs = String(extra.waitMs);
                }}
            }};
            const visible = element => {{
                if (!element) return false;
                const style = parentWindow.getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden'
                    && Number(style.opacity) > 0 && rect.width > 0 && rect.height > 0;
            }};
            const resultIsStable = () => {{
                const marker = doc.getElementById('{marker_id}');
                if (!marker) {{ stableFrames = 0; return false; }}

                const busy = Array.from(doc.querySelectorAll(
                    '[data-testid="stSkeleton"], [data-testid="stSpinner"], [data-testid="stStatusWidget"]'
                )).some(visible);
                if (busy) {{ stableFrames = 0; return false; }}

                const plots = Array.from(doc.querySelectorAll('.js-plotly-plot')).filter(visible);
                const plotReady = plots.some(plot =>
                    plot.querySelector('.main-svg') && plot.getBoundingClientRect().height > 80
                );
                if ({require_plot_js} && !plotReady) {{ stableFrames = 0; return false; }}

                const main = doc.querySelector('[data-testid="stMain"]') || doc.scrollingElement;
                if (!main) {{ stableFrames = 0; return false; }}
                const signature = [
                    main.scrollHeight,
                    main.clientHeight,
                    plots.length,
                    doc.querySelectorAll('[data-testid="stElementContainer"]').length,
                ].join(':');
                if (signature === previousSignature) stableFrames += 1;
                else {{ previousSignature = signature; stableFrames = 0; }}
                return stableFrames >= 5;
            }};
            const finish = reason => {{
                if (revealScheduled || controller.cancelled) return;
                if (overlay.dataset.transitionToken !== '{transition_token}') return;
                revealScheduled = true;
                record('result-ready', {{reason, waitMs:Date.now() - startedAt}});
                parentWindow.setTimeout(() => {{
                    if (controller.cancelled || overlay.dataset.transitionToken !== '{transition_token}') return;
                    const card = overlay.querySelector('.processing-card');
                    if (card) {{
                        card.style.transition = 'opacity 180ms ease-in, transform 180ms ease-in';
                        card.style.opacity = '0';
                        card.style.transform = 'scale(0.98)';
                    }}
                    record('pulse-hidden');
                    parentWindow.setTimeout(() => {{
                        if (controller.cancelled || overlay.dataset.transitionToken !== '{transition_token}') return;
                        overlay.style.pointerEvents = 'none';
                        overlay.style.transition = 'opacity 220ms ease-out';
                        overlay.style.opacity = '0';
                        record('overlay-fade-start');
                        parentWindow.setTimeout(() => {{
                            if (overlay.parentNode && overlay.dataset.transitionToken === '{transition_token}') {{
                                overlay.remove();
                            }}
                            record('overlay-removed');
                            if (parentWindow.__triajeTransitionController === controller) {{
                                parentWindow.__triajeTransitionController = null;
                            }}
                        }}, 240);
                    }}, 180);
                }}, {stability_margin_ms});
            }};
            const check = () => {{
                if (controller.cancelled) return;
                if (resultIsStable()) {{
                    finish('stable');
                    return;
                }}
                if (Date.now() - startedAt >= 8000) {{
                    finish('timeout');
                    return;
                }}
                parentWindow.requestAnimationFrame(check);
            }};
            record('readiness-wait-start');
            parentWindow.requestAnimationFrame(check);
        }})();
        </script>
        """,
        height=0, width=0,
    )


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
def _scroll_to_top():
    components.html(
        """
        <script>
        const resetScroll = () => {
            const doc = window.parent.document;
            if (doc.activeElement && typeof doc.activeElement.blur === 'function') {
                doc.activeElement.blur();
            }
            const main = doc.querySelector('[data-testid="stMain"]')
                || doc.querySelector('[data-testid="stAppViewContainer"] .main')
                || doc.querySelector('section.main')
                || doc.scrollingElement;
            if (main && typeof main.scrollTo === 'function') {
                main.scrollTo({top: 0, left: 0, behavior: 'auto'});
            }
            doc.documentElement.scrollTop = 0;
            doc.body.scrollTop = 0;
        };
        window.parent.requestAnimationFrame(() =>
            window.parent.requestAnimationFrame(resetScroll)
        );
        [0, 80, 180, 300].forEach(delay =>
            window.parent.setTimeout(resetScroll, delay)
        );
        </script>
        """,
        height=0, width=0,
    )


def _render_initial_welcome():
    mounted_at_ms = int(time.time() * 1000)
    st.markdown(
        f"""
        <div id="triaje-initial-welcome" class="initial-welcome" data-mounted-at="{mounted_at_ms}">
            <div class="initial-welcome-content">
                <div class="initial-welcome-brand">tr<span class="brand-ia">IA</span>je</div>
                <div class="initial-welcome-title">Bienvenido a trIAje</div>
                <div class="initial-welcome-copy">
                    Sistema de apoyo al triaje clínico desarrollado<br>
                    como Trabajo de Fin de Grado<br>
                    <span style="color:var(--text-muted); font-size:0.9em;">Carlos Rubio Martínez</span>
                </div>
                <div class="initial-welcome-status">
                    <span class="initial-welcome-dot" aria-hidden="true"></span>
                    <span>Menos tiempo clasificando, más tiempo cuidando.</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _control_initial_welcome():
    components.html(
        """
        <script>
        (function() {
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const overlay = doc.getElementById('triaje-initial-welcome');
            if (!overlay) return;

            const mountedAt = Number(overlay.dataset.mountedAt) || Date.now();
            const minimumVisibleMs = 2000;
            const maximumVisibleMs = 10000;
            const log = parentWindow.__triajeWelcomeLog ||= [];
            let stableFrames = 0;
            let previousSignature = '';
            let finished = false;

            const visible = element => {
                if (!element) return false;
                const style = parentWindow.getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden'
                    && Number(style.opacity) > 0 && rect.width > 0 && rect.height > 0;
            };
            const formIsReady = () => {
                const buttons = Array.from(doc.querySelectorAll('button'));
                const hasReviewButton = buttons.some(button =>
                    button.textContent.includes('Revisar episodio') && visible(button)
                );
                const hasAge = visible(doc.querySelector('input[placeholder="años"]'));
                const hasSex = ['Hombre', 'Mujer', 'Otro'].every(label =>
                    buttons.some(button => button.textContent.trim() === label && visible(button))
                );
                const busy = Array.from(doc.querySelectorAll(
                    '[data-testid="stSkeleton"], [data-testid="stSpinner"], [data-testid="stStatusWidget"]'
                )).some(visible);
                if (!hasReviewButton || !hasAge || !hasSex || busy) {
                    stableFrames = 0;
                    return false;
                }
                const main = doc.querySelector('[data-testid="stMain"]') || doc.scrollingElement;
                const signature = [
                    main ? main.scrollHeight : 0,
                    doc.querySelectorAll('[data-testid="stElementContainer"]').length,
                    buttons.length,
                ].join(':');
                if (signature === previousSignature) stableFrames += 1;
                else { previousSignature = signature; stableFrames = 0; }
                return stableFrames >= 4;
            };
            const finish = reason => {
                if (finished) return;
                finished = true;
                const elapsed = Date.now() - mountedAt;
                const wait = Math.max(0, minimumVisibleMs - elapsed);
                parentWindow.setTimeout(() => {
                    if (!overlay.isConnected) return;
                    log.push({event:'welcome-fade-start', reason, at:Date.now(), elapsed:Date.now() - mountedAt});
                    overlay.classList.add('is-leaving');
                    parentWindow.setTimeout(() => {
                        if (overlay.isConnected) overlay.remove();
                        log.push({event:'welcome-removed', reason, at:Date.now(), elapsed:Date.now() - mountedAt});
                        if (log.length > 40) log.splice(0, log.length - 40);
                    }, 360);
                }, wait);
            };
            const check = () => {
                if (finished || !overlay.isConnected) return;
                if (formIsReady()) return finish('form-ready');
                if (Date.now() - mountedAt >= maximumVisibleMs) return finish('timeout');
                parentWindow.requestAnimationFrame(check);
            };
            log.push({event:'welcome-controller-ready', at:Date.now(), elapsed:Date.now() - mountedAt});
            parentWindow.requestAnimationFrame(check);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _install_new_episode_click_scroll():
    components.html(
        """
        <script>
        (function() {
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            if (parentWindow.__triajeNewEpisodeClickHandler) {
                doc.removeEventListener('click', parentWindow.__triajeNewEpisodeClickHandler, true);
            }
            const handler = event => {
                const button = event.target && event.target.closest
                    ? event.target.closest('button') : null;
                if (!button || !button.textContent.toLowerCase().includes('nuevo episodio')) return;
                const main = doc.querySelector('[data-testid="stMain"]')
                    || doc.querySelector('[data-testid="stAppViewContainer"] .main')
                    || doc.scrollingElement;
                if (main && typeof main.scrollTo === 'function') {
                    main.scrollTo({top: 0, left: 0, behavior: 'smooth'});
                }
                parentWindow.__triajeNewEpisodeClickScrolledAt = Date.now();
                doc.removeEventListener('click', handler, true);
                parentWindow.__triajeNewEpisodeClickHandler = null;
            };
            parentWindow.__triajeNewEpisodeClickHandler = handler;
            doc.addEventListener('click', handler, true);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _scroll_new_episode_when_form_ready(scroll_token: str):
    components.html(
        f"""
        <script>
        (function() {{
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const token = '{scroll_token}';
            if (!token || parentWindow.__triajeConsumedNewEpisodeToken === token) return;
            if (parentWindow.__triajeNewEpisodeObserver) {{
                parentWindow.__triajeNewEpisodeObserver.disconnect();
            }}
            let finished = false;
            const finish = () => {{
                if (finished) return;
                const anchor = doc.getElementById('triaje-form-start');
                const reviewReady = Array.from(doc.querySelectorAll('button')).some(
                    button => button.textContent.includes('Revisar episodio')
                );
                if (!anchor || !reviewReady) return;
                finished = true;
                const main = doc.querySelector('[data-testid="stMain"]')
                    || doc.querySelector('[data-testid="stAppViewContainer"] .main')
                    || doc.scrollingElement;
                if (main && typeof main.scrollTo === 'function' && main.scrollTop > 8) {{
                    main.scrollTo({{top: 0, left: 0, behavior: 'smooth'}});
                }}
                parentWindow.__triajeConsumedNewEpisodeToken = token;
                if (parentWindow.__triajeNewEpisodeObserver) {{
                    parentWindow.__triajeNewEpisodeObserver.disconnect();
                    parentWindow.__triajeNewEpisodeObserver = null;
                }}
            }};
            const observer = new MutationObserver(finish);
            parentWindow.__triajeNewEpisodeObserver = observer;
            observer.observe(doc.body, {{childList: true, subtree: true}});
            parentWindow.requestAnimationFrame(finish);
            parentWindow.setTimeout(() => {{
                if (!finished && parentWindow.__triajeNewEpisodeObserver === observer) {{
                    observer.disconnect();
                    parentWindow.__triajeNewEpisodeObserver = null;
                }}
            }}, 2500);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _identity_header_html():
    return (
        '<div class="app-header">'
        '<div class="logo-wrap">'
        '<span class="logo-text">tr<span class="brand-ia">IA</span>je</span>'
        '</div>'
        '<div class="header-copy">'
        '<div class="header-title">Sistema de apoyo al triaje clínico</div>'
        '<div class="header-meta">Trabajo de Fin de Grado'
        '<span class="header-meta-separator">·</span>Carlos Rubio Martínez</div>'
        '</div></div>'
    )


def _render_identity_header():
    st.markdown(_identity_header_html(), unsafe_allow_html=True)


def _render_header():
    if st.session_state.fase == 3:
        col_identity, col_action = st.columns([5, 1.35], gap="medium", vertical_alignment="center")
        with col_identity:
            _render_identity_header()
        with col_action:
            with st.container(key="top_new_episode_action"):
                if st.button(
                    "Nuevo episodio",
                    type="secondary",
                    key="nuevo_caso_top",
                    icon=":material/refresh:",
                ):
                    _reset_estado()
    else:
        st.markdown(
            _identity_header_html() + '<div class="header-divider"></div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)


def _render_footer():
    st.markdown(
        '<div class="app-footer">'
        '<div class="app-footer-brand">tr<span class="brand-ia">IA</span>je</div>'
        '<div class="app-footer-title">Sistema de apoyo al triaje clínico</div>'
        '<div class="app-footer-meta">Trabajo de Fin de Grado · Carlos Rubio Martínez</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _reset_estado():
    _cleanup_js_transition_artifacts()
    keys_to_clear = [
        "fase", "ultimo_vector", "ultima_narrativa",
        "ultimo_resultado_ml", "ultima_explicacion_shap",
        "ultimo_umbral_alerta_a1",
        "datos_formulario", "vector_revisado_pendiente",
        "review_transition_token", "result_reveal_pending",
        "result_transition_token",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.new_episode_scroll_token = uuid.uuid4().hex
    st.session_state.scroll_to_top_pending = True
    st.rerun()


# ─────────────────────────────────────────────────────────────
# FASE 1 — FORMULARIO
# ─────────────────────────────────────────────────────────────
def _render_fase_1():
    pending_scroll = st.session_state.get("scroll_to_top_pending", False)
    scroll_token = st.session_state.get("new_episode_scroll_token", "") if pending_scroll else ""
    st.markdown('<div id="triaje-form-start" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sec-label">Registro de episodio</span>', unsafe_allow_html=True)

    with st.form("triage_intake_form_narrativo", border=False):
        form_col, side_col = st.columns([2, 1], gap="large")

        with form_col:
            st.markdown(
                '<div class="form-section-title"><span class="required-dot"></span> Datos mínimos</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="minimum_data_controls"):
                edad_col, sexo_col, llegada_col = st.columns(3, gap="small")
                with edad_col:
                    edad_raw = st.text_input("Edad *", placeholder="años")
                with sexo_col:
                    sexo = st.segmented_control(
                        "Sexo *", ["M", "F", "Otro"],
                        format_func=lambda value: {"M": "Hombre", "F": "Mujer", "Otro": "Otro"}[value],
                        width="stretch",
                    )
                with llegada_col:
                    metodo_llegada = st.selectbox(
                        "Método de llegada",
                        ["desconocido", "autonomo", "ambulancia", "helicoptero", "otro"],
                        index=0,
                    )

            st.markdown('<div class="form-spacer"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="form-section-title"><span class="required-dot"></span> Relato clínico guiado</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="intake_narrative_row"):
                motivo_col, duracion_col = st.columns([3, 2], gap="small")
                with motivo_col:
                    motivo_consulta = st.text_input("Motivo principal o síntoma guía *", placeholder="Dolor abdominal desde la tarde")
                with duracion_col:
                    duracion_sintomas = st.text_input("Duración / evolución", placeholder="Desde hace unas horas")
            relato_clinico = st.text_area(
                "Relato clínico breve",
                placeholder="Paciente consciente y orientado. Refiere dolor abdominal progresivo. Niega dolor torácico.",
                height=150,
            )

            st.markdown('<div class="form-spacer"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="form-section-title"><span class="required-dot"></span> Constantes vitales y dolor</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="intake_vitals_primary"):
                ta_s_col, ta_d_col, fc_col = st.columns(3, gap="small")
                with ta_s_col:
                    presion_sistolica_raw = st.text_input("TA sistólica", placeholder="mmHg")
                with ta_d_col:
                    presion_diastolica_raw = st.text_input("TA diastólica", placeholder="mmHg")
                with fc_col:
                    frecuencia_cardiaca_raw = st.text_input("FC", placeholder="lpm")

            with st.container(key="intake_vitals_secondary"):
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
            st.caption("Escala: 0 = sin dolor · 10 = máximo dolor referido.")

            st.markdown('<div class="form-spacer"></div>', unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Contexto clínico</div>', unsafe_allow_html=True)
            antecedentes = st.text_area("Antecedentes relevantes", placeholder="Hipertensión arterial", height=82)
            medicacion = st.text_area("Medicación habitual", placeholder="Enalapril", height=82)
            observaciones = st.text_area("Observaciones de triaje", placeholder="Acude acompañado. Buen estado general.", height=82)

        with side_col:
            st.markdown('<div class="form-section-title">Discriminadores de prioridad</div>', unsafe_allow_html=True)
            signos_alarma = st.pills(
                "Marcar si está presente", SIGNOS_ALARMA,
                selection_mode="multi", default=[], width="stretch",
            )

        edad, edad_error = _parse_int_input(edad_raw, "Edad", 0, 120)
        presion_sistolica, ta_s_error = _parse_int_input(presion_sistolica_raw, "TA sistólica", 40, 300)
        presion_diastolica, ta_d_error = _parse_int_input(presion_diastolica_raw, "TA diastólica", 20, 200)
        frecuencia_cardiaca, fc_error = _parse_int_input(frecuencia_cardiaca_raw, "FC", 20, 300)
        frecuencia_respiratoria, fr_error = _parse_int_input(frecuencia_respiratoria_raw, "FR", 4, 60)
        saturacion_oxigeno, spo2_error = _parse_float_input(saturacion_oxigeno_raw, "SpO₂", 50, 100)
        temperatura, temp_error = _parse_float_input(temperatura_raw, "Temperatura", 30, 45)

        avisos_numericos = [a for a in (edad_error, ta_s_error, ta_d_error, fc_error, fr_error, spo2_error, temp_error) if a]

        datos_formulario = TriageFormData(
            edad=edad, sexo=sexo or "", metodo_llegada=metodo_llegada,
            motivo_consulta=motivo_consulta, sintomas_frecuentes=signos_alarma or [],
            sintomas_adicionales=relato_clinico, signos_alarma=signos_alarma or [],
            duracion_sintomas=duracion_sintomas,
            presion_sistolica=presion_sistolica, presion_diastolica=presion_diastolica,
            frecuencia_cardiaca=frecuencia_cardiaca, frecuencia_respiratoria=frecuencia_respiratoria,
            saturacion_oxigeno=saturacion_oxigeno, temperatura=temperatura,
            nivel_dolor=None if dolor_valor == "No registrado" else int(dolor_valor),
            antecedentes=antecedentes, medicacion=medicacion, observaciones=observaciones,
        )

        tiene_constante = any([
            presion_sistolica is not None, frecuencia_cardiaca is not None,
            frecuencia_respiratoria is not None, saturacion_oxigeno is not None,
            temperatura is not None,
        ])

        _, submit_col, _ = st.columns([1, 1, 1])
        with submit_col:
            analizar = st.form_submit_button("Revisar episodio", type="primary", icon=":material/monitor_heart:")

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
                st.markdown(f'<div class="triage-warning">{aviso}</div>', unsafe_allow_html=True)

            narrativa = generar_narrativa_triaje_texto_libre(datos_formulario)
            st.session_state.datos_formulario = datos_formulario
            st.session_state.ultima_narrativa = narrativa.strip()
            st.session_state.ultimo_vector = None
            st.session_state.ultimo_resultado_ml = None
            st.session_state.review_transition_token = uuid.uuid4().hex
            _push_review_js_overlay(
                _render_review_preparation_card_inner(0, PROGRESO_PROCESAMIENTO[0]),
                st.session_state.review_transition_token,
            )
            st.session_state.fase = 2
            st.rerun()

    st.markdown(
        """<p class="text-note" style="margin-top:0.9rem; text-align:center;">El formulario guía la recogida mínima y prepara una revisión estructurada. No sustituye el criterio profesional.</p>""",
        unsafe_allow_html=True,
    )
    if scroll_token:
        _scroll_new_episode_when_form_ready(scroll_token)
        st.session_state.scroll_to_top_pending = False


# ─────────────────────────────────────────────────────────────
# FASE 2 — PROCESAMIENTO + REVISIÓN
# ─────────────────────────────────────────────────────────────
PASOS_PROCESAMIENTO = [
    {"label": "Organizando relato clínico", "help": "Ordenando el motivo de consulta y el relato registrado.", "area": "Relato"},
    {"label": "Identificando datos relevantes", "help": "Separando síntomas, duración y contexto clínico.", "area": "Relato"},
    {"label": "Revisando constantes vitales", "help": "Comprobando constantes, temperatura y dolor.", "area": "Constantes"},
    {"label": "Comprobando antecedentes y medicación", "help": "Relacionando antecedentes, tratamiento habitual y observaciones.", "area": "Contexto"},
    {"label": "Buscando señales de prioridad", "help": "Revisando hallazgos que pueden requerir atención preferente.", "area": "Prioridad"},
    {"label": "Preparando revisión editable", "help": "Preparando los datos para que pueda corregirlos o confirmarlos.", "area": "Revisión"},
    {"label": "Revisión lista", "help": "Los datos están preparados para su comprobación.", "area": "Revisión"},
]
PROGRESO_PROCESAMIENTO = [0.10, 0.22, 0.36, 0.50, 0.64, 0.92, 1.0]
AREAS_PROCESAMIENTO = ["Relato", "Constantes", "Contexto", "Prioridad", "Revisión"]

_ICON_DONE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
_ICON_ACTIVE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0891B2" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/></svg>'
_ICON_PENDING = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6B7F93" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/></svg>'


def _render_processing_steps_html(current_step):
    steps_html = ""
    for i, paso in enumerate(PASOS_PROCESAMIENTO):
        if i < current_step:
            css_class, icon = "done", _ICON_DONE
        elif i == current_step:
            css_class, icon = "active", _ICON_ACTIVE
        else:
            css_class, icon = "pending", _ICON_PENDING
        steps_html += (
            f'<div class="processing-step {css_class}">'
            f'<span class="step-icon">{icon}</span>'
            f'<span>{html.escape(paso["label"])}</span></div>'
        )
    return steps_html


def _render_processing_areas_html(current_step):
    active_area = PASOS_PROCESAMIENTO[current_step]["area"]
    completed_areas = {p["area"] for p in PASOS_PROCESAMIENTO[:current_step] if p["area"] != active_area}
    return "".join(
        f'<div class="processing-area {"done" if area in completed_areas else "active" if area == active_area else ""}">{html.escape(area)}</div>'
        for area in AREAS_PROCESAMIENTO
    )


PROGRESO_RESULTADO = [0.12, 0.34, 0.58, 0.82, 1.0]


def _render_result_preparation_screen(step_index, progress_pct, *, leaving=False):
    progress_value = max(0, min(100, round(progress_pct * 100)))
    screen_class = "processing-screen is-leaving" if leaving else "processing-screen"
    return (
        f'<div class="{screen_class}"><div class="processing-card result-compact">'
        f'<div class="result-processing-mark" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{progress_value}">'
        '<svg viewBox="0 0 120 80" aria-hidden="true">'
        '<defs><clipPath id="result-pulse-clip" class="result-pulse-clip">'
        f'<rect x="0" y="0" height="80" style="width:{progress_value}%"/>'
        '</clipPath></defs>'
        '<path class="result-pulse-base" d="M4 42h20l9-19 15 38 15-53 16 45 10-23 9 12h18"/>'
        '<path class="result-pulse-progress" clip-path="url(#result-pulse-clip)" d="M4 42h20l9-19 15 38 15-53 16 45 10-23 9 12h18"/>'
        '</svg></div>'
        '<div class="processing-title">Calculando prioridad y explicación</div>'
        '</div></div>'
    )


def _preparar_explicacion_resultado(predictor, result, clase_predicha):
    try:
        import pandas as pd
        import plotly.graph_objects as go
        from triaje_ia.ml.explicabilidad import explicar_prediccion

        explainer = _cargar_explainer(predictor._clf)
        resultado_shap = explicar_prediccion(explainer, result.X, clase_predicha=clase_predicha)
        valores_shap = [float(v) for v in resultado_shap.shap_values]
        factores = select_balanced_shap_factors(resultado_shap.feature_names, valores_shap, per_direction=3)

        df_shap = pd.DataFrame({"Feature": [clean_feature_name(n) for n in resultado_shap.feature_names], "SHAP": valores_shap})
        df_shap["Abs_SHAP"] = df_shap["SHAP"].abs()
        df_shap = df_shap.sort_values(by="Abs_SHAP", ascending=True).tail(10)
        df_shap["Feature"] = [label + (" " * idx) for idx, label in enumerate(df_shap["Feature"])]
        shap_min = min(0.0, float(df_shap["SHAP"].min()))
        shap_max = max(0.0, float(df_shap["SHAP"].max()))
        shap_span = max(shap_max - shap_min, 0.1)
        x_padding = max(shap_span * 0.20, 0.08)
        x_range = [shap_min - x_padding, shap_max + x_padding]

        fill_colores = ["rgba(8, 145, 178, 0.82)" if v > 0 else "rgba(100, 116, 139, 0.78)" for v in df_shap["SHAP"]]
        line_colores = ["#0E7490" if v > 0 else "#475569" for v in df_shap["SHAP"]]

        fig = go.Figure(go.Bar(
            x=df_shap["SHAP"], y=df_shap["Feature"], orientation="h",
            marker=dict(color=fill_colores, line=dict(color=line_colores, width=1.5)),
            text=[f"{v:+.3f}" for v in df_shap["SHAP"]], textposition="outside", cliponaxis=False,
            textfont=dict(family="Figtree, Noto Sans, sans-serif", size=15, color="#1F2D3D", weight="bold"),
            hovertemplate="<b>%{y}</b><br>Impacto en el resultado: %{x:+.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Distribución de contribuciones", x=0, font=dict(size=21, color="#0F172A", weight="bold")),
            margin=dict(l=20, r=110, t=70, b=75), height=465, bargap=0.22,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Figtree, Noto Sans, sans-serif", color="#253242", size=15),
            xaxis=dict(range=x_range, showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.5, showticklabels=True, tickfont=dict(color="#64748B", size=14), automargin=True, title=dict(text="Impacto local en la estimación", font=dict(size=16))),
            yaxis=dict(showgrid=False, tickfont=dict(color="#0F172A", size=15, weight="bold"), automargin=True),
            showlegend=False, dragmode=False,
        )
        return {"ok": True, "fig": fig, "factores": factores, "shap_values": resultado_shap.shap_values, "error": None}
    except Exception as e:
        return {"ok": False, "fig": None, "factores": [], "shap_values": [], "error": str(e)}


def _vector_review_text(value):
    return "" if value is None else str(value)


def _render_fase_2(transition_slot):
    narrativa = st.session_state.ultima_narrativa
    if not narrativa:
        st.session_state.fase = 1
        st.rerun()
        return

    if st.session_state.ultimo_vector is None:
        _render_header()
        transition_token = st.session_state.get("review_transition_token", "")
        if not transition_token:
            transition_token = uuid.uuid4().hex
            st.session_state.review_transition_token = transition_token
        last_display_state = {"step": None, "progress": None}

        def _update_display(step_index, progress_pct, elapsed_s=0.0, *, force=False):
            previous_progress = last_display_state["progress"]
            same_step = last_display_state["step"] == step_index
            small_progress_change = (
                previous_progress is not None
                and abs(progress_pct - previous_progress) < 0.02
            )
            if not force and same_step and small_progress_change:
                return
            _push_review_js_overlay(
                _render_review_preparation_card_inner(step_index, progress_pct),
                transition_token,
            )
            last_display_state["step"] = step_index
            last_display_state["progress"] = progress_pct

        try:
            start_time = time.monotonic()
            _update_display(0, PROGRESO_PROCESAMIENTO[0], 0.0, force=True)
            extraer_fn = _cargar_extractor()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(extraer_fn, narrativa, modelo=MODELO_LLM_FIJO)
                while not future.done():
                    elapsed = time.monotonic() - start_time
                    if elapsed < 1.2:
                        step_index, progress_pct = 0, PROGRESO_PROCESAMIENTO[0]
                    elif elapsed < 2.4:
                        step_index, progress_pct = 1, PROGRESO_PROCESAMIENTO[1]
                    elif elapsed < 3.6:
                        step_index, progress_pct = 2, PROGRESO_PROCESAMIENTO[2]
                    elif elapsed < 5.0:
                        step_index, progress_pct = 3, PROGRESO_PROCESAMIENTO[3]
                    elif elapsed < 8.0:
                        step_index, progress_pct = 4, PROGRESO_PROCESAMIENTO[4]
                    else:
                        step_index = 5
                        progress_pct = min(0.94, 0.78 + ((elapsed - 8.0) * 0.01))
                    _update_display(step_index, progress_pct, elapsed)
                    time.sleep(0.25)
                vector = future.result()
            elapsed = time.monotonic() - start_time
            _update_display(5, PROGRESO_PROCESAMIENTO[5], elapsed, force=True)
            st.session_state.ultimo_vector = vector
            st.session_state.ultimo_resultado_ml = None
            st.session_state.ultima_explicacion_shap = None
            st.session_state.vector_revisado_pendiente = None
            _update_display(6, PROGRESO_PROCESAMIENTO[6], elapsed, force=True)
            st.rerun()
        except ConnectionError:
            _cleanup_review_js_overlay()
            st.error("No se ha podido conectar con el procesamiento local. Compruebe que está iniciado y vuelva a intentarlo.")
            if st.button("Volver al registro", type="secondary"):
                st.session_state.review_transition_token = ""
                st.session_state.fase = 1
                st.rerun()
        except Exception as e:
            _cleanup_review_js_overlay()
            st.error(f"No se ha podido preparar la revisión. Detalle técnico: {e}")
            if st.button("Volver al registro", type="secondary"):
                st.session_state.review_transition_token = ""
                st.session_state.fase = 1
                st.rerun()
        return

    # Vector ya existe: mostrar formulario de revisión
    vector = st.session_state.ultimo_vector

    # El formulario vive en su propio placeholder para poder vaciarlo de forma
    # explícita e inmediata en el mismo ciclo en que se pulsa "Confirmar",
    # ANTES de cambiar de fase. Así no queda ningún widget residual del
    # formulario esperando a que un rerun futuro lo reconcilie: eso es lo que
    # provocaba el microcorte visual al llegar a los resultados finales.
    form_slot = st.empty()
    with form_slot.container():
        _render_header()
        st.markdown('<span class="sec-label">Revisión del episodio</span>', unsafe_allow_html=True)

        alertas = validar_vector_clinico(vector, narrativa)
        for alerta in [a for a in alertas if a.nivel in (NivelAlerta.WARNING, NivelAlerta.ERROR)]:
            css_class = "warning-validation" if alerta.nivel == NivelAlerta.WARNING else "alert-clinical"
            st.markdown(
                f'<div class="{css_class}"><strong>[{html.escape(alerta.campo)}]</strong> {html.escape(alerta.mensaje)}</div>',
                unsafe_allow_html=True,
            )

        with st.form("review_vector_form", border=False):
            col_left, col_right = st.columns(2, gap="large")

            with col_left:
                st.markdown('<div class="form-section-title"><span class="required-dot"></span> Datos clínicos revisables</div>', unsafe_allow_html=True)
                with st.container(key="review_identity_controls"):
                    edad_col, sexo_col, llegada_col = st.columns(3, gap="small")
                    with edad_col:
                        edad_raw = st.text_input("Edad", value=_vector_review_text(vector.edad))
                    with sexo_col:
                        sexo_opts = ["M", "F", "Otro"]
                        sexo = st.selectbox("Sexo", sexo_opts, index=sexo_opts.index(vector.sexo) if vector.sexo in sexo_opts else 2, format_func=lambda value: {"M": "Hombre", "F": "Mujer", "Otro": "Otro"}[value])
                    with llegada_col:
                        llegada_opts = ["desconocido", "autonomo", "ambulancia", "helicoptero", "otro"]
                        metodo_llegada = st.selectbox("Método de llegada", llegada_opts, index=llegada_opts.index(vector.metodo_llegada) if vector.metodo_llegada in llegada_opts else 0)

                sintomas_presentes = st.text_area("Síntomas / motivo", value=join_review_terms_display(vector.sintomas_presentes), height=92)
                duracion_sintomas = st.text_input("Duración o evolución", value=vector.duracion_sintomas or "")
                patologias_previas = st.text_area("Antecedentes relevantes", value=join_review_terms_display(vector.patologias_previas), height=82)
                medicacion_habitual = st.text_area("Medicación habitual", value=join_review_terms_display(vector.medicacion_habitual), height=82)

            with col_right:
                st.markdown('<div class="form-section-title">Constantes y dolor</div>', unsafe_allow_html=True)
                ta_s_col, ta_d_col = st.columns(2, gap="small")
                with ta_s_col:
                    presion_sistolica_raw = st.text_input("TA sistólica", value=_vector_review_text(vector.presion_sistolica))
                with ta_d_col:
                    presion_diastolica_raw = st.text_input("TA diastólica", value=_vector_review_text(vector.presion_diastolica))
                fc_col, fr_col = st.columns(2, gap="small")
                with fc_col:
                    frecuencia_cardiaca_raw = st.text_input("FC", value=_vector_review_text(vector.frecuencia_cardiaca))
                with fr_col:
                    frecuencia_respiratoria_raw = st.text_input("FR", value=_vector_review_text(vector.frecuencia_respiratoria))
                spo2_col, temp_col = st.columns(2, gap="small")
                with spo2_col:
                    saturacion_oxigeno_raw = st.text_input("SpO₂", value=_vector_review_text(vector.saturacion_oxigeno))
                with temp_col:
                    temperatura_raw = st.text_input("Temperatura", value=_vector_review_text(vector.temperatura))
                dolor_valor = st.select_slider("Dolor EVA/NRS", options=["No registrado", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], value="No registrado" if vector.nivel_dolor is None else int(vector.nivel_dolor))
                st.caption("Escala: 0 = sin dolor · 10 = máximo dolor referido.")

                with st.expander("Ver narrativa estructurada", expanded=False):
                    st.markdown(f'<p class="review-narrative">{html.escape(narrativa)}</p>', unsafe_allow_html=True)

            with st.container(key="review_action_layout"):
                back_col, confirm_col = st.columns([1, 1], gap="medium")
                with back_col:
                    back = st.form_submit_button("Volver al registro", type="secondary", icon=":material/arrow_back:")
                with confirm_col:
                    confirm = st.form_submit_button("Confirmar y calcular nivel", type="primary", icon=":material/check_circle:")

    review_transition_token = st.session_state.get("review_transition_token", "")
    if review_transition_token:
        st.markdown(
            f'<div id="{_review_ready_marker_id(review_transition_token)}" style="display:none"></div>',
            unsafe_allow_html=True,
        )
        _reveal_review_behind_js_overlay(review_transition_token)

    if back:
        form_slot.empty()
        _cleanup_review_js_overlay()
        st.session_state.review_transition_token = ""
        st.session_state.ultimo_vector = None
        st.session_state.ultimo_resultado_ml = None
        st.session_state.ultima_explicacion_shap = None
        st.session_state.vector_revisado_pendiente = None
        st.session_state.fase = 1
        st.rerun()

    if confirm:
        edad, edad_error = _parse_int_input(edad_raw, "Edad", 0, 120)
        presion_sistolica, ta_s_error = _parse_int_input(presion_sistolica_raw, "TA sistólica", 40, 300)
        presion_diastolica, ta_d_error = _parse_int_input(presion_diastolica_raw, "TA diastólica", 20, 200)
        frecuencia_cardiaca, fc_error = _parse_int_input(frecuencia_cardiaca_raw, "FC", 20, 300)
        frecuencia_respiratoria, fr_error = _parse_int_input(frecuencia_respiratoria_raw, "FR", 4, 60)
        saturacion_oxigeno, spo2_error = _parse_float_input(saturacion_oxigeno_raw, "SpO₂", 50, 100)
        temperatura, temp_error = _parse_float_input(temperatura_raw, "Temperatura", 30, 45)
        errores = [e for e in (edad_error, ta_s_error, ta_d_error, fc_error, fr_error, spo2_error, temp_error) if e]
        if edad is None:
            errores.append("Edad: campo obligatorio.")

        if errores:
            for err in errores:
                st.error(err)
            return

        try:
            vector_revisado = construir_vector_desde_revision(
                edad=edad, sexo=sexo, sintomas_presentes=sintomas_presentes,
                patologias_previas=patologias_previas, medicacion_habitual=medicacion_habitual,
                presion_sistolica=presion_sistolica, presion_diastolica=presion_diastolica,
                frecuencia_cardiaca=frecuencia_cardiaca, frecuencia_respiratoria=frecuencia_respiratoria,
                saturacion_oxigeno=saturacion_oxigeno, temperatura=temperatura,
                nivel_dolor=None if dolor_valor == "No registrado" else int(dolor_valor),
                duracion_sintomas=duracion_sintomas, metodo_llegada=metodo_llegada,
            )
        except Exception as e:
            st.error(f"No se pudo validar la revisión del episodio: {e}")
            return

        # Vaciar el placeholder del formulario AQUÍ, en el mismo rerun del
        # clic de "Confirmar" y antes de tocar la fase. Esto le da a
        # Streamlit un ciclo completo (toda la fase 25, con su propio
        # tiempo de cálculo) para reconciliar la eliminación del formulario
        # con calma, mucho antes de que el overlay del pulso empiece a
        # desvanecerse sobre los resultados finales.
        form_slot.empty()

        _cleanup_js_transition_artifacts()
        st.session_state.review_transition_token = ""
        st.session_state.ultimo_vector = vector_revisado
        st.session_state.vector_revisado_pendiente = vector_revisado
        st.session_state.ultimo_resultado_ml = None
        st.session_state.ultima_explicacion_shap = None
        st.session_state.result_transition_token = uuid.uuid4().hex
        st.session_state.fase = 25
        st.rerun()

def _render_fase_resultado_preparacion(transition_slot):
    _scroll_to_top()

    vector_revisado = st.session_state.vector_revisado_pendiente
    narrativa = st.session_state.ultima_narrativa
    transition_token = st.session_state.get("result_transition_token")
    if not transition_token:
        transition_token = uuid.uuid4().hex
        st.session_state.result_transition_token = transition_token
    if vector_revisado is None or not narrativa:
        st.session_state.fase = 1
        st.rerun()
        return

    container = transition_slot

    def _update_display(step_index):
        container.markdown(_render_result_preparation_screen(step_index, PROGRESO_RESULTADO[step_index]), unsafe_allow_html=True)

    total_start = time.monotonic()

    try:
        _update_display(0)

        _update_display(1)
        predictor = _cargar_predictor()
        result = predictor.predict(vector_revisado, narrativa)
        st.session_state.ultimo_umbral_alerta_a1 = getattr(predictor, "_warning_threshold_a1", 0.40)

        _update_display(2)

        _update_display(3)
        explicacion = _preparar_explicacion_resultado(predictor, result, result.clase_predicha)

        _update_display(4)
        completed_state_start = time.monotonic()

        visual_remaining = max(
            0.80 - (time.monotonic() - total_start),
            0.22 - (time.monotonic() - completed_state_start),
        )
        if visual_remaining > 0:
            time.sleep(visual_remaining)

        # Fijar el overlay JS (fuera del árbol de Streamlit) mostrando
        # exactamente lo mismo que ya se ve ahora mismo: no hay cambio
        # visual en este instante, solo pasa a estar bajo control de JS
        # puro. La pequeña pausa siguiente le da tiempo al navegador a
        # insertarlo de verdad en el DOM antes de que arranque el rerun.
        _push_js_overlay(
            _render_result_preparation_card_inner(4, PROGRESO_RESULTADO[4]),
            transition_token,
        )
        time.sleep(0.15)

        st.session_state.ultimo_vector = vector_revisado
        st.session_state.ultimo_resultado_ml = result
        st.session_state.ultima_explicacion_shap = explicacion
        st.session_state.vector_revisado_pendiente = None
        st.session_state.result_reveal_pending = True
        st.session_state.fase = 3
        st.rerun()
    except Exception as e:
        container.empty()
        st.error(f"Error preparando el resultado clínico: {e}")
        if st.button("Volver a la revisión", type="secondary"):
            st.session_state.vector_revisado_pendiente = None
            st.session_state.fase = 2
            st.rerun()


def _render_fase_3(transition_slot):
    _scroll_to_top()

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
    col_vector, col_resultado = st.columns([1, 1], gap="large")

    # ─ Columna izquierda ──
    with col_vector:
        st.markdown('<span class="sec-label">Resumen del episodio</span>', unsafe_allow_html=True)

        edad_v = getattr(vector, "edad", None)
        sexo_v = getattr(vector, "sexo", None)
        paciente_parts = []
        if edad_v is not None:
            paciente_parts.append(f"{edad_v} años")
        if sexo_v:
            paciente_parts.append({"M": "Hombre", "F": "Mujer", "Otro": "Otro"}.get(str(sexo_v), str(sexo_v)))
        paciente_str = " · ".join(paciente_parts) if paciente_parts else "No registrado"

        st.markdown(
            '<div class="vector-summary-card">'
            '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">'
            '<span class="vector-section-title" style="margin-bottom:0;">Paciente</span>'
            '<span style="display:inline-flex; align-items:center; gap:0.35rem; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px; padding:0.15rem 0.55rem;">'
            '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
            '<span style="font-size:0.8125rem; color:#166534; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">Revisado</span></span></div>'
            f'<div class="vector-value" style="font-weight:600;">{html.escape(paciente_str)}</div></div>',
            unsafe_allow_html=True,
        )

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
            v_col = "#0F172A" if valor is not None else "var(--text-muted)"
            items_html += f'<div class="vector-vital-item"><div class="vector-vital-label">{nombre}</div><div class="vector-vital-value" style="color:{v_col};">{v_str}</div></div>'

        st.markdown(
            '<div class="vector-summary-card" style="animation-delay:70ms;">'
            '<div class="vector-section-title">Constantes vitales</div>'
            f'<div class="vector-vital-grid">{items_html}</div></div>',
            unsafe_allow_html=True,
        )

        sintomas = display_clinical_terms(getattr(vector, "sintomas_presentes", []) or [])
        patologias = display_clinical_terms(getattr(vector, "patologias_previas", []) or [])
        medicacion_v = display_clinical_terms(getattr(vector, "medicacion_habitual", []) or [])

        def _render_tag(text, bg, tc, bc):
            return f'<span style="display:inline-flex; background:{bg}; color:{tc}; border:1px solid {bc}; font-size:0.8125rem; font-weight:500; padding:3px 10px; border-radius:16px; margin:0 4px 5px 0; line-height:1.4;">{html.escape(text)}</span>'

        secciones_tags = [
            ("Síntomas presentes", sintomas, "#EFF6FF", "#1D4ED8", "#93C5FD"),
            ("Antecedentes", patologias, "#FFF7ED", "#C2410C", "#FDBA74"),
            ("Medicación", medicacion_v, "#F0FDF4", "#15803D", "#86EFAC"),
        ]

        any_tags = False
        for idx, (titulo, items, bg, tc, bc) in enumerate(secciones_tags):
            if items:
                any_tags = True
                tags_joined = "".join(_render_tag(i, bg, tc, bc) for i in items)
                st.markdown(
                    f'<div class="vector-summary-card" style="animation-delay:{140 + idx * 70}ms;">'
                    f'<div class="vector-section-title">{titulo}</div>'
                    f'<div style="display:flex; flex-wrap:wrap;">{tags_joined}</div></div>',
                    unsafe_allow_html=True,
                )

        if not any_tags:
            st.markdown(
                '<div class="vector-summary-card" style="animation-delay:140ms;">'
                '<div class="vector-section-title">Datos clínicos</div>'
                '<p class="empty-clinical-state">No se detectaron síntomas, antecedentes ni medicación.</p></div>',
                unsafe_allow_html=True,
            )

    # ── Columna derecha ─
    with col_resultado:
        st.markdown('<span class="sec-label">Nivel de prioridad sugerido</span>', unsafe_allow_html=True)

        if alerta_a1_activada:
            st.markdown(f"""
            <div class="alert-clinical" style="display:flex; align-items:flex-start; gap:0.6rem;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; margin-top:2px;">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <div><strong>Alerta de seguridad clínica:</strong> P(acuity=1) = {probas_fila[0]:.1%} supera el umbral de aviso ({umbral_alerta_a1:.0%}). Revisar posible criticidad A1 antes de confirmar el nivel sugerido.</div>
            </div>""", unsafe_allow_html=True)

        alertas = validar_vector_clinico(vector, narrativa)
        alertas_warn = [a for a in alertas if a.nivel in (NivelAlerta.WARNING, NivelAlerta.ERROR)]
        for alerta in alertas_warn:
            if alerta.nivel == NivelAlerta.ERROR:
                icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; transform:translateY(2px); margin-right:4px;"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'
            else:
                icon = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; transform:translateY(2px); margin-right:4px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
            st.markdown(f'<div class="warning-validation" style="display:flex; align-items:flex-start;">{icon}<span><strong>[{alerta.campo}]</strong> {alerta.mensaje}</span></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hero-esi" style="background:{esi['bg']}; border:2px solid {esi['border']};">
            <div style="display:flex; align-items:center; gap:1.1rem;">
                <div style="width:46px; height:46px; border-radius:50%; background:#FFF; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,0.08);">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{esi['color']}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                </div>
                <div><div class="hero-esi-label" style="color:{esi['color']};">Nivel sugerido</div><div class="hero-esi-title" style="color:{esi['color']};">{esi['label']}</div></div>
            </div>
            <div class="hero-esi-confidence"><div class="label">Confianza</div><div class="value">{confianza:.1%}</div></div>
        </div>""", unsafe_allow_html=True)

        clases_nombre = ["ESI 1", "ESI 2", "ESI 3", "ESI 4", "ESI 5"]
        colores_barra = ["#DC2626", "#EA580C", "#CA8A04", "#16A34A", "#2563EB"]

        if hasattr(probas_fila, "__len__") and len(probas_fila) == 5:
            barras_html = ""
            for i in range(5):
                try:
                    p = float(probas_fila[i])
                except (ValueError, TypeError):
                    p = 0.0
                width_pct = p * 100
                is_max = (i == clase_predicha - 1)
                text_color = "#0F172A" if is_max else "#486581"
                font_weight = "700" if is_max else "500"
                bar_opacity = "1" if is_max else "0.55"
                icon_opacity = "1" if is_max else "0.75"
                dot_html = f"<span style='display:inline-block; width:8px; height:8px; border-radius:50%; background-color:{colores_barra[i]}; opacity:{icon_opacity}; box-shadow:0 1px 2px rgba(0,0,0,0.15); margin-right:0.4rem; transform:translateY(-1px);'></span>"
                barras_html += (
                    f"<div style='margin-bottom:1.1rem;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.35rem;'>"
                    f"<span class='prob-label' style='color:{text_color}; font-weight:{font_weight}; display:flex; align-items:center;'>{dot_html}{clases_nombre[i]}</span>"
                    f"<span style=\"font-family:var(--font-mono); font-size:0.8125rem; color:{text_color}; font-weight:{font_weight};\">{p:.1%}</span></div>"
                    f"<div class='prob-bar-bg'><div class='prob-bar-fill' style='--target-width:{width_pct:.1f}%; background:{colores_barra[i]}; opacity:{bar_opacity};'></div></div></div>"
                )
            st.markdown(
                f"<div class='card fade-in' style='margin-bottom:1.2rem;'>"
                f"<div style='font-family:var(--font-mono); font-size:0.8125rem; letter-spacing:0.1em; text-transform:uppercase; color:#0F172A; margin-bottom:1.1rem; font-weight:700;'>Probabilidades por nivel ESI</div>"
                f"{barras_html}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Probabilidades por nivel ESI no disponibles para este resultado.")

    # ─ SHAP ──
    st.markdown('<div style="height:1px; background:var(--border); margin:1.5rem 0 2rem 0;"></div>', unsafe_allow_html=True)

    if explicacion_shap and explicacion_shap.get("ok"):
        st.markdown(
            '<span class="sec-label">Principales factores asociados al resultado</span>'
            '<p class="result-intro">Los valores positivos aumentan el apoyo al nivel sugerido y los negativos lo reducen.</p>',
            unsafe_allow_html=True,
        )
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(factor.nombre)}</td>"
            f"<td class='{'effect-up' if factor.valor > 0 else 'effect-down'}'>{'Aumenta' if factor.valor > 0 else 'Disminuye'}</td>"
            f"<td class='contribution'>{factor.valor:+.3f}</td></tr>"
            for factor in explicacion_shap.get("factores", [])
        )
        st.markdown(
            "<div class='shap-table-wrap'><table class='shap-table'>"
            "<thead><tr><th>Factor clínico</th><th>Efecto</th><th>Contribución</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(explicacion_shap["fig"], use_container_width=True, config={"displayModeBar": False})
    else:
        detalle = ""
        if explicacion_shap and explicacion_shap.get("error"):
            detalle = f': {html.escape(explicacion_shap["error"])}'
        st.markdown(f'<p class="explanation-unavailable">Explicación no disponible{detalle}</p>', unsafe_allow_html=True)

    # ── CTA ──
    st.markdown('<div style="height:1px; background:var(--border); margin:1.5rem 0 2rem 0;"></div>', unsafe_allow_html=True)
    col_spacer_l, col_btn_bottom, col_spacer_r = st.columns([1, 1, 1])
    with col_btn_bottom:
        if st.button("Iniciar nuevo episodio", type="primary", key="nuevo_caso_bottom", icon=":material/refresh:"):
            _reset_estado()

    # El overlay persistente solo se retira cuando el marcador del episodio
    # actual existe y el resultado completo permanece estable en el DOM.
    reveal_pending = st.session_state.get("result_reveal_pending", False)
    if reveal_pending:
        st.session_state.result_reveal_pending = False
        transition_token = st.session_state.get("result_transition_token", "")
        marker_id = _ready_marker_id(transition_token)
        st.markdown(f'<div id="{marker_id}" style="display:none"></div>', unsafe_allow_html=True)
        _reveal_behind_js_overlay(
            transition_token,
            require_plot=bool(explicacion_shap and explicacion_shap.get("ok")),
        )

    _install_new_episode_click_scroll()


# ─────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────
fase_actual = st.session_state.get("fase", 1)
transition_slot = st.empty()
show_initial_welcome = (
    fase_actual == 1
    and st.session_state.get("initial_welcome_pending", False)
)

if show_initial_welcome:
    _render_initial_welcome()

if fase_actual != 2:
    _render_header()

if fase_actual == 1:
    _render_fase_1()
elif fase_actual == 2:
    _render_fase_2(transition_slot)
elif fase_actual == 25:
    _render_fase_resultado_preparacion(transition_slot)
elif fase_actual == 3:
    _render_fase_3(transition_slot)
else:
    st.session_state.fase = 1
    st.rerun()

if fase_actual in (1, 3):
    _render_footer()

if show_initial_welcome:
    _control_initial_welcome()
    st.session_state.initial_welcome_pending = False
