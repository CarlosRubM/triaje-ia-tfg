"""Helpers to present clinical model terms in Spanish without changing internals."""

from __future__ import annotations

import unicodedata


def _key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.replace("_", " ").split())


TERM_DISPLAY_MAP = {
    # Symptoms and chief complaints used by the internal model contract.
    "chest pain": "dolor torácico",
    "dyspnea": "disnea",
    "shortness of breath": "disnea",
    "abdominal pain": "dolor abdominal",
    "right lower quadrant abdominal pain": "dolor en fosa ilíaca derecha",
    "left lower quadrant abdominal pain": "dolor en fosa ilíaca izquierda",
    "right upper quadrant abdominal pain": "dolor en hipocondrio derecho",
    "headache": "cefalea",
    "thunderclap headache": "cefalea brusca intensa",
    "fever": "fiebre",
    "cough": "tos",
    "nausea": "náuseas",
    "vomiting": "vómitos",
    "dizziness": "mareo",
    "syncope": "síncope",
    "presyncope": "presíncope",
    "altered mental status": "alteración del nivel de conciencia",
    "bleeding": "sangrado activo",
    "rectal bleeding": "rectorragia",
    "trauma": "traumatismo",
    "fall": "caída",
    "sweating": "sudoración",
    "bradycardia": "bradicardia",
    "tachycardia": "taquicardia",
    "hypoxemia": "hipoxemia",
    "ear pain": "otalgia",
    "ear fullness": "sensación de oído tapado",
    "dental pain": "dolor dental",
    "toe pain": "dolor en dedo del pie",
    "ingrown toenail": "uña encarnada",
    "redness": "enrojecimiento",
    "minor wound": "herida leve",
    "wound check": "revisión de herida",
    "suture removal": "retirada de sutura",
    "dry eye": "sequedad ocular",
    "pruritic rash": "exantema pruriginoso",
    "administrative request": "consulta administrativa",
    "medication refill": "renovación de medicación",
    "checkup": "revisión",
    # Medication classes.
    "biguanides": "biguanidas",
    "beta blockers cardiac selective": "betabloqueantes cardioselectivos",
    "anticoagulants - coumarin": "anticoagulantes cumarínicos",
    "vitamin k antagonists": "antagonistas de la vitamina K",
    "diuretic - loop": "diuréticos de asa",
    "loop diuretics": "diuréticos de asa",
    "insulin analogs": "análogos de insulina",
    "insulin analogues": "análogos de insulina",
    "salicylate analgesics": "salicilatos / antiagregantes",
    "thienopyridine": "tienopiridinas / antiagregantes",
    "proton pump inhibitors": "inhibidores de la bomba de protones",
    "analgesic opioid agonists": "opioides",
    "opioid analgesics": "opioides",
    "antianxiety agent - benzodiazepines": "benzodiacepinas",
    "benzodiazepines": "benzodiacepinas",
    "asthma/copd therapy - beta 2-adrenergic agents": "broncodilatadores inhalados",
    "asthma therapy - inhaled corticosteroids": "corticoides inhalados",
    "antipsychotics - second generation": "antipsicóticos de segunda generación",
    "ssris": "ISRS",
    "ace inhibitors": "IECA",
    # Common comorbidities and abbreviations.
    "hta": "hipertensión arterial",
    "hipertension arterial": "hipertensión arterial",
    "fibrilacion auricular": "fibrilación auricular",
    "fibrilacion auricular cronica": "fibrilación auricular crónica",
    "epoc": "EPOC",
    "diabetes": "diabetes",
    "diabetes mellitus": "diabetes mellitus",
    "diabetes mellitus tipo 2": "diabetes mellitus tipo 2",
    "cardiopatia cronica": "cardiopatía crónica",
    "ictus previo": "ictus previo",
}


REVIEW_INTERNAL_MAP = {
    _key(display): internal for internal, display in TERM_DISPLAY_MAP.items()
}
REVIEW_INTERNAL_MAP.update(
    {
        "dolor toracico": "chest pain",
        "dolor en el pecho": "chest pain",
        "opresion toracica": "chest pain",
        "disnea": "dyspnea",
        "falta de aire": "dyspnea",
        "dificultad respiratoria": "dyspnea",
        "fiebre": "fever",
        "tos": "cough",
        "dolor abdominal": "abdominal pain",
        "nauseas": "nausea",
        "vomitos": "vomiting",
        "mareo": "dizziness",
        "sincope": "syncope",
        "presincope": "presyncope",
        "sangrado": "bleeding",
        "sangrado activo": "bleeding",
        "rectorragia": "rectal bleeding",
        "traumatismo": "trauma",
        "caida": "fall",
        "bajo nivel de conciencia": "altered mental status",
        "alteracion del nivel de conciencia": "altered mental status",
        "alteracion del nivel de consciencia": "altered mental status",
        "confusion": "altered mental status",
        "somnolencia": "altered mental status",
        "anticoagulantes cumarinicos": "anticoagulants - coumarin",
        "betabloqueantes cardioselectivos": "beta blockers cardiac selective",
        "biguanidas": "biguanides",
        "diureticos de asa": "diuretic - loop",
        "analogos de insulina": "insulin analogs",
    }
)


FEATURE_DISPLAY_MAP = {
    "age": "Edad",
    "o2sat": "Saturación de oxígeno",
    "heartrate": "Frecuencia cardiaca",
    "resprate": "Frecuencia respiratoria",
    "sbp": "TA sistólica",
    "dbp": "TA diastólica",
    "temp": "Temperatura",
    "pain": "Dolor registrado",
    "news2": "NEWS2",
    "temperature_missing": "Temperatura no registrada",
    "o2sat_missing": "Saturación no registrada",
    "pain_missing": "Dolor no registrado",
    "llegada_ambulancia": "Llegada en ambulancia",
    "llegada_helicoptero": "Llegada en helicóptero",
    "llegada_autonoma": "Llegada por medios propios",
    "llegada_desconocida": "Método de llegada no registrado",
    "shock_index": "Índice de shock",
    "shock_index_alto": "Índice de shock alto",
    "shock_index_severo": "Índice de shock severo",
    "qsofa": "qSOFA",
    "qsofa_positivo": "qSOFA positivo",
}


CC_FEATURE_MAP = {
    "disnea": "disnea",
    "gi_agudo": "cuadro digestivo agudo",
    "neuro_ams": "alteración del nivel de conciencia",
    "cefalea": "cefalea",
    "trauma": "traumatismo",
    "psiquiatrico": "síntoma psiquiátrico",
    "intoxicacion": "intoxicación",
    "infeccioso": "síndrome infeccioso",
    "urologico": "síntoma urológico",
    "hemorragia_activa": "hemorragia activa",
    "alergia_anafilaxia": "alergia / anafilaxia",
    "derivacion_urgente": "derivación urgente",
    "dolor_toracico": "dolor torácico",
}


def display_clinical_term(term: str) -> str:
    """Return the Spanish clinical label for an internal term."""
    raw = str(term).strip()
    if not raw:
        return ""
    return TERM_DISPLAY_MAP.get(_key(raw), raw)


def display_clinical_terms(terms: list[str] | tuple[str, ...] | None) -> list[str]:
    """Translate a list of internal clinical terms for display."""
    return [
        translated
        for term in (terms or [])
        if (translated := display_clinical_term(str(term)))
    ]


def internalize_review_term(term: str) -> str:
    """Convert a Spanish review term back to the internal model term when known."""
    raw = str(term).strip()
    if not raw:
        return ""
    return REVIEW_INTERNAL_MAP.get(_key(raw), raw)


def join_review_terms_display(values: list[str] | tuple[str, ...] | None) -> str:
    """Format extracted list fields as Spanish editable comma-separated text."""
    return ", ".join(display_clinical_terms(values))


def display_feature_name(feature: str) -> str:
    """Return a Spanish label for a model/SHAP feature name."""
    raw = str(feature).strip()
    if not raw:
        return "Factor clínico del modelo"

    if raw in FEATURE_DISPLAY_MAP:
        return FEATURE_DISPLAY_MAP[raw]

    if raw.startswith("cc_"):
        suffix = raw.removeprefix("cc_")
        label = CC_FEATURE_MAP.get(suffix)
        if label:
            return f"Motivo: {label}"
        return "Motivo clínico"

    if raw.startswith("hx_"):
        return "Antecedente clínico"

    if raw.startswith("med_"):
        return "Medicación habitual"

    if raw.startswith("bert_svd"):
        suffix = raw.removeprefix("bert_svd_") or raw[-2:]
        return f"Contexto clínico del relato ({suffix})"

    return "Factor clínico del modelo"
