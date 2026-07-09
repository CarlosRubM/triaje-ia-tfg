"""Normalizacion ligera del VectorClinico extraido por el LLM."""

from __future__ import annotations

import unicodedata

from triaje_ia.llm.schemas import VectorClinico

_SINTOMAS_EQUIVALENTES = {
    "dolor toracico": "chest pain",
    "disnea": "dyspnea",
    "fiebre": "fever",
    "tos": "cough",
    "dolor abdominal": "abdominal pain",
    "vomitos": "vomiting",
    "nauseas": "nausea",
    "mareo": "dizziness",
    "sincope": "syncope",
    "sangrado": "bleeding",
    "sangrado activo": "bleeding",
    "traumatismo": "trauma",
    "trauma": "trauma",
    "caida": "fall",
    "edema labios y lengua": "angioedema",
    "edema labial": "angioedema",
    "estridor": "stridor",
    "convulsion": "seizure",
    "convulsiones": "seizure",
    "cefalea trueno": "thunderclap headache",
    "cefalea en trueno": "thunderclap headache",
    "rigidez nuca": "neck stiffness",
    "rigidez de nuca": "neck stiffness",
    "fotofobia": "photophobia",
    "cianosis": "cyanosis",
    "palidez": "pallor",
    "palido": "pallor",
    "vertigo": "vertigo",
    "perdida visual": "vision loss",
    "perdida de vision": "vision loss",
    "loss of vision": "vision loss",
    "oido taponado": "hearing loss",
    "oido tapado": "hearing loss",
    "sensacion de oido tapado": "hearing loss",
    "decreased hearing": "hearing loss",
    "herida simple": "minor wound",
    "corte superficial": "minor wound",
    "quemadura pequena": "minor burn",
    "ampolla": "blister",
    "hematochezia": "rectal bleeding",
    "productive cough": "cough",
    "bajo nivel de conciencia": "altered mental status",
    "alteracion del nivel de conciencia": "altered mental status",
    "alteracion del nivel de consciencia": "altered mental status",
    "shortness of breath": "dyspnea",
    "difficulty breathing": "dyspnea",
    "breathlessness": "dyspnea",
    "chest discomfort": "chest pain",
    "thoracic pain": "chest pain",
    "abdominal ache": "abdominal pain",
    "stomach pain": "abdominal pain",
    "vomits": "vomiting",
    "emesis": "vomiting",
    "nauseas": "nausea",
    "feverish": "fever",
    "confusion": "altered mental status",
    "low level of consciousness": "altered mental status",
    "prescription refill": "medication refill",
    "medication renewal": "medication refill",
    "repeat prescription": "medication refill",
    "administrative paperwork": "administrative request",
    "medical certificate": "administrative request",
    "sick note": "administrative request",
    "wound review": "wound check",
    "surgical wound review": "wound check",
    "stitch removal": "suture removal",
    "dry eyes": "dry eye",
    "eye dryness": "dry eye",
    "ear blockage": "ear fullness",
    "blocked ear": "ear fullness",
    "minor cut": "minor wound",
    "superficial cut": "minor wound",
    "somnolencia": "altered mental status",
    "somnolent": "altered mental status",
    "drowsiness": "altered mental status",
    "diaphoresis": "sweating",
    "sudoracion": "sweating",
    "renovacion de medicacion": "medication refill",
    "consulta administrativa": "administrative request",
}

_SINTOMAS_VAGOS = {
    "bad general condition",
    "feels unwell",
    "general discomfort",
    "general malaise",
    "malaise",
    "unwell",
}

_MEDICACION_EQUIVALENTE = {
    "metformin": "biguanides",
    "metformina": "biguanides",
    "biguanidas": "biguanides",
    "bisoprolol": "beta blockers cardiac selective",
    "atenolol": "beta blockers cardiac selective",
    "betabloqueantes cardioselectivos": "beta blockers cardiac selective",
    "beta blockers": "beta blockers cardiac selective",
    "warfarin": "anticoagulants - coumarin",
    "acenocumarol": "anticoagulants - coumarin",
    "sintrom": "anticoagulants - coumarin",
    "anticoagulantes cumarinicos": "anticoagulants - coumarin",
    "furosemide": "diuretic - loop",
    "furosemida": "diuretic - loop",
    "loop diuretics": "diuretic - loop",
    "diureticos de asa": "diuretic - loop",
    "insulin": "insulin analogs",
    "insulina": "insulin analogs",
    "analogos de insulina": "insulin analogs",
    "aspirin": "salicylate analgesics",
    "aspirina": "salicylate analgesics",
    "adiro": "salicylate analgesics",
    "clopidogrel": "thienopyridine",
    "omeprazole": "proton pump inhibitors",
    "omeprazol": "proton pump inhibitors",
    "opioid analgesics": "analgesic opioid agonists",
    "benzodiazepines": "antianxiety agent - benzodiazepines",
    "bronchodilators - inhalation": "asthma/copd therapy - beta 2-adrenergic agents",
    "steroids - inhalation": "asthma therapy - inhaled corticosteroids",
}


def _limpiar_texto(valor: str) -> str:
    texto = unicodedata.normalize("NFKD", valor.strip().lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.split())


def _normalizar_lista(
    valores: list[str],
    equivalencias: dict[str, str],
    max_elementos: int | None = None,
) -> list[str]:
    normalizados: list[str] = []
    vistos: set[str] = set()

    for valor in valores:
        limpio = _limpiar_texto(valor)
        if not limpio:
            continue
        limpio = equivalencias.get(limpio, limpio)
        if limpio not in vistos:
            normalizados.append(limpio)
            vistos.add(limpio)

    if max_elementos is not None:
        return normalizados[:max_elementos]
    return normalizados


def _quitar_sintomas_vagos_si_hay_mejores(sintomas: list[str]) -> list[str]:
    if len(sintomas) <= 1:
        return sintomas
    filtrados = [s for s in sintomas if s not in _SINTOMAS_VAGOS]
    return filtrados or sintomas


def normalizar_vector_clinico(vector: VectorClinico) -> VectorClinico:
    """
    Aplica reglas simples y deterministas tras la extraccion LLM.

    No infiere datos nuevos ni cambia constantes vitales. Solo limpia texto,
    unifica sinonimos frecuentes y evita duplicados para estabilizar el adapter.
    """
    sintomas = _normalizar_lista(
        vector.sintomas_presentes,
        _SINTOMAS_EQUIVALENTES,
        max_elementos=12,
    )
    sintomas = _quitar_sintomas_vagos_si_hay_mejores(sintomas)

    patologias = _normalizar_lista(vector.patologias_previas, {}, max_elementos=12)
    medicacion = _normalizar_lista(
        vector.medicacion_habitual,
        _MEDICACION_EQUIVALENTE,
        max_elementos=12,
    )

    return vector.model_copy(
        update={
            "sintomas_presentes": sintomas,
            "patologias_previas": patologias,
            "medicacion_habitual": medicacion,
        }
    )
