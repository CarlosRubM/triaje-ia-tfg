"""Normalizacion ligera del VectorClinico extraido por el LLM."""

from __future__ import annotations

from triaje_ia.llm.schemas import VectorClinico

_SINTOMAS_EQUIVALENTES = {
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
    "bisoprolol": "beta blockers cardiac selective",
    "atenolol": "beta blockers cardiac selective",
    "warfarin": "vitamin K antagonists",
    "acenocumarol": "vitamin K antagonists",
    "sintrom": "vitamin K antagonists",
    "furosemide": "loop diuretics",
    "furosemida": "loop diuretics",
    "insulin": "insulin analogues",
    "insulina": "insulin analogues",
    "aspirin": "antiplatelet agents",
    "aspirina": "antiplatelet agents",
    "adiro": "antiplatelet agents",
    "clopidogrel": "antiplatelet agents",
    "omeprazole": "proton pump inhibitors",
    "omeprazol": "proton pump inhibitors",
}


def _limpiar_texto(valor: str) -> str:
    return " ".join(valor.strip().lower().split())


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
