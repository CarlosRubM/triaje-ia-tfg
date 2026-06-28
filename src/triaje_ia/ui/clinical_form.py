"""Helpers for the guided triage intake form.

This module deliberately has no Streamlit dependency so the clinical narrative
sent to the LLM can be tested as a plain transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


SEX_LABELS = {
    "M": "varon",
    "F": "mujer",
    "Otro": "otro/no binario",
}


@dataclass
class TriageFormData:
    edad: int | None = None
    sexo: str = ""
    metodo_llegada: str = "desconocido"
    motivo_consulta: str = ""
    sintomas_frecuentes: list[str] = field(default_factory=list)
    sintomas_adicionales: str = ""
    signos_alarma: list[str] = field(default_factory=list)
    duracion_sintomas: str = ""
    presion_sistolica: int | None = None
    presion_diastolica: int | None = None
    frecuencia_cardiaca: int | None = None
    frecuencia_respiratoria: int | None = None
    saturacion_oxigeno: float | None = None
    temperatura: float | None = None
    nivel_dolor: int | None = None
    antecedentes: str = ""
    medicacion: str = ""
    observaciones: str = ""


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _split_terms(value: str | None) -> list[str]:
    raw = (value or "").replace(";", ",").replace("\n", ",")
    terms = [_clean_text(part) for part in raw.split(",")]

    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in terms:
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            unique_terms.append(term)
    return unique_terms


def _merge_terms(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for item in group:
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


def _format_list(items: list[str]) -> str:
    return ", ".join(items) if items else "no registrado"


def _format_sentence(value: str | None, empty: str) -> str:
    text = _clean_text(value)
    if not text:
        return empty
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _format_vitals(data: TriageFormData) -> str:
    vitals: list[str] = []
    if data.presion_sistolica is not None and data.presion_diastolica is not None:
        vitals.append(
            f"TA {data.presion_sistolica}/{data.presion_diastolica} mmHg"
        )
    elif data.presion_sistolica is not None:
        vitals.append(f"TA sistólica {data.presion_sistolica} mmHg")
    elif data.presion_diastolica is not None:
        vitals.append(f"TA diastólica {data.presion_diastolica} mmHg")

    if data.frecuencia_cardiaca is not None:
        vitals.append(f"FC {data.frecuencia_cardiaca} lpm")
    if data.frecuencia_respiratoria is not None:
        vitals.append(f"FR {data.frecuencia_respiratoria} rpm")
    if data.saturacion_oxigeno is not None:
        vitals.append(f"SpO₂ {data.saturacion_oxigeno:g}%")
    if data.temperatura is not None:
        vitals.append(f"temperatura {data.temperatura:g} °C")

    return ", ".join(vitals) if vitals else "constantes no registradas"


def generar_narrativa_triaje(data: TriageFormData) -> str:
    """Build a structured clinical narrative for the LLM extractor."""
    sintomas = _merge_terms(
        data.sintomas_frecuentes,
        _split_terms(data.sintomas_adicionales),
    )
    antecedentes = _split_terms(data.antecedentes)
    medicacion = _split_terms(data.medicacion)

    paciente = "Paciente"
    if data.edad is not None:
        paciente += f" de {data.edad} años"
    if data.sexo:
        paciente += f", sexo {data.sexo} ({SEX_LABELS.get(data.sexo, data.sexo)})"
    paciente += "."

    lines = [
        "Texto estructurado de triaje para extracción clínica:",
        paciente,
        f"Método de llegada: {_clean_text(data.metodo_llegada) or 'desconocido'}.",
        f"Motivo principal de consulta: {_clean_text(data.motivo_consulta) or 'no registrado'}.",
        f"Duración o evolución: {_clean_text(data.duracion_sintomas) or 'no registrada'}.",
        f"Síntomas y hallazgos referidos: {_format_list(sintomas)}.",
        f"Signos de alarma o discriminadores de prioridad: {_format_list(data.signos_alarma)}.",
        f"Constantes vitales en triaje: {_format_vitals(data)}.",
    ]

    if data.nivel_dolor is None:
        lines.append("Dolor: escala EVA/NRS no registrada.")
    else:
        lines.append(f"Dolor: EVA/NRS {data.nivel_dolor}/10.")

    lines.extend([
        f"Antecedentes relevantes: {_format_list(antecedentes)}.",
        f"Medicación habitual: {_format_list(medicacion)}.",
    ])

    observaciones = _clean_text(data.observaciones)
    if observaciones:
        lines.append(f"Observaciones libres de triaje: {observaciones}.")

    return "\n".join(lines)


def generar_narrativa_triaje_texto_libre(data: TriageFormData) -> str:
    """Build a structured narrative while preserving free clinical text."""
    sintomas_guiados = _merge_terms(data.sintomas_frecuentes)
    relato = _clean_text(data.sintomas_adicionales)
    antecedentes = _clean_text(data.antecedentes)
    medicacion = _clean_text(data.medicacion)

    paciente = "Paciente"
    if data.edad is not None:
        paciente += f" de {data.edad} años"
    if data.sexo:
        paciente += f", sexo {data.sexo} ({SEX_LABELS.get(data.sexo, data.sexo)})"
    paciente += "."

    lines = [
        "Narrativa estructurada de triaje:",
        paciente,
        f"Método de llegada: {_clean_text(data.metodo_llegada) or 'desconocido'}.",
        f"Motivo principal de consulta: {_clean_text(data.motivo_consulta) or 'no registrado'}.",
        f"Duración o evolución: {_clean_text(data.duracion_sintomas) or 'no registrada'}.",
    ]

    if sintomas_guiados:
        lines.append(
            f"Discriminadores de prioridad observados: {_format_list(sintomas_guiados)}."
        )
    lines.append(f"Relato clínico de triaje: {_format_sentence(relato, 'no registrado.')}")
    lines.append(f"Constantes vitales en triaje: {_format_vitals(data)}.")

    if data.nivel_dolor is None:
        lines.append("Dolor: escala EVA/NRS no registrada.")
    else:
        lines.append(f"Dolor: EVA/NRS {data.nivel_dolor}/10.")

    lines.extend([
        f"Antecedentes médicos relevantes: {_format_sentence(antecedentes, 'no registrados.')}",
        f"Fármacos habituales referidos: {_format_sentence(medicacion, 'no registrados.')}",
    ])

    observaciones = _clean_text(data.observaciones)
    if observaciones:
        lines.append(f"Observaciones de triaje: {_format_sentence(observaciones, 'no registradas.')}")

    return "\n".join(lines)


def validar_entrada_triaje(data: TriageFormData) -> list[str]:
    """Return non-blocking UX warnings for incomplete triage input."""
    avisos: list[str] = []

    if data.edad is None:
        avisos.append("Edad no registrada.")
    if not data.sexo:
        avisos.append("Sexo no registrado.")
    if not _clean_text(data.motivo_consulta):
        avisos.append("Falta motivo principal o sintoma guia.")

    ta_incompleta = (
        data.presion_sistolica is None
        and data.presion_diastolica is not None
    ) or (
        data.presion_sistolica is not None
        and data.presion_diastolica is None
    )
    if ta_incompleta:
        avisos.append("Tensión arterial incompleta: registre sistólica y diastólica.")

    vitales_registradas = [
        data.presion_sistolica is not None and data.presion_diastolica is not None,
        data.frecuencia_cardiaca is not None,
        data.frecuencia_respiratoria is not None,
        data.saturacion_oxigeno is not None,
        data.temperatura is not None,
    ]
    if sum(vitales_registradas) <= 1:
        avisos.append(
            "Constantes no registradas o muy incompletas; el analisis no se bloquea."
        )

    return avisos


def entrada_minima_completa(data: TriageFormData) -> bool:
    return (
        data.edad is not None
        and bool(data.sexo)
        and bool(_clean_text(data.motivo_consulta))
    )
