"""Pure helpers for reviewing and correcting LLM-extracted clinical vectors."""

from __future__ import annotations

from triaje_ia.llm.normalizer import normalizar_vector_clinico
from triaje_ia.llm.schemas import VectorClinico
from triaje_ia.ui.clinical_display import internalize_review_term


def split_review_terms(value: str | None) -> list[str]:
    """Split comma/semicolon/newline separated review terms preserving order."""
    raw = (value or "").replace(";", ",").replace("\n", ",")
    seen: set[str] = set()
    terms: list[str] = []
    for part in raw.split(","):
        term = " ".join(part.strip().split())
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def join_review_terms(values: list[str] | tuple[str, ...] | None) -> str:
    """Format extracted list fields as editable comma-separated text."""
    return ", ".join(str(v) for v in (values or []) if str(v).strip())


def internalize_review_terms(value: str | None) -> list[str]:
    """Split Spanish-visible review terms and convert known values to internals."""
    return [
        internal
        for term in split_review_terms(value)
        if (internal := internalize_review_term(term))
    ]


def construir_vector_desde_revision(
    *,
    edad: int,
    sexo: str,
    sintomas_presentes: str,
    patologias_previas: str = "",
    medicacion_habitual: str = "",
    presion_sistolica: int | None = None,
    presion_diastolica: int | None = None,
    frecuencia_cardiaca: int | None = None,
    frecuencia_respiratoria: int | None = None,
    saturacion_oxigeno: float | None = None,
    temperatura: float | None = None,
    nivel_dolor: int | None = None,
    duracion_sintomas: str | None = None,
    metodo_llegada: str = "desconocido",
) -> VectorClinico:
    """Build, validate and normalize a VectorClinico after human review."""
    vector = VectorClinico(
        edad=edad,
        sexo=sexo,
        sintomas_presentes=internalize_review_terms(sintomas_presentes),
        patologias_previas=split_review_terms(patologias_previas),
        medicacion_habitual=internalize_review_terms(medicacion_habitual),
        presion_sistolica=presion_sistolica,
        presion_diastolica=presion_diastolica,
        frecuencia_cardiaca=frecuencia_cardiaca,
        frecuencia_respiratoria=frecuencia_respiratoria,
        saturacion_oxigeno=saturacion_oxigeno,
        temperatura=temperatura,
        nivel_dolor=nivel_dolor,
        duracion_sintomas=(duracion_sintomas or None),
        metodo_llegada=metodo_llegada or "desconocido",
    )
    return normalizar_vector_clinico(vector)
