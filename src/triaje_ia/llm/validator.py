"""
src/triaje_ia/llm/validator.py
────────────────────────────────
Validación semántica del VectorClinico extraído por el LLM.

Detecta errores que Pydantic no ve (coherencia, no tipos/rangos).
Devuelve warnings, no errores — el sistema sigue funcionando.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from triaje_ia.llm.schemas import VectorClinico

_PATRON_DOLOR = re.compile(r"\b(dolor|pain|eva|escala\s+dolor|\d{1,2}\s*/\s*10)\b")
_PATRON_DOLOR_NEGADO = re.compile(
    r"\b(niega|no refiere|sin|ausencia de)\s+dolor\b"
)
_PATRON_SATURACION = re.compile(
    r"\b(sat|spo2|saturaci[oó]n|ox[ií]geno|oxygen)\b"
)
_PATRON_TENSION = re.compile(
    r"\b(ta|pa|tensi[oó]n|presi[oó]n arterial|blood pressure)\b"
)
_PATRON_ACOMPANANTE = re.compile(
    r"\b(acompañad[oa]|acompanad[oa])\s+(por\s+)?(su\s+)?"
    r"(mujer|esposa|marido|esposo|pareja)\b"
)
_PATRON_POSESIVO_ACOMPANANTE = re.compile(
    r"\bsu\s+(mujer|esposa|marido|esposo|pareja)\b"
)
_PATRON_MUJER = re.compile(r"\b(mujer|female|femenina|embarazada)\b")
_PATRON_HOMBRE = re.compile(r"\b(var[oó]n|hombre|male|masculino)\b")


class NivelAlerta(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class AlertaValidacion:
    campo: str
    mensaje: str
    nivel: NivelAlerta
    sugerencia: str = ""

    def __str__(self) -> str:
        return f"{self.nivel.value.upper()} [{self.campo}] {self.mensaje}"


def _texto_para_validar_sexo(texto: str) -> str:
    texto = _PATRON_ACOMPANANTE.sub(" ", texto)
    return _PATRON_POSESIVO_ACOMPANANTE.sub(" ", texto)


def validar_vector_clinico(
    v: VectorClinico, narrativa: str = "",
) -> list[AlertaValidacion]:
    """
    Valida coherencia semántica del VectorClinico.

    Args:
        v: VectorClinico validado por Pydantic.
        narrativa: texto original (para validación cruzada).

    Returns:
        Lista de AlertaValidacion. Vacía = sin problemas.
    """
    alertas: list[AlertaValidacion] = []
    texto = narrativa.lower() if narrativa else ""

    # 1. Sistólica > diastólica
    if v.presion_sistolica is not None and v.presion_diastolica is not None:
        if v.presion_sistolica <= v.presion_diastolica:
            alertas.append(AlertaValidacion(
                campo="presión arterial",
                mensaje=(
                    f"Sistólica ({v.presion_sistolica}) <= "
                    f"Diastólica ({v.presion_diastolica}). Posible inversión."
                ),
                nivel=NivelAlerta.ERROR,
                sugerencia="Verifique las cifras de PA.",
            ))

    # 2. Temperatura vs fiebre mencionada
    if v.temperatura is not None and texto:
        if v.temperatura >= 38.0 and any(t in texto for t in ["afebril", "sin fiebre"]):
            alertas.append(AlertaValidacion(
                campo="temperatura",
                mensaje=f"Tª {v.temperatura}°C (fiebre) pero texto dice 'afebril'.",
                nivel=NivelAlerta.WARNING,
            ))

    # 3. Edad vs descriptores
    if texto:
        if any(d in texto for d in ["ancian", "elderly"]) and v.edad < 55:
            alertas.append(AlertaValidacion(
                campo="edad",
                mensaje=f"Texto sugiere anciano pero edad={v.edad}.",
                nivel=NivelAlerta.WARNING,
            ))

    # 4. Síntomas vacíos con narrativa larga
    if len(v.sintomas_presentes) == 0 and len(texto) > 50:
        alertas.append(AlertaValidacion(
            campo="sintomas_presentes",
            mensaje=f"No se extrajeron síntomas de {len(texto)} caracteres.",
            nivel=NivelAlerta.WARNING,
        ))

    # 5. Sexo incoherente
    if texto:
        texto_sexo = _texto_para_validar_sexo(texto)
        mujer = _PATRON_MUJER.search(texto_sexo) is not None
        hombre = _PATRON_HOMBRE.search(texto_sexo) is not None
        if mujer and not hombre and v.sexo == "M":
            alertas.append(AlertaValidacion(
                campo="sexo", mensaje="Texto sugiere femenina pero sexo=M.",
                nivel=NivelAlerta.WARNING,
            ))
        if hombre and not mujer and v.sexo == "F":
            alertas.append(AlertaValidacion(
                campo="sexo", mensaje="Texto sugiere masculino pero sexo=F.",
                nivel=NivelAlerta.WARNING,
            ))

    # 6. Datos mencionados en texto pero no extraidos a campos estructurados
    if texto:
        dolor_mencionado = _PATRON_DOLOR.search(texto)
        dolor_negado = _PATRON_DOLOR_NEGADO.search(texto)
        if dolor_mencionado and not dolor_negado and v.nivel_dolor is None:
            alertas.append(AlertaValidacion(
                campo="nivel_dolor",
                mensaje="El texto menciona dolor, pero no se extrajo escala 0-10.",
                nivel=NivelAlerta.INFO,
                sugerencia="Si aparece una escala EVA/NRS, revise la extraccion.",
            ))

        if _PATRON_SATURACION.search(texto) and v.saturacion_oxigeno is None:
            alertas.append(AlertaValidacion(
                campo="saturacion_oxigeno",
                mensaje=(
                    "El texto menciona saturacion/oxigeno, "
                    "pero no se extrajo SpO2."
                ),
                nivel=NivelAlerta.WARNING,
            ))

        tension_mencionada = _PATRON_TENSION.search(texto)
        tension_incompleta = (
            v.presion_sistolica is None or v.presion_diastolica is None
        )
        if tension_mencionada and tension_incompleta:
            alertas.append(AlertaValidacion(
                campo="presion_arterial",
                mensaje=(
                    "El texto menciona TA/PA, "
                    "pero la presion arterial esta incompleta."
                ),
                nivel=NivelAlerta.WARNING,
            ))

    return alertas


def resumen_validacion(alertas: list[AlertaValidacion]) -> str:
    if not alertas:
        return "Sin alertas de validacion"
    n_e = sum(1 for a in alertas if a.nivel == NivelAlerta.ERROR)
    n_w = sum(1 for a in alertas if a.nivel == NivelAlerta.WARNING)
    lines = [f"Validación: {n_e} errores, {n_w} warnings"]
    for a in alertas:
        lines.append(f"  {a}")
    return "\n".join(lines)
