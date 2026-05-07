"""
src/triaje_ia/llm/validator.py
────────────────────────────────
Validación semántica del VectorClinico extraído por el LLM.

Detecta errores que Pydantic no ve (coherencia, no tipos/rangos).
Devuelve warnings, no errores — el sistema sigue funcionando.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from triaje_ia.llm.schemas import VectorClinico


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
        prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
        return f"{prefix.get(self.nivel.value, '?')} [{self.campo}] {self.mensaje}"


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
                mensaje=f"Sistólica ({v.presion_sistolica}) ≤ Diastólica ({v.presion_diastolica}). Posible inversión.",
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
        mujer = any(m in texto for m in ["mujer", "female", "embarazada"])
        hombre = any(m in texto for m in ["varón", "hombre", "male"])
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

    return alertas


def resumen_validacion(alertas: list[AlertaValidacion]) -> str:
    if not alertas:
        return "✅ Sin alertas de validación"
    n_e = sum(1 for a in alertas if a.nivel == NivelAlerta.ERROR)
    n_w = sum(1 for a in alertas if a.nivel == NivelAlerta.WARNING)
    lines = [f"Validación: {n_e} errores, {n_w} warnings"]
    for a in alertas:
        lines.append(f"  {a}")
    return "\n".join(lines)
