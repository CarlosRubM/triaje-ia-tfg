import unicodedata

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal


class VectorClinico(BaseModel):
    """
    Esquema de extracción estructurada para triaje de urgencias.
    Las variables están alineadas con MIMIC-IV-ED para compatibilidad directa.
    """
    edad: int = Field(..., ge=0, le=120, description="Edad en años")
    sexo: Literal["M", "F", "Otro"] = Field(..., description="Sexo biológico")

    sintomas_presentes: List[str] = Field(
        ..., description="Síntomas agudos detectados en la narrativa"
    )
    patologias_previas: List[str] = Field(
        default=[], description="Enfermedades crónicas del paciente"
    )
    medicacion_habitual: List[str] = Field(
        default=[], description="Medicación domiciliaria habitual"
    )

    # Constantes vitales — opcionales porque pueden no mencionarse
    presion_sistolica: Optional[int] = Field(None, ge=40, le=300, description="mmHg")
    presion_diastolica: Optional[int] = Field(None, ge=20, le=200, description="mmHg")
    frecuencia_cardiaca: Optional[int] = Field(None, ge=20, le=300, description="lpm")
    frecuencia_respiratoria: Optional[int] = Field(None, ge=4, le=60, description="rpm")
    saturacion_oxigeno: Optional[float] = Field(None, ge=50.0, le=100.0, description="SpO2 %")
    temperatura: Optional[float] = Field(None, ge=30.0, le=45.0, description="°C")

    nivel_dolor: Optional[int] = Field(None, ge=0, le=10, description="Escala 0-10")
    duracion_sintomas: Optional[str] = Field(None, description="Tiempo de evolución")
    metodo_llegada: Literal[
        "ambulancia",
        "helicoptero",
        "autonomo",
        "otro",
        "desconocido",
    ] = Field(
        "desconocido",
        description="Metodo de llegada a urgencias si se menciona en la narrativa",
    )

    @field_validator("metodo_llegada", mode="before")
    @classmethod
    def normalizar_metodo_llegada(cls, value: object) -> str:
        if value is None or value == "":
            return "desconocido"
        texto = unicodedata.normalize("NFKD", str(value).strip().lower())
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        equivalencias = {
            "ambulance": "ambulancia",
            "ambulancia": "ambulancia",
            "uvi movil": "ambulancia",
            "samu": "ambulancia",
            "061": "ambulancia",
            "ems": "ambulancia",
            "helicopter": "helicoptero",
            "helicoptero": "helicoptero",
            "hems": "helicoptero",
            "heli": "helicoptero",
            "walk in": "autonomo",
            "autonomo": "autonomo",
            "por sus medios": "autonomo",
            "coche propio": "autonomo",
            "other": "otro",
            "otro": "otro",
            "unknown": "desconocido",
            "desconocido": "desconocido",
        }
        return equivalencias.get(texto, texto)
