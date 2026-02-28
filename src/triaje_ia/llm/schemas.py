from pydantic import BaseModel, Field
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