"""Tests para el módulo de validación semántica del LLM."""

from triaje_ia.llm.schemas import VectorClinico
from triaje_ia.llm.validator import (
    AlertaValidacion,
    NivelAlerta,
    validar_vector_clinico,
    resumen_validacion,
)


def _vector_basico(**kwargs) -> VectorClinico:
    defaults = dict(
        edad=50, sexo="M", sintomas_presentes=["dolor torácico"],
    )
    defaults.update(kwargs)
    return VectorClinico(**defaults)


def test_sin_alertas_vector_valido():
    v = _vector_basico(
        presion_sistolica=130, presion_diastolica=80,
        frecuencia_cardiaca=75, temperatura=36.5,
    )
    alertas = validar_vector_clinico(v)
    assert len(alertas) == 0


def test_sistolica_menor_que_diastolica():
    v = _vector_basico(presion_sistolica=70, presion_diastolica=120)
    alertas = validar_vector_clinico(v)
    assert any(a.nivel == NivelAlerta.ERROR for a in alertas)
    assert any("inversión" in a.mensaje.lower() or "sistólica" in a.mensaje.lower()
               for a in alertas)


def test_fiebre_con_texto_afebril():
    v = _vector_basico(temperatura=39.0)
    alertas = validar_vector_clinico(v, narrativa="Paciente afebril en domicilio")
    assert any(a.nivel == NivelAlerta.WARNING and "afebril" in a.mensaje
               for a in alertas)


def test_edad_anciano_inconsistente():
    v = _vector_basico(edad=25)
    alertas = validar_vector_clinico(v, narrativa="Anciano de 25 años")
    assert any("anciano" in a.mensaje.lower() or "edad" in a.campo for a in alertas)


def test_sintomas_vacios_narrativa_larga():
    v = _vector_basico(sintomas_presentes=[])
    narrativa = "Paciente que acude por dolor torácico de 2 horas de evolución con irradiación"
    alertas = validar_vector_clinico(v, narrativa=narrativa)
    assert any("síntomas" in a.campo or "sintomas" in a.campo for a in alertas)


def test_sexo_incoherente_mujer():
    v = _vector_basico(sexo="M")
    alertas = validar_vector_clinico(v, narrativa="Mujer de 45 años con dolor abdominal")
    assert any("sexo" in a.campo for a in alertas)


def test_resumen_sin_alertas():
    assert "✅" in resumen_validacion([])


def test_resumen_con_alertas():
    alertas = [
        AlertaValidacion("test", "msg", NivelAlerta.ERROR),
        AlertaValidacion("test2", "msg2", NivelAlerta.WARNING),
    ]
    r = resumen_validacion(alertas)
    assert "1 errores" in r
    assert "1 warnings" in r
