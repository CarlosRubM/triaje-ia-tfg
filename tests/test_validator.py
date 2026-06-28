"""Tests para el módulo de validación semántica del LLM."""

from triaje_ia.llm.schemas import VectorClinico
from triaje_ia.llm.validator import (
    AlertaValidacion,
    NivelAlerta,
    resumen_validacion,
    validar_vector_clinico,
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


def test_no_avisa_temperatura_si_fiebre_esta_negada():
    v = _vector_basico(temperatura=None)

    alertas = validar_vector_clinico(
        v,
        narrativa="Caida con herida sangrante. Niega dolor toracico y fiebre.",
    )

    assert not any(a.campo == "temperatura" for a in alertas)


def test_edad_anciano_inconsistente():
    v = _vector_basico(edad=25)
    alertas = validar_vector_clinico(v, narrativa="Anciano de 25 años")
    assert any("anciano" in a.mensaje.lower() or "edad" in a.campo for a in alertas)


def test_sintomas_vacios_narrativa_larga():
    v = _vector_basico(sintomas_presentes=[])
    narrativa = (
        "Paciente que acude por dolor torácico de 2 horas "
        "de evolución con irradiación"
    )
    alertas = validar_vector_clinico(v, narrativa=narrativa)
    assert any("síntomas" in a.campo or "sintomas" in a.campo for a in alertas)


def test_sexo_incoherente_mujer():
    v = _vector_basico(sexo="M")
    alertas = validar_vector_clinico(
        v, narrativa="Mujer de 45 años con dolor abdominal"
    )
    assert any("sexo" in a.campo for a in alertas)


def test_no_confunde_mujer_acompanante_con_sexo_del_paciente():
    v = _vector_basico(sexo="M")

    alertas = validar_vector_clinico(
        v,
        narrativa=(
            "Varón de 67 años que acude acompañado por su mujer "
            "por dolor torácico."
        ),
    )

    assert not any(a.campo == "sexo" for a in alertas)


def test_avisa_si_texto_menciona_dolor_sin_escala_extraida():
    v = _vector_basico(nivel_dolor=None)

    alertas = validar_vector_clinico(v, narrativa="Paciente con dolor toracico intenso")

    assert any(a.campo == "nivel_dolor" for a in alertas)


def test_no_avisa_escala_dolor_si_el_dolor_esta_negado():
    v = _vector_basico(nivel_dolor=None)

    alertas = validar_vector_clinico(
        v, narrativa="Paciente estable. Niega dolor toracico"
    )

    assert not any(a.campo == "nivel_dolor" for a in alertas)


def test_avisa_si_texto_menciona_saturacion_sin_spo2_extraida():
    v = _vector_basico(saturacion_oxigeno=None)

    alertas = validar_vector_clinico(v, narrativa="Sat baja segun triaje inicial")

    assert any(a.campo == "saturacion_oxigeno" for a in alertas)


def test_avisa_si_texto_menciona_tension_incompleta():
    v = _vector_basico(presion_sistolica=130, presion_diastolica=None)

    alertas = validar_vector_clinico(v, narrativa="TA tomada en triaje")

    assert any(a.campo == "presion_arterial" for a in alertas)


def test_avisa_si_constantes_mencionadas_no_se_extraen():
    v = _vector_basico(
        presion_sistolica=None,
        presion_diastolica=None,
        frecuencia_cardiaca=None,
        frecuencia_respiratoria=None,
        saturacion_oxigeno=None,
        temperatura=None,
    )

    alertas = validar_vector_clinico(
        v,
        narrativa=(
            "TA 130/76, FC 74, FR 16, SatO2 98%, "
            "T 36.5, Glasgow 15"
        ),
    )

    campos = {a.campo for a in alertas}
    assert "frecuencia_cardiaca" in campos
    assert "frecuencia_respiratoria" in campos
    assert "saturacion_oxigeno" in campos
    assert "temperatura" in campos
    assert "presion_arterial" in campos
    assert "vector_incompleto" in campos


def test_resumen_sin_alertas():
    assert resumen_validacion([]) == "Sin alertas de validacion"


def test_resumen_con_alertas():
    alertas = [
        AlertaValidacion("test", "msg", NivelAlerta.ERROR),
        AlertaValidacion("test2", "msg2", NivelAlerta.WARNING),
    ]
    r = resumen_validacion(alertas)
    assert "1 errores" in r
    assert "1 warnings" in r
