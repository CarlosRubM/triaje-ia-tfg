import numpy as np
import pytest
from triaje_ia.inference.adapter import _celsius_a_fahrenheit, _mapear_sexo, vectorclinico_a_features
from triaje_ia.data.features import TODAS_FEATURES
from triaje_ia.llm.schemas import VectorClinico


def test_celsius_a_fahrenheit_37():
    assert _celsius_a_fahrenheit(37.0) == pytest.approx(98.6, abs=0.1)


def test_celsius_a_fahrenheit_none():
    assert _celsius_a_fahrenheit(None) is None


def test_mapear_sexo_m():
    assert _mapear_sexo("M") == "M"


def test_mapear_sexo_f():
    assert _mapear_sexo("F") == "F"


def test_mapear_sexo_otro_conservador():
    assert _mapear_sexo("Otro") == "M"


def test_vectorclinico_a_features_produce_88_columnas():
    v = VectorClinico(
        edad=45,
        sexo="M",
        sintomas_presentes=["chest pain"],
        medicacion_habitual=["atenolol", "warfarin"],
        presion_sistolica=130,
        presion_diastolica=85,
        frecuencia_cardiaca=90,
        frecuencia_respiratoria=18,
        saturacion_oxigeno=97.0,
        temperatura=37.0,
        nivel_dolor=5,
    )
    df = vectorclinico_a_features(v)
    assert list(df.columns) == TODAS_FEATURES
    assert len(df) == 1


def test_vectorclinico_a_features_sin_vitales():
    v = VectorClinico(
        edad=70,
        sexo="F",
        sintomas_presentes=["dyspnea"],
    )
    df = vectorclinico_a_features(v)
    assert list(df.columns) == TODAS_FEATURES
    assert df["o2sat_missing"].iloc[0] == 1


def test_qsofa_y_news2_no_usan_pain_missing_como_proxy_ams():
    v = VectorClinico(
        edad=45,
        sexo="M",
        sintomas_presentes=["checkup"],
        presion_sistolica=120,
        presion_diastolica=80,
        frecuencia_cardiaca=80,
        frecuencia_respiratoria=16,
        saturacion_oxigeno=98.0,
        temperatura=37.0,
        nivel_dolor=None,
    )
    df = vectorclinico_a_features(v)

    assert df["pain_missing"].iloc[0] == 1
    assert df["news2"].iloc[0] == 0
    assert df["qsofa"].iloc[0] == 0
    assert df["qsofa_positivo"].iloc[0] == 0


def test_qsofa_positivo_exige_dos_criterios_fisiologicos():
    v = VectorClinico(
        edad=45,
        sexo="M",
        sintomas_presentes=["dyspnea"],
        presion_sistolica=100,
        presion_diastolica=70,
        frecuencia_cardiaca=80,
        frecuencia_respiratoria=22,
        saturacion_oxigeno=98.0,
        temperatura=37.0,
        nivel_dolor=None,
    )
    df = vectorclinico_a_features(v)

    assert df["pain_missing"].iloc[0] == 1
    assert df["qsofa"].iloc[0] == 2
    assert df["qsofa_positivo"].iloc[0] == 1


def test_patologias_previas_activan_macroantecedentes_textuales():
    v = VectorClinico(
        edad=70,
        sexo="M",
        sintomas_presentes=["dyspnea"],
        patologias_previas=[
            "EPOC",
            "diabetes mellitus",
            "insuficiencia renal cronica",
            "fibrilacion auricular",
        ],
    )
    df = vectorclinico_a_features(v)

    assert df["hx_respiratorio"].iloc[0] == 1
    assert df["hx_metabolico_renal"].iloc[0] == 1
    assert df["hx_cardiaco"].iloc[0] == 1


def test_patologias_previas_vacias_dejan_hx_a_cero():
    v = VectorClinico(
        edad=50,
        sexo="F",
        sintomas_presentes=["headache"],
        patologias_previas=[],
    )
    df = vectorclinico_a_features(v)

    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    assert int(df[hx_cols].sum(axis=1).iloc[0]) == 0


def test_proxy_antecedentes_no_infiere_frecuentacion():
    v = VectorClinico(
        edad=65,
        sexo="F",
        sintomas_presentes=["abdominal pain"],
        patologias_previas=["cardiopatia cronica", "EPOC"],
    )
    df = vectorclinico_a_features(v)

    assert df["hx_cardiaco"].iloc[0] == 1
    assert df["hx_respiratorio"].iloc[0] == 1
    assert df["n_visitas_previas"].iloc[0] == 0
    assert df["visitas_ultimo_mes"].iloc[0] == 0
    assert df["visitas_ultimo_año"].iloc[0] == 0
    assert df["frecuentador"].iloc[0] == 0
    assert df["primera_visita"].iloc[0] == 1
