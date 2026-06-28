import json
from pathlib import Path

import pytest

from triaje_ia.data.features import (
    FEATURES_BINARIAS,
    FEATURES_CATEGORICAS,
    FEATURES_CONTINUAS,
    TODAS_FEATURES,
)
from triaje_ia.inference.adapter import vectorclinico_a_features
from triaje_ia.llm.schemas import VectorClinico


def _vector(**kwargs) -> VectorClinico:
    defaults = {
        "edad": 45,
        "sexo": "M",
        "sintomas_presentes": ["checkup"],
        "patologias_previas": [],
        "medicacion_habitual": [],
        "presion_sistolica": 120,
        "presion_diastolica": 80,
        "frecuencia_cardiaca": 80,
        "frecuencia_respiratoria": 16,
        "saturacion_oxigeno": 98.0,
        "temperatura": 37.0,
        "nivel_dolor": 0,
        "metodo_llegada": "desconocido",
    }
    defaults.update(kwargs)
    return VectorClinico(**defaults)


def _features(**kwargs):
    return vectorclinico_a_features(_vector(**kwargs)).iloc[0]


def _feature_with_prefix(prefix: str) -> str:
    return next(feature for feature in TODAS_FEATURES if feature.startswith(prefix))


def test_catalogo_de_features_cubre_todo_el_contrato_tabular():
    catalogo = FEATURES_CONTINUAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS

    assert catalogo == TODAS_FEATURES
    assert len(catalogo) == 88
    assert len(set(catalogo)) == len(catalogo)


def test_feature_list_del_modelo_es_subconjunto_del_contrato_tabular():
    feature_list_path = Path("models/feature_list.json")
    payload = json.loads(feature_list_path.read_text(encoding="utf-8"))
    tabulares_modelo = [
        f for f in payload["features"] if not f.startswith("bert_svd_")
    ]

    assert set(tabulares_modelo).issubset(TODAS_FEATURES)
    assert payload["n_features"] == len(payload["features"])


@pytest.mark.parametrize(
    ("metodo", "flag_activa"),
    [
        ("ambulancia", "llegada_ambulancia"),
        ("helicoptero", "llegada_helicoptero"),
        ("autonomo", "llegada_autonoma"),
        ("otro", "llegada_autonoma"),
        ("desconocido", "llegada_desconocida"),
    ],
)
def test_metodo_llegada_activa_una_sola_flag_y_no_cae_en_desconocida(
    metodo, flag_activa
):
    row = _features(metodo_llegada=metodo)
    llegada_flags = [
        "llegada_autonoma",
        "llegada_ambulancia",
        "llegada_helicoptero",
        "llegada_desconocida",
    ]

    assert row[flag_activa] == 1
    assert int(row[llegada_flags].sum()) == 1
    if metodo != "desconocido":
        assert row["llegada_desconocida"] == 0


@pytest.mark.parametrize(
    ("entrada", "normalizado"),
    [
        ("AMBULANCE", "ambulancia"),
        ("helicóptero", "helicoptero"),
        ("WALK IN", "autonomo"),
        ("OTHER", "otro"),
        (None, "desconocido"),
    ],
)
def test_schema_normaliza_valores_posibles_de_llegada(entrada, normalizado):
    vector = _vector(metodo_llegada=entrada)

    assert vector.metodo_llegada == normalizado


@pytest.mark.parametrize(
    ("edad", "anciano", "anciano_mayor"),
    [
        (0, 0, 0),
        (64, 0, 0),
        (65, 1, 0),
        (74, 1, 0),
        (75, 1, 1),
        (120, 1, 1),
    ],
)
def test_edad_cubre_valores_frontera(edad, anciano, anciano_mayor):
    row = _features(edad=edad)

    assert row["age"] == edad
    assert row["anciano"] == anciano
    assert row["anciano_mayor"] == anciano_mayor


@pytest.mark.parametrize(
    ("sexo", "gender"),
    [("M", "M"), ("F", "F"), ("Otro", "M")],
)
def test_sexo_cubre_todos_los_valores_del_schema(sexo, gender):
    row = _features(sexo=sexo)

    assert row["gender"] == gender


def test_constantes_normales_activan_zona_verde_y_excluyen_criticos():
    row = _features()

    assert row["zona_verde"] == 1
    assert row["n_vitales_anomalos"] == 0
    assert row["vitales_criticos"] == 0
    assert row["qsofa_positivo"] == 0
    assert row["news2_alto"] == 0


def test_constantes_faltantes_solo_activan_missingness_esperada():
    row = _features(
        temperatura=None,
        saturacion_oxigeno=None,
        nivel_dolor=None,
    )

    assert row["temperature_missing"] == 1
    assert row["o2sat_missing"] == 1
    assert row["pain_missing"] == 1
    assert row["fiebre"] == 0
    assert row["o2sat_bajo_92"] == 0
    assert row["zona_verde"] == 0


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"temperatura": 38.0}, "fiebre"),
        ({"saturacion_oxigeno": 91.0}, "o2sat_bajo_92"),
        ({"frecuencia_respiratoria": 25}, "taquipnea_grave"),
        ({"presion_sistolica": 89}, "hipotension"),
        ({"presion_sistolica": 180}, "hipertension_severa"),
        ({"frecuencia_cardiaca": 120, "presion_sistolica": 100}, "shock_index_alto"),
        ({"frecuencia_cardiaca": 140, "presion_sistolica": 100}, "shock_index_severo"),
    ],
)
def test_constantes_patologicas_activan_su_flag(kwargs, flag):
    row = _features(**kwargs)

    assert row[flag] == 1


def test_qsofa_positivo_exige_fr_y_pas_en_rango_de_riesgo():
    row = _features(frecuencia_respiratoria=22, presion_sistolica=100)

    assert row["qsofa"] == 2
    assert row["qsofa_positivo"] == 1


def test_qsofa_no_se_activa_con_un_solo_criterio():
    row = _features(frecuencia_respiratoria=22, presion_sistolica=120)

    assert row["qsofa"] == 1
    assert row["qsofa_positivo"] == 0


@pytest.mark.parametrize(
    ("medicacion", "flag"),
    [
        (["anticoagulants - coumarin"], "med_anticoagulante"),
        (["salicylate analgesics"], "med_antiagregante"),
        (["insulin analogs"], "med_insulina"),
        (["biguanides"], "med_antidiabetico_oral"),
        (["glucocorticoids"], "med_corticoide_sistemico"),
        (["analgesic opioid agonists"], "med_opiaceo"),
        (["antianxiety agent - benzodiazepines"], "med_benzodiacepina"),
        (["beta blockers cardiac selective"], "med_betabloqueante"),
        (["ace inhibitors"], "med_ace_ara2"),
        (["diuretic - loop"], "med_diuretico_asa"),
        (["diuretic - thiazides and related"], "med_diuretico_tiazida"),
        (["antipsychotic"], _feature_with_prefix("med_antipsic")),
        (["selective serotonin reuptake inhibitors"], "med_ssri_snri"),
        (["anticonvulsant - gaba analogs"], "med_anticonvulsivante"),
        (
            ["asthma/copd therapy - beta 2-adrenergic agents"],
            "med_respiratorio_inhalado",
        ),
        (["immunosuppressive - calcineurin inhibitors"], "med_inmunosupresor"),
        (["digitalis glycosides"], "med_digoxina"),
        (["antiarrhythmic - class iii"], "med_antiaritmico"),
    ],
)
def test_cada_familia_de_medicacion_activa_su_flag(medicacion, flag):
    row = _features(medicacion_habitual=medicacion)

    assert row[flag] == 1


def test_sin_medicacion_excluye_flags_farmacologicas_y_riesgos():
    row = _features(medicacion_habitual=[])
    med_cols = [c for c in TODAS_FEATURES if c.startswith("med_")]
    riesgo_cols = [
        "alto_riesgo_sangrado",
        "alto_riesgo_delirium",
        "riesgo_depresion_resp",
    ]

    assert row["n_medicamentos"] == 0
    assert row["sin_medicacion"] == 1
    assert row["polifarmacia"] == 0
    assert row["hiperpolifarmacia"] == 0
    assert int(row[med_cols + riesgo_cols].sum()) == 0


def test_polifarmacia_e_hiperpolifarmacia_respetan_umbrales():
    cinco = _features(medicacion_habitual=[f"unknown med {i}" for i in range(5)])
    diez = _features(medicacion_habitual=[f"unknown med {i}" for i in range(10)])

    assert cinco["polifarmacia"] == 1
    assert cinco["hiperpolifarmacia"] == 0
    assert diez["polifarmacia"] == 1
    assert diez["hiperpolifarmacia"] == 1


def test_corticoide_inhalado_no_activa_corticoide_sistemico():
    row = _features(medicacion_habitual=["inhaled corticosteroids"])

    assert row["med_corticoide_sistemico"] == 0


def test_superflags_farmacologicas_respetan_condiciones_compuestas():
    sangrado = _features(medicacion_habitual=["anticoagulants - coumarin"])
    delirium = _features(medicacion_habitual=["antipsychotic"])
    depresion = _features(
        medicacion_habitual=[
            "analgesic opioid agonists",
            "antianxiety agent - benzodiazepines",
        ]
    )

    assert sangrado["alto_riesgo_sangrado"] == 1
    assert delirium["alto_riesgo_delirium"] == 1
    assert depresion["riesgo_depresion_resp"] == 1


@pytest.mark.parametrize(
    ("sintoma", "flag"),
    [
        ("chest pain", "cc_dolor_toracico"),
        ("dyspnea", "cc_disnea"),
        ("abdominal pain", "cc_gi_agudo"),
        ("altered mental status", "cc_neuro_ams"),
        ("headache", "cc_cefalea"),
        ("fall", "cc_trauma"),
        ("suicidal ideation", "cc_psiquiatrico"),
        ("alcohol intoxication", "cc_intoxicacion"),
        ("fever", "cc_infeccioso"),
        ("dysuria", "cc_urologico"),
        ("rectal bleeding", "cc_hemorragia_activa"),
        ("anaphylaxis", "cc_alergia_anafilaxia"),
        ("referral", "cc_derivacion_urgente"),
    ],
)
def test_cada_sindrome_de_chief_complaint_activa_su_flag(sintoma, flag):
    row = _features(sintomas_presentes=[sintoma])

    assert row[flag] == 1


@pytest.mark.parametrize(
    ("antecedente", "flag"),
    [
        ("hipertension arterial", "hx_cardiaco"),
        ("EPOC", "hx_respiratorio"),
        ("ictus previo", "hx_neuro"),
        ("depresion", "hx_psiquiatrico"),
        ("alcoholismo", "hx_abuso_sustancias"),
        ("cirrosis", "hx_digestivo"),
        ("diabetes mellitus", "hx_metabolico_renal"),
        ("VIH", "hx_infeccioso"),
        ("fractura previa", "hx_trauma_muscular"),
    ],
)
def test_cada_macroantecedente_activa_su_hx(antecedente, flag):
    row = _features(patologias_previas=[antecedente])

    assert row[flag] == 1


def test_antecedentes_vacios_no_inventan_hx_ni_frecuentacion():
    row = _features(patologias_previas=[])
    hx_cols = [c for c in TODAS_FEATURES if c.startswith("hx_")]

    assert int(row[hx_cols].sum()) == 0
    assert row["n_visitas_previas"] == 0
    assert row["visitas_ultimo_mes"] == 0
    assert row[_feature_with_prefix("visitas_ultimo_a")] == 0
    assert row["primera_visita"] == 1
    assert row["frecuentador"] == 0
