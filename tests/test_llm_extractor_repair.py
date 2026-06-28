from triaje_ia.llm.extractor import _completar_datos_explicitos
from triaje_ia.llm.schemas import VectorClinico


def _vector(**kwargs) -> VectorClinico:
    defaults = {
        "edad": 55,
        "sexo": "M",
        "sintomas_presentes": [],
    }
    defaults.update(kwargs)
    return VectorClinico(**defaults)


def test_completa_constantes_explicitas_si_el_llm_las_omite():
    vector = _vector()

    reparado = _completar_datos_explicitos(
        "TA 130/76, FC 74, FR 16, SatO2 98%, T 36.5, Glasgow 15",
        vector,
    )

    assert reparado.presion_sistolica == 130
    assert reparado.presion_diastolica == 76
    assert reparado.frecuencia_cardiaca == 74
    assert reparado.frecuencia_respiratoria == 16
    assert reparado.saturacion_oxigeno == 98.0
    assert reparado.temperatura == 36.5


def test_completa_motivo_administrativo_si_no_hay_sintomas():
    vector = _vector()

    reparado = _completar_datos_explicitos(
        "Varon solicita receta de medicacion habitual porque se le acabo.",
        vector,
    )

    assert reparado.sintomas_presentes == ["medication refill"]


def test_completa_llegada_en_ambulancia_si_el_llm_la_omite():
    vector = _vector(metodo_llegada="desconocido")

    reparado = _completar_datos_explicitos(
        "Mujer trasladada en ambulancia por disnea.",
        vector,
    )

    assert reparado.metodo_llegada == "ambulancia"


def test_completa_llegada_en_helicoptero_con_prioridad_sobre_ambulancia():
    vector = _vector(metodo_llegada="desconocido")

    reparado = _completar_datos_explicitos(
        "Traslado en helicoptero medicalizado. En helipuerto espera ambulancia.",
        vector,
    )

    assert reparado.metodo_llegada == "helicoptero"


def test_no_sobrescribe_metodo_llegada_ya_extraido():
    vector = _vector(metodo_llegada="ambulancia")

    reparado = _completar_datos_explicitos(
        "Acude por sus medios tras aviso telefonico.",
        vector,
    )

    assert reparado.metodo_llegada == "ambulancia"


def test_completa_llegada_autonoma_por_sus_medios():
    vector = _vector(metodo_llegada="desconocido")

    reparado = _completar_datos_explicitos(
        "Paciente acude por sus medios por dolor abdominal.",
        vector,
    )

    assert reparado.metodo_llegada == "autonomo"


def test_completa_trauma_y_sangrado_explicitos():
    vector = _vector(sintomas_presentes=["abdominal pain"])

    reparado = _completar_datos_explicitos(
        "Accidente de moto con dolor abdominal y herida sangrante en muslo.",
        vector,
    )

    assert "trauma" in reparado.sintomas_presentes
    assert "bleeding" in reparado.sintomas_presentes


def test_completa_trauma_craneal_y_herida_cuero_cabelludo_literales():
    vector = _vector(sintomas_presentes=["trauma"])

    reparado = _completar_datos_explicitos(
        "Traumatismo craneal con herida en cuero cabelludo tras caída.",
        vector,
    )

    assert "head injury" in reparado.sintomas_presentes
    assert "scalp wound" in reparado.sintomas_presentes


def test_completa_fractura_abierta_literal():
    vector = _vector(sintomas_presentes=["trauma"])

    reparado = _completar_datos_explicitos(
        "Paciente con fractura abierta de tibia tras accidente.",
        vector,
    )

    assert "open fracture" in reparado.sintomas_presentes


def test_completa_bradipnea_literal():
    vector = _vector(sintomas_presentes=["altered mental status"])

    reparado = _completar_datos_explicitos(
        "Paciente somnoliento con bradipnea marcada en triaje.",
        vector,
    )

    assert "bradypnea" in reparado.sintomas_presentes


def test_completa_somnolencia_como_ams_explicito():
    vector = _vector(sintomas_presentes=[])

    reparado = _completar_datos_explicitos(
        "Paciente con somnolencia, responde solo al dolor.",
        vector,
    )

    assert "altered mental status" in reparado.sintomas_presentes


def test_completa_somnolienta_y_discriminador_ams_guiado():
    vector = _vector(sintomas_presentes=["trauma", "bleeding"])

    reparado = _completar_datos_explicitos(
        (
            "Discriminadores de prioridad observados: alteracion del nivel "
            "de consciencia, sangrado activo, trauma mayor. "
            "Relato clinico: Somnolienta desde entonces."
        ),
        vector,
    )

    assert "altered mental status" in reparado.sintomas_presentes


def test_completa_hipoglucemia_si_glucemia_baja_es_lititeral():
    vector = _vector(sintomas_presentes=["dizziness"])

    reparado = _completar_datos_explicitos(
        "Mareo y sudoracion. Glucemia 42 mg/dl en triaje.",
        vector,
    )

    assert "hypoglycemia" in reparado.sintomas_presentes
