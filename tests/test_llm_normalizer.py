from triaje_ia.llm.normalizer import normalizar_vector_clinico
from triaje_ia.llm.schemas import VectorClinico


def _vector(**kwargs) -> VectorClinico:
    defaults = {
        "edad": 60,
        "sexo": "M",
        "sintomas_presentes": [],
    }
    defaults.update(kwargs)
    return VectorClinico(**defaults)


def test_normaliza_sintomas_frecuentes_y_elimina_duplicados():
    vector = _vector(
        sintomas_presentes=[
            " Shortness of breath ",
            "dyspnea",
            "Chest discomfort",
        ]
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.sintomas_presentes == ["dyspnea", "chest pain"]


def test_no_deja_sintoma_vago_si_hay_sintomas_mas_informativos():
    vector = _vector(
        sintomas_presentes=["feels unwell", "abdominal pain", "malaise"]
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.sintomas_presentes == ["abdominal pain"]


def test_normaliza_medicacion_a_clase_terapeutica_basica():
    vector = _vector(medicacion_habitual=["Metformina", "Adiro", "biguanides"])

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.medicacion_habitual == ["biguanides", "salicylate analgesics"]


def test_normaliza_medicacion_a_patrones_usados_por_features():
    vector = _vector(
        medicacion_habitual=[
            "Sintrom",
            "Furosemida",
            "Insulina",
            "Clopidogrel",
        ]
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.medicacion_habitual == [
        "anticoagulants - coumarin",
        "diuretic - loop",
        "insulin analogs",
        "thienopyridine",
    ]
