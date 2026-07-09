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


def test_normaliza_motivos_leves_y_administrativos():
    vector = _vector(
        sintomas_presentes=[
            "Prescription refill",
            "Wound review",
            "Dry eyes",
        ]
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.sintomas_presentes == [
        "medication refill",
        "wound check",
        "dry eye",
    ]


def test_normaliza_sinonimos_observados_en_auditoria_llm():
    vector = _vector(
        sintomas_presentes=["somnolencia", "diaphoresis"],
        medicacion_habitual=[
            "opioid analgesics",
            "benzodiazepines",
            "bronchodilators - inhalation",
            "steroids - inhalation",
        ],
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.sintomas_presentes == ["altered mental status", "sweating"]
    assert normalizado.medicacion_habitual == [
        "analgesic opioid agonists",
        "antianxiety agent - benzodiazepines",
        "asthma/copd therapy - beta 2-adrenergic agents",
        "asthma therapy - inhaled corticosteroids",
    ]


def test_normaliza_sinonimos_clinicos_espanoles_de_auditoria():
    vector = _vector(
        sintomas_presentes=[
            "edema labios y lengua",
            "estridor",
            "convulsiones",
            "cefalea trueno",
            "rigidez nuca",
            "fotofobia",
            "sudoración",
            "cianosis",
            "palidez",
            "vértigo",
            "hematochezia",
            "productive cough",
        ],
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.sintomas_presentes == [
        "angioedema",
        "stridor",
        "seizure",
        "thunderclap headache",
        "neck stiffness",
        "photophobia",
        "sweating",
        "cyanosis",
        "pallor",
        "vertigo",
        "rectal bleeding",
        "cough",
    ]


def test_normaliza_sinonimos_leves_y_sensoriales_de_auditoria():
    vector = _vector(
        sintomas_presentes=[
            "pérdida visual",
            "loss of vision",
            "oído taponado",
            "decreased hearing",
            "herida simple",
            "corte superficial",
            "quemadura pequeña",
            "ampolla",
        ],
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.sintomas_presentes == [
        "vision loss",
        "hearing loss",
        "minor wound",
        "minor burn",
        "blister",
    ]


def test_normaliza_terminos_visibles_de_la_ui_en_espanol():
    vector = _vector(
        sintomas_presentes=[
            "dolor torácico",
            "alteración del nivel de conciencia",
            "sangrado activo",
        ],
        medicacion_habitual=[
            "anticoagulantes cumarínicos",
            "betabloqueantes cardioselectivos",
        ],
    )

    normalizado = normalizar_vector_clinico(vector)

    assert normalizado.sintomas_presentes == [
        "chest pain",
        "altered mental status",
        "bleeding",
    ]
    assert normalizado.medicacion_habitual == [
        "anticoagulants - coumarin",
        "beta blockers cardiac selective",
    ]
