import pytest
from pydantic import ValidationError

from triaje_ia.ui.vector_review import (
    construir_vector_desde_revision,
    split_review_terms,
)


def test_split_review_terms_deduplica_y_respeta_orden():
    assert split_review_terms("disnea, fiebre; disnea\n tos ") == [
        "disnea",
        "fiebre",
        "tos",
    ]


def test_construir_vector_desde_revision_normaliza_listas_y_medicacion():
    vector = construir_vector_desde_revision(
        edad=68,
        sexo="M",
        metodo_llegada="ambulancia",
        sintomas_presentes="dolor toracico, disnea",
        patologias_previas="HTA, diabetes",
        medicacion_habitual="metformina, Sintrom",
        presion_sistolica=156,
        presion_diastolica=92,
        frecuencia_cardiaca=118,
        frecuencia_respiratoria=24,
        saturacion_oxigeno=91,
        temperatura=37.8,
        nivel_dolor=8,
        duracion_sintomas="2 horas",
    )

    assert vector.metodo_llegada == "ambulancia"
    assert vector.sintomas_presentes == ["chest pain", "dyspnea"]
    assert vector.patologias_previas == ["hta", "diabetes"]
    assert "biguanides" in vector.medicacion_habitual
    assert "anticoagulants - coumarin" in vector.medicacion_habitual


def test_construir_vector_desde_revision_acepta_terminos_visibles_en_espanol():
    vector = construir_vector_desde_revision(
        edad=74,
        sexo="F",
        sintomas_presentes="dolor torácico, disnea, sangrado activo",
        medicacion_habitual="anticoagulantes cumarínicos",
    )

    assert vector.sintomas_presentes == ["chest pain", "dyspnea", "bleeding"]
    assert vector.medicacion_habitual == ["anticoagulants - coumarin"]


def test_construir_vector_desde_revision_bloquea_valores_fuera_de_rango():
    with pytest.raises(ValidationError):
        construir_vector_desde_revision(
            edad=130,
            sexo="M",
            sintomas_presentes="fiebre",
        )
