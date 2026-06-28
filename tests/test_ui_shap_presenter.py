from triaje_ia.ui.shap_presenter import clean_feature_name, split_shap_factors


def test_clean_feature_name_usa_nombres_clinicos():
    assert clean_feature_name("resprate") == "Frecuencia respiratoria"
    assert clean_feature_name("cc_chest_pain") == "Motivo: dolor torácico"
    assert clean_feature_name("bert_svd_04") == "Contexto clínico del relato (04)"
    assert clean_feature_name("temperature_missing") == "Temperatura no registrada"
    assert clean_feature_name("unknown_feature") == "Factor clínico del modelo"


def test_split_shap_factors_separa_signo_y_ordena_por_magnitud():
    positivos, negativos = split_shap_factors(
        ["age", "o2sat", "resprate", "temp"],
        [0.02, -0.31, 0.15, -0.04],
        top_n=2,
    )

    assert [factor.nombre for factor in positivos] == [
        "Frecuencia respiratoria",
        "Edad",
    ]
    assert [factor.nombre for factor in negativos] == [
        "Saturación de oxígeno",
        "Temperatura",
    ]
    assert positivos[0].valor == 0.15
    assert negativos[0].valor == -0.31
