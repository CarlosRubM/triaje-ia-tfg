from triaje_ia.ui.shap_presenter import (
    clean_feature_name,
    group_shap_factors,
    select_balanced_shap_factors,
    split_shap_factors,
)


def test_clean_feature_name_usa_nombres_clinicos():
    assert clean_feature_name("resprate") == "Frecuencia respiratoria"
    assert clean_feature_name("cc_dolor_toracico") == "Motivo: dolor torácico"
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


def test_group_shap_factors_agrupa_bert_y_nombres_duplicados():
    factors = group_shap_factors(
        ["bert_svd_01", "bert_svd_02", "heartrate", "heartrate"],
        [0.4, -0.1, 0.2, 0.3],
    )

    values = {factor.nombre: factor.valor for factor in factors}
    assert values["Información del relato clínico"] == 0.30000000000000004
    assert values["Frecuencia cardiaca"] == 0.5


def test_group_shap_factors_devuelve_seis_por_magnitud_absoluta():
    factors = group_shap_factors(
        ["age", "o2sat", "heartrate", "resprate", "sbp", "dbp", "temp", "news2"],
        [0.1, -0.8, 0.2, 0.7, -0.6, 0.5, 0.4, 0.3],
    )

    assert len(factors) == 6
    assert [abs(factor.valor) for factor in factors] == sorted(
        (abs(factor.valor) for factor in factors), reverse=True
    )


def test_select_balanced_shap_factors_devuelve_tres_por_direccion():
    factors = select_balanced_shap_factors(
        ["age", "o2sat", "heartrate", "resprate", "sbp", "dbp", "temp", "news2"],
        [0.9, 0.8, 0.7, 0.1, -0.9, -0.8, -0.7, -0.1],
    )

    assert len([factor for factor in factors if factor.valor > 0]) == 3
    assert len([factor for factor in factors if factor.valor < 0]) == 3
