from triaje_ia.ui.clinical_display import (
    display_clinical_term,
    display_feature_name,
    internalize_review_term,
    join_review_terms_display,
)


def test_display_clinical_term_traduce_sintomas_y_medicacion():
    assert display_clinical_term("chest pain") == "dolor torácico"
    assert display_clinical_term("dyspnea") == "disnea"
    assert display_clinical_term("anticoagulants - coumarin") == "anticoagulantes cumarínicos"


def test_internalize_review_term_recupera_contrato_interno():
    assert internalize_review_term("dolor torácico") == "chest pain"
    assert internalize_review_term("sangrado activo") == "bleeding"
    assert internalize_review_term("anticoagulantes cumarínicos") == "anticoagulants - coumarin"


def test_join_review_terms_display_formatea_en_espanol():
    assert join_review_terms_display(["chest pain", "dyspnea"]) == "dolor torácico, disnea"


def test_display_feature_name_evita_ingles_crudo_en_shap():
    assert display_feature_name("cc_chest_pain") == "Motivo: dolor torácico"
    assert display_feature_name("bert_svd_04") == "Contexto clínico del relato (04)"
    assert display_feature_name("temperature_missing") == "Temperatura no registrada"
    assert display_feature_name("unknown_feature") == "Factor clínico del modelo"
