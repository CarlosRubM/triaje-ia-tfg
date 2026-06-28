from triaje_ia.ui.clinical_form import (
    TriageFormData,
    entrada_minima_completa,
    generar_narrativa_triaje,
    generar_narrativa_triaje_texto_libre,
    validar_entrada_triaje,
)


def test_genera_narrativa_clara_con_caso_completo():
    data = TriageFormData(
        edad=68,
        sexo="M",
        motivo_consulta="dolor toracico opresivo",
        sintomas_frecuentes=["dolor toracico", "disnea"],
        sintomas_adicionales="sudoracion, nauseas",
        signos_alarma=["dolor toracico activo"],
        duracion_sintomas="2 horas",
        presion_sistolica=156,
        presion_diastolica=92,
        frecuencia_cardiaca=118,
        frecuencia_respiratoria=24,
        saturacion_oxigeno=91,
        temperatura=37.8,
        nivel_dolor=8,
        antecedentes="HTA, diabetes",
        medicacion="metformina, enalapril",
        observaciones="palido y diaforetico",
    )

    narrativa = generar_narrativa_triaje(data)

    assert "Paciente de 68 años, sexo M" in narrativa
    assert "Motivo principal de consulta: dolor toracico opresivo." in narrativa
    assert "dolor toracico, disnea, sudoracion, nauseas" in narrativa
    assert "TA 156/92 mmHg" in narrativa
    assert "FC 118 lpm" in narrativa
    assert "FR 24 rpm" in narrativa
    assert "SpO₂ 91%" in narrativa
    assert "temperatura 37.8 °C" in narrativa
    assert "EVA/NRS 8/10" in narrativa
    assert "HTA, diabetes" in narrativa
    assert "metformina, enalapril" in narrativa
    assert "palido y diaforetico" in narrativa


def test_constantes_ausentes_no_fallan_y_generan_aviso():
    data = TriageFormData(
        edad=35,
        sexo="F",
        motivo_consulta="fiebre y dolor abdominal",
        sintomas_frecuentes=["fiebre", "dolor abdominal"],
    )

    narrativa = generar_narrativa_triaje(data)
    avisos = validar_entrada_triaje(data)

    assert "constantes no registradas" in narrativa
    assert any("Constantes no registradas" in aviso for aviso in avisos)
    assert entrada_minima_completa(data)


def test_tension_arterial_incompleta_genera_aviso():
    data = TriageFormData(
        edad=72,
        sexo="F",
        motivo_consulta="mareo",
        presion_sistolica=92,
    )

    narrativa = generar_narrativa_triaje(data)
    avisos = validar_entrada_triaje(data)

    assert "TA sistólica 92 mmHg" in narrativa
    assert any("Tensión arterial incompleta" in aviso for aviso in avisos)


def test_entrada_minima_exige_edad_sexo_y_motivo():
    assert not entrada_minima_completa(TriageFormData())
    assert not entrada_minima_completa(
        TriageFormData(edad=45, sexo="M", motivo_consulta=" ")
    )
    assert entrada_minima_completa(
        TriageFormData(edad=45, sexo="M", motivo_consulta="traumatismo")
    )


def test_narrativa_texto_libre_preserva_relato_clinico():
    data = TriageFormData(
        edad=79,
        sexo="F",
        motivo_consulta="caida",
        sintomas_adicionales=(
            "Caida no presenciada en domicilio, somnolienta desde entonces, "
            "niega dolor toracico y fiebre."
        ),
        duracion_sintomas="esta manana",
        presion_sistolica=94,
        presion_diastolica=58,
        frecuencia_cardiaca=112,
        saturacion_oxigeno=93,
        nivel_dolor=4,
        antecedentes="fibrilacion auricular, hipertension",
        medicacion="Sintrom, bisoprolol",
        observaciones="acude con su hija",
    )

    narrativa = generar_narrativa_triaje_texto_libre(data)

    assert "Narrativa estructurada de triaje:" in narrativa
    assert "Relato clínico de triaje: Caida no presenciada en domicilio, somnolienta desde entonces, niega dolor toracico y fiebre." in narrativa
    assert "Antecedentes médicos relevantes: fibrilacion auricular, hipertension." in narrativa
    assert "Fármacos habituales referidos: Sintrom, bisoprolol." in narrativa
    assert "TA 94/58 mmHg" in narrativa
    assert "EVA/NRS 4/10" in narrativa


def test_narrativa_texto_libre_conserva_negaciones_y_no_obliga_farmacos():
    data = TriageFormData(
        edad=34,
        sexo="M",
        motivo_consulta="fiebre",
        sintomas_adicionales="Fiebre desde ayer, tos seca. Niega disnea.",
        medicacion="niega medicacion",
    )

    narrativa = generar_narrativa_triaje_texto_libre(data)

    assert "Fiebre desde ayer, tos seca. Niega disnea." in narrativa
    assert "Fármacos habituales referidos: niega medicacion." in narrativa
    assert "Dolor: escala EVA/NRS no registrada." in narrativa


def test_narrativa_texto_libre_incluye_llegada_y_discriminadores():
    data = TriageFormData(
        edad=74,
        sexo="F",
        metodo_llegada="ambulancia",
        motivo_consulta="caida",
        sintomas_frecuentes=[
            "bajo nivel de conciencia",
            "sangrado activo",
            "trauma mayor",
        ],
        sintomas_adicionales=(
            "Caida no presenciada. Somnolienta desde entonces. "
            "Niega dolor toracico."
        ),
        duracion_sintomas="esta manana",
        presion_sistolica=94,
        presion_diastolica=58,
        frecuencia_cardiaca=112,
        saturacion_oxigeno=93,
        nivel_dolor=4,
        antecedentes="fibrilacion auricular",
        medicacion="Sintrom",
    )

    narrativa = generar_narrativa_triaje_texto_libre(data)

    assert "Método de llegada: ambulancia." in narrativa
    assert "Discriminadores de prioridad observados: bajo nivel de conciencia, sangrado activo, trauma mayor." in narrativa
    assert "Niega dolor toracico." in narrativa
    assert "TA 94/58 mmHg" in narrativa
    assert "Fármacos habituales referidos: Sintrom." in narrativa
