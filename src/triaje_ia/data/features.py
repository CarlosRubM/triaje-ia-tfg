"""
src/triaje_ia/data/features.py

Ingeniería de features sobre el dataset base producido por loader.py + cleaner.py.
Produce un DataFrame con las 88 features definitivas listas para entrenar.

Arquitectura de bloques:

  BLOQUE 1 — Vitales, scores y banderas clínicas
    Missingness NMAR:     temperature_missing, o2sat_missing, pain_missing
    Vitales crudos:       pain, o2sat, resprate, heartrate, dbp, sbp
    Banderas clínicas:    o2sat_bajo_92, taquipnea_grave, fiebre, hipotension,
                          hipertension_severa
    Scores compuestos:    qsofa, qsofa_positivo, news2, news2_alto,
                          shock_index_alto, shock_index_severo, shock_oculto_anciano,
                          vitales_criticos, n_vitales_anomalos, sirs_positivo,
                          zona_verde, pulse_pressure
    DESCARTADAS:          temperature (sustituida por fiebre), taquicardia
                          (contenida en n_vitales_anomalos, r=0.925)

  BLOQUE 2 — Demográfico y logístico
    Edad:                 age (continuo), anciano (≥65), anciano_mayor (≥75)
    Identidad:            identidad_desconocida (race=UNKNOWN)
    Transporte:           llegada_autonoma, llegada_ambulancia,
                          llegada_helicoptero, llegada_desconocida
    DESCARTADAS:          race (equidad algorítmica), variables temporales
                          (hipótesis weekend effect refutada)

  BLOQUE 3 — Medicación (clases ATC)
    17 flags individuales: med_anticoagulante, med_antiagregante, med_insulina,
                           med_antidiabetico_oral, med_corticoide_sistemico,
                           med_opiaceo, med_benzodiacepina, med_betabloqueante,
                           med_ace_ara2, med_diuretico_asa, med_diuretico_tiazida,
                           med_antipsicótico, med_ssri_snri, med_anticonvulsivante,
                           med_respiratorio_inhalado, med_inmunosupresor,
                           med_digoxina, med_antiaritmico
    3 super-flags:         alto_riesgo_sangrado, alto_riesgo_delirium,
                           riesgo_depresion_resp
    DESCARTADAS:           med_aine, med_diuretico_tiazida (solo componentes),
                           triple_whammy (H=55, sin masa estadística)

  BLOQUE 4 — Polifarmacia
    n_medicamentos (continuo), sin_medicacion, polifarmacia (≥5),
    hiperpolifarmacia (≥10)

  BLOQUE 5 — Chief complaint (13 síndromes clínicos)
    cc_dolor_toracico, cc_disnea, cc_gi_agudo, cc_neuro_ams, cc_cefalea,
    cc_trauma, cc_psiquiatrico, cc_intoxicacion, cc_infeccioso, cc_urologico,
    cc_hemorragia_activa, cc_alergia_anafilaxia, cc_derivacion_urgente
    Regex auditadas con experimento de alineación LLM (F1 ≥ 0.67 tras revisión)
    DESCARTADA: cc_missing (prevalencia 0.30%)

  BLOQUE 6 — Historial ED y comorbilidades CCS
    Frecuentación:        n_visitas_previas, visitas_ultimo_mes, visitas_ultimo_año,
                          dias_desde_ultima_visita, primera_visita, frecuentador
    Comorbilidades CCS:   hx_cardiaco, hx_respiratorio, hx_neuro, hx_psiquiatrico,
                          hx_abuso_sustancias, hx_digestivo, hx_metabolico_renal,
                          hx_infeccioso, hx_trauma_muscular
    Regla lookback:       outtime_previa < intime_actual (evita solapamientos)

  TOTAL: 88 features + 1 target (acuity)

Notas de implementación:
  - NaN nativos en vitales: XGBoost/LightGBM/RF los manejan sin imputación
  - Temperatura en °F en MIMIC — umbral fiebre ≥100.4°F (38.0°C)
  - NEWS2 vectorizado con np.select (validado, H idéntico al iterativo)
  - Bloque 6 vectorizado con isin() + cummax()
"""

import numpy as np
import pandas as pd
from loguru import logger

from triaje_ia.config import DATA_INTERIM, DATA_PROCESSED

# Constantes clínicas

# Umbrales vitales (°F para temperatura, unidades estándar para el resto)
UMBRAL_FIEBRE_F        = 100.4   # ≥ 100.4°F = 38.0°C
UMBRAL_TAQUICARDIA     = 100
UMBRAL_TAQUIPNEA_GRAVE = 24
UMBRAL_O2SAT_BAJO      = 92
UMBRAL_HIPOTENSION     = 90
UMBRAL_HTA_SEVERA      = 180
UMBRAL_SHOCK_ALTO      = 1.0
UMBRAL_SHOCK_SEVERO    = 1.4
UMBRAL_SHOCK_ANCIANO   = 0.85
UMBRAL_ANCIANO         = 65
UMBRAL_ANCIANO_MAYOR   = 75
UMBRAL_NEWS2_ALTO      = 7
UMBRAL_VITALES_CRITICOS = 2
UMBRAL_FRECUENTADOR    = 3       # visitas en 30 días

# Macro-flags CCS — códigos reales disponibles en MIMIC-IV-ED (83 categorías)
CCS_MACROFLAGS: dict[str, list[int]] = {
    "hx_cardiaco":         [98, 100, 101, 103, 106, 108, 117, 118],
    "hx_respiratorio":     [122, 123, 126, 127, 128, 130, 133, 134],
    "hx_neuro":            [83, 93, 95, 109],
    "hx_psiquiatrico":     [651, 657, 659, 662],
    "hx_abuso_sustancias": [660, 661],
    "hx_digestivo":        [142, 145, 146, 147, 149, 151, 152, 153, 154, 155, 250, 251],
    "hx_metabolico_renal": [50, 55, 59, 157, 160],
    "hx_infeccioso":       [2, 7, 197],
    "hx_trauma_muscular":  [204, 205, 228, 229, 230, 231, 232, 233, 235, 236, 238, 239, 244],
}

# Lista definitiva de features para 04_models.ipynb
FEATURES_CONTINUAS: list[str] = [
    "pain", "o2sat", "resprate", "heartrate", "dbp", "sbp",
    "qsofa", "news2", "n_vitales_anomalos", "pulse_pressure",
    "age", "n_medicamentos",
    "n_visitas_previas", "visitas_ultimo_mes",
    "visitas_ultimo_año", "dias_desde_ultima_visita",
]

FEATURES_BINARIAS: list[str] = [
    # missingness vitales
    "temperature_missing", "o2sat_missing", "pain_missing",
    # banderas clínicas
    "o2sat_bajo_92", "taquipnea_grave", "fiebre",
    "hipotension", "hipertension_severa",
    # scores binarios
    "qsofa_positivo", "news2_alto", "shock_index_alto",
    "shock_index_severo", "shock_oculto_anciano",
    "vitales_criticos", "sirs_positivo", "zona_verde",
    # demográfico
    "anciano", "anciano_mayor", "identidad_desconocida",
    # transporte
    "llegada_autonoma", "llegada_ambulancia",
    "llegada_helicoptero", "llegada_desconocida",
    # medicación individual
    "med_anticoagulante", "med_antiagregante", "med_insulina",
    "med_antidiabetico_oral", "med_corticoide_sistemico",
    "med_opiaceo", "med_benzodiacepina", "med_betabloqueante",
    "med_ace_ara2", "med_diuretico_asa", "med_diuretico_tiazida",
    "med_antipsicótico", "med_ssri_snri", "med_anticonvulsivante",
    "med_respiratorio_inhalado", "med_inmunosupresor",
    "med_digoxina", "med_antiaritmico",
    # super-flags
    "alto_riesgo_sangrado", "alto_riesgo_delirium", "riesgo_depresion_resp",
    # polifarmacia
    "sin_medicacion", "polifarmacia", "hiperpolifarmacia",
    # chief complaint
    "cc_dolor_toracico", "cc_disnea", "cc_gi_agudo", "cc_neuro_ams",
    "cc_cefalea", "cc_trauma", "cc_psiquiatrico", "cc_intoxicacion",
    "cc_infeccioso", "cc_urologico", "cc_hemorragia_activa",
    "cc_alergia_anafilaxia", "cc_derivacion_urgente",
    # frecuentación
    "primera_visita", "frecuentador",
    # comorbilidades CCS
    "hx_cardiaco", "hx_respiratorio", "hx_neuro", "hx_psiquiatrico",
    "hx_abuso_sustancias", "hx_digestivo", "hx_metabolico_renal",
    "hx_infeccioso", "hx_trauma_muscular",
]

FEATURES_CATEGORICAS: list[str] = ["gender"]

TODAS_FEATURES: list[str] = (
    FEATURES_CONTINUAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS
)

TARGET = "acuity"

# BLOQUE 1 — Vitales, scores y banderas clínicas

def _calcular_missingness_vitales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Banderas de missingness NMAR para los tres vitales con mayor poder
    discriminativo por ausencia. Patrón en U confirmado: alta ausencia
    en acuity 1 (crítico directo a box) y acuity 5 (leve sin tomar).
    """
    df = df.copy()
    df["temperature_missing"] = df["temperature"].isna().astype("int8")
    df["o2sat_missing"]       = df["o2sat"].isna().astype("int8")
    df["pain_missing"]        = df["pain"].isna().astype("int8")
    logger.info("Missingness vitales calculado")
    return df

def _calcular_banderas_vitales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Banderas clínicas binarias sobre vitales crudos.
    Temperatura en °F — umbrales en °F.
    taquicardia DESCARTADA: contenida en n_vitales_anomalos (r=0.925).
    """
    df = df.copy()

    df["o2sat_bajo_92"]     = (df["o2sat"] < UMBRAL_O2SAT_BAJO).astype("int8")
    df["taquipnea_grave"]   = (df["resprate"] > UMBRAL_TAQUIPNEA_GRAVE).astype("int8")
    df["fiebre"]            = (df["temperature"] >= UMBRAL_FIEBRE_F).astype("int8")
    df["hipotension"]       = (df["sbp"] < UMBRAL_HIPOTENSION).astype("int8")
    df["hipertension_severa"] = (df["sbp"] >= UMBRAL_HTA_SEVERA).astype("int8")

    # taquicardia se calcula internamente para usar en scores compuestos
    # pero NO se incluye en FEATURES_BINARIAS (correlación r=0.925 con n_vitales_anomalos)
    df["_taquicardia_aux"]  = (df["heartrate"] > UMBRAL_TAQUICARDIA).astype("int8")

    logger.info("Banderas vitales calculadas")
    return df

def _calcular_scores_compuestos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores clínicos compuestos vectorizados.
    NEWS2 implementado con np.select (validado, H idéntico al iterativo).
    qSOFA usa pain_missing como proxy de alteración del nivel de consciencia.
    """
    df = df.copy()

    rr  = df["resprate"].values
    o2  = df["o2sat"].values
    sbp = df["sbp"].values
    hr  = df["heartrate"].values
    tmp = df["temperature"].values  # °F

    # NEWS2 vectorizado
    score = np.zeros(len(df))

    score += np.select(
        [np.isnan(rr), rr <= 8, rr <= 11, rr <= 20, rr <= 24],
        [0, 3, 1, 0, 2], default=3
    )
    score += np.select(
        [np.isnan(o2), o2 <= 91, o2 <= 93, o2 <= 95],
        [0, 3, 2, 1], default=0
    )
    score += np.select(
        [np.isnan(sbp), sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219],
        [0, 3, 2, 1, 0], default=3
    )
    score += np.select(
        [np.isnan(hr), hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130],
        [0, 3, 1, 0, 1, 2], default=3
    )
    # temperatura en °F
    score += np.select(
        [np.isnan(tmp), tmp <= 95.0, tmp <= 96.8, tmp <= 100.4, tmp <= 102.2],
        [0, 3, 1, 0, 1], default=2
    )
    score += np.where(df["pain_missing"].values == 1, 3, 0)

    df["news2"]      = score
    df["news2_alto"] = (score >= UMBRAL_NEWS2_ALTO).astype("int8")

    # qSOFA vectorizado
    qsofa = np.zeros(len(df))
    qsofa += np.where(np.nan_to_num(rr) >= 22, 1, 0)
    qsofa += np.where(np.nan_to_num(sbp) <= 100, 1, 0)
    qsofa += df["pain_missing"].values  # proxy AMS

    df["qsofa"]          = qsofa
    df["qsofa_positivo"] = (qsofa >= 2).astype("int8")

    # Shock index
    si = np.where(
        (np.isnan(hr) | np.isnan(sbp) | (sbp == 0)),
        np.nan,
        hr / sbp
    )
    df["shock_index_alto"]   = (np.nan_to_num(si, nan=0) >= UMBRAL_SHOCK_ALTO).astype("int8")
    df["shock_index_severo"] = (np.nan_to_num(si, nan=0) >= UMBRAL_SHOCK_SEVERO).astype("int8")
    df["shock_oculto_anciano"] = (
        (df["age"].fillna(0) >= UMBRAL_ANCIANO) &
        (np.nan_to_num(si, nan=0) >= UMBRAL_SHOCK_ANCIANO)
    ).astype("int8")

    # Pulse pressure
    df["pulse_pressure"] = df["sbp"] - df["dbp"]

    # Vitales críticos (≥2 de 5 banderas)
    df["vitales_criticos"] = (
        df["_taquicardia_aux"].fillna(0) +
        df["taquipnea_grave"].fillna(0) +
        df["o2sat_bajo_92"].fillna(0) +
        df["fiebre"].fillna(0) +
        df["shock_index_alto"].fillna(0)
    ).astype("int8")
    df["vitales_criticos"] = (df["vitales_criticos"] >= UMBRAL_VITALES_CRITICOS).astype("int8")

    # n_vitales_anomalos (conteo continuo)
    df["n_vitales_anomalos"] = (
        df["_taquicardia_aux"].fillna(0) +
        df["taquipnea_grave"].fillna(0) +
        df["o2sat_bajo_92"].fillna(0) +
        df["fiebre"].fillna(0) +
        df["shock_index_alto"].fillna(0)
    ).astype("int8")

    # SIRS positivo (≥2 criterios)
    sirs_taquicardia  = df["_taquicardia_aux"].fillna(0)
    sirs_taquipnea    = (np.nan_to_num(rr) > 20).astype(int)
    sirs_temperatura  = (
        (np.nan_to_num(tmp, nan=98.6) > 100.4) |
        (np.nan_to_num(tmp, nan=98.6) < 96.8)
    ).astype(int)
    sirs_score = sirs_taquicardia + sirs_taquipnea + sirs_temperatura
    df["sirs_positivo"] = (sirs_score >= 2).astype("int8")

    # Zona verde (todos los vitales normales)
    df["zona_verde"] = (
        (np.nan_to_num(hr, nan=999) >= 60)  & (np.nan_to_num(hr, nan=999) <= 100) &
        (np.nan_to_num(rr, nan=999) >= 12)  & (np.nan_to_num(rr, nan=999) <= 20)  &
        (np.nan_to_num(o2, nan=0)   >= 95)  &
        (np.nan_to_num(sbp, nan=999) >= 100) & (np.nan_to_num(sbp, nan=999) <= 140) &
        (np.nan_to_num(tmp, nan=0)  >= 96.8) & (np.nan_to_num(tmp, nan=0)  <= 99.5) &
        (df["pain_missing"] == 0)
    ).astype("int8")

    logger.info("Scores compuestos calculados")
    return df

def calcular_features_bloque1(df: pd.DataFrame) -> pd.DataFrame:
    """Orquestador Bloque 1: missingness + banderas + scores."""
    df = _calcular_missingness_vitales(df)
    df = _calcular_banderas_vitales(df)
    df = _calcular_scores_compuestos(df)
    # eliminar columna auxiliar interna
    df = df.drop(columns=["_taquicardia_aux"], errors="ignore")
    logger.info("Bloque 1 completado: missingness + banderas + scores")
    return df

# BLOQUE 2 — Demográfico y logístico

def calcular_features_bloque2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features demográficas y logísticas.
    race excluida por equidad algorítmica — solo identidad_desconocida
    como proxy de consciencia alterada.
    Variables temporales descartadas (hipótesis weekend effect refutada).
    """
    df = df.copy()

    # Edad
    df["anciano"]       = (df["age"].fillna(0) >= UMBRAL_ANCIANO).astype("int8")
    df["anciano_mayor"] = (df["age"].fillna(0) >= UMBRAL_ANCIANO_MAYOR).astype("int8")

    # Identidad desconocida (proxy AMS)
    df["identidad_desconocida"] = (
        df["race"].astype(str).str.upper().str.strip() == "UNKNOWN"
    ).astype("int8")

    # Transporte
    transport = df["arrival_transport"].astype(str).str.upper().str.strip()
    df["llegada_autonoma"]    = transport.isin(["WALK IN", "OTHER"]).astype("int8")
    df["llegada_ambulancia"]  = (transport == "AMBULANCE").astype("int8")
    df["llegada_helicoptero"] = (transport == "HELICOPTER").astype("int8")
    df["llegada_desconocida"] = (transport == "UNKNOWN").astype("int8")

    logger.info("Bloque 2 completado: demográfico y logístico")
    return df

# BLOQUE 3 — Medicación (clases ATC)

def calcular_features_bloque3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags de medicación activa desde medicacion_raw (etcdescription de medrecon).
    17 clases ATC individuales + 3 super-flags de riesgo.
    Términos extraídos del vocabulario real de MIMIC-IV-ED (1,201 términos únicos).

    DESCARTADAS:
    - med_aine (H=602): solo como componente de super-flags
    - med_diuretico_tiazida (H=248): distribución plana, solo Triple Whammy
    - triple_whammy (H=55): sin masa estadística suficiente
    """
    df   = df.copy()
    med  = df["medicacion_raw"].fillna("").str.lower()

    # Clases individuales
    df["med_anticoagulante"] = (
        med.str.contains("anticoagulants - coumarin",         case=False, na=False) |
        med.str.contains("low molecular weight heparins",      case=False, na=False) |
        med.str.contains("direct factor xa inhibitors",        case=False, na=False)
    ).astype("int8")

    df["med_antiagregante"] = (
        med.str.contains("salicylate analgesics",              case=False, na=False) |
        med.str.contains("thienopyridine",                     case=False, na=False)
    ).astype("int8")

    df["med_insulina"] = (
        med.str.contains("insulin analogs",                    case=False, na=False) |
        med.str.contains("human insulins",                     case=False, na=False)
    ).astype("int8")

    df["med_antidiabetico_oral"] = (
        med.str.contains("biguanides",                         case=False, na=False) |
        med.str.contains("sulfonylurea",                       case=False, na=False) |
        med.str.contains("dipeptidyl peptidase-4",             case=False, na=False)
    ).astype("int8")

    df["med_corticoide_sistemico"] = (
        med.str.contains("glucocorticoids",                    case=False, na=False) &
        ~med.str.contains("nasal corticosteroids",             case=False, na=False) &
        ~med.str.contains("dermatological - glucocorticoid",   case=False, na=False) &
        ~med.str.contains("inhaled corticosteroids",           case=False, na=False)
    ).astype("int8")

    df["med_opiaceo"] = (
        med.str.contains("analgesic opioid agonists",          case=False, na=False) |
        med.str.contains("analgesic opioid oxycodone",         case=False, na=False) |
        med.str.contains("analgesic opioid hydrocodone",       case=False, na=False) |
        med.str.contains("analgesic opioid codeine",           case=False, na=False)
    ).astype("int8")

    df["med_benzodiacepina"] = (
        med.str.contains("antianxiety agent - benzodiazepines", case=False, na=False) |
        med.str.contains("sedative-hypnotic - gaba",           case=False, na=False)
    ).astype("int8")

    df["med_betabloqueante"] = (
        med.str.contains("beta blockers cardiac selective",    case=False, na=False) |
        med.str.contains("beta blockers non-cardiac selective", case=False, na=False) |
        med.str.contains("alpha-beta blockers",                case=False, na=False)
    ).astype("int8")

    df["med_ace_ara2"] = (
        med.str.contains("ace inhibitors",                     case=False, na=False) |
        med.str.contains("angiotensin ii receptor blockers",   case=False, na=False) |
        med.str.contains("angiotensin receptor blocker",       case=False, na=False) |
        med.str.contains("ace inhibitor and diuretic",         case=False, na=False)
    ).astype("int8")

    df["med_diuretico_asa"] = (
        med.str.contains("diuretic - loop",                    case=False, na=False)
    ).astype("int8")

    # tiazida: solo componente de Triple Whammy — se mantiene para super-flag
    _tiazida = (
        med.str.contains("diuretic - thiazides and related",          case=False, na=False) |
        med.str.contains("diuretic - potassium sparing-thiazide",     case=False, na=False)
    ).astype("int8")
    df["med_diuretico_tiazida"] = _tiazida

    df["med_antipsicótico"] = (
        med.str.contains("antipsychotic",                      case=False, na=False) |
        med.str.contains("bipolar therapy agents - atypical",  case=False, na=False) |
        med.str.contains("phenothiazines",                     case=False, na=False)
    ).astype("int8")

    df["med_ssri_snri"] = (
        med.str.contains("selective serotonin reuptake inhibitors",       case=False, na=False) |
        med.str.contains("serotonin-norepinephrine reuptake inhibitors",  case=False, na=False) |
        med.str.contains("norepinephrine and dopamine reuptake inhibitors", case=False, na=False) |
        med.str.contains("tricyclics and related",                         case=False, na=False) |
        med.str.contains("serotonin-2 antagonist-reuptake inhibitors",    case=False, na=False) |
        med.str.contains("alpha-2 receptor antagonists",                  case=False, na=False)
    ).astype("int8")

    df["med_anticonvulsivante"] = (
        med.str.contains("anticonvulsant - gaba analogs",               case=False, na=False) |
        med.str.contains("anticonvulsant - pyrrolidine derivatives",    case=False, na=False) |
        med.str.contains("anticonvulsant - phenyltriazine derivatives", case=False, na=False) |
        med.str.contains("anticonvulsant - carboxylic acid derivatives", case=False, na=False) |
        med.str.contains("anticonvulsant - hydantoins",                 case=False, na=False) |
        med.str.contains("anticonvulsant - iminostilbene derivatives",  case=False, na=False) |
        med.str.contains("anticonvulsant - monosaccharide derivatives", case=False, na=False)
    ).astype("int8")

    df["med_respiratorio_inhalado"] = (
        med.str.contains("asthma/copd therapy - beta 2-adrenergic agents",        case=False, na=False) |
        med.str.contains("asthma/copd - anticholinergic agents",                  case=False, na=False) |
        med.str.contains("asthma therapy - inhaled corticosteroids",              case=False, na=False) |
        med.str.contains("asthma/copd therapy - beta adrenergic-glucocorticoid",  case=False, na=False) |
        med.str.contains("asthma/copd therapy - beta adrenergic-anticholinergic", case=False, na=False) |
        med.str.contains("asthma therapy - leukotriene receptor antagonists",     case=False, na=False)
    ).astype("int8")

    df["med_inmunosupresor"] = (
        med.str.contains("immunosuppressive - calcineurin inhibitors",                       case=False, na=False) |
        med.str.contains("immunosuppressive - inosine monophosphate dehydrogenase inhibitors", case=False, na=False)
    ).astype("int8")

    df["med_digoxina"] = (
        med.str.contains("digitalis glycosides",               case=False, na=False)
    ).astype("int8")

    df["med_antiaritmico"] = (
        med.str.contains("antiarrhythmic - class iii",          case=False, na=False)
    ).astype("int8")

    # AINE (solo componente de super-flags)
    _aine = (
        med.str.contains("nsaid analgesics",                   case=False, na=False)
    ).astype("int8")

    # Super-flags
    df["alto_riesgo_sangrado"] = (
        (df["med_anticoagulante"] == 1) |
        ((df["med_antiagregante"] == 1) & (_aine == 1)) |
        ((df["med_anticoagulante"] == 1) & (df["med_ssri_snri"] == 1))
    ).astype("int8")

    df["alto_riesgo_delirium"] = (
        (df["med_benzodiacepina"] == 1) | (df["med_antipsicótico"] == 1)
    ).astype("int8")

    df["riesgo_depresion_resp"] = (
        (df["med_opiaceo"] == 1) &
        (
            (df["med_benzodiacepina"]    == 1) |
            (df["med_antipsicótico"]     == 1) |
            (df["med_anticonvulsivante"] == 1)
        )
    ).astype("int8")

    logger.info("Bloque 3 completado: 18 flags ATC + 3 super-flags")
    return df

# BLOQUE 4 — Polifarmacia

def calcular_features_bloque4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de polifarmacia desde n_medicamentos.
    Umbrales clínicos estándar: polifarmacia ≥5, hiperpolifarmacia ≥10.
    sin_medicacion tiene patrón en U (acuity 1 trauma + acuity 5 joven sano).
    """
    df = df.copy()

    df["sin_medicacion"]    = (df["n_medicamentos"] == 0).astype("int8")
    df["polifarmacia"]      = (df["n_medicamentos"] >= 5).astype("int8")
    df["hiperpolifarmacia"] = (df["n_medicamentos"] >= 10).astype("int8")

    logger.info("Bloque 4 completado: polifarmacia")
    return df

# BLOQUE 5 — Chief complaint (13 síndromes clínicos)

def calcular_features_bloque5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags booleanas multi-label de chief complaint.
    Regex auditadas con experimento de alineación LLM (llama-3.1-8b-instant,
    400 filas estratificadas, F1 por flag). Versión v3 tras correcciones.

    Clasificación multi-label: un paciente puede tener múltiples flags = 1.
    cc_missing descartada (prevalencia 0.30%).

    En producción: el LLM devuelve directamente los booleanos via JSON
    (ver prompts/triage_cc_classifier.txt). No se aplica str.contains.
    """
    df = df.copy()
    cc = df["chiefcomplaint"].fillna("").str.lower().str.strip()

    df["cc_dolor_toracico"] = cc.str.contains(
        r"\bchest\s*pain\b|\bcp\b|\bchest\s*tightness\b|\bchest\s*pressure\b"
        r"|\bchest\s*discomfort\b|\bangina\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_disnea"] = cc.str.contains(
        r"\bdyspnea\b|\bsob\b|\bshortness\s*of\s*breath\b|\bbreathing\b"
        r"|\brespiratory\s*distress\b|\bbreathless\b|\bdyspnea\s*on\s*exertion\b"
        r"|\borthopnea\b|\bhypoxia\b|\bsmoke\s*inhalation\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_gi_agudo"] = cc.str.contains(
        r"\babd\s*pain\b|\babdominal\s*pain\b|\bepigastric\b|\brlq\b|\bllq\b"
        r"|\bruq\b|\bn/v\b|\bn/v/d\b|\bnausea\b|\bvomiting\b|\bdiarrhea\b"
        r"|\bgastric\b|\bbowel\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_neuro_ams"] = cc.str.contains(
        r"\baltered\s*mental\s*status\b|\bams\b|\bweakness\b|\bseizure\b"
        r"|\bsyncope\b|\bdizziness\b|\bconfusion\b|\bunresponsive\b"
        r"|\bloss\s*of\s*consciousness\b|\bnumbness\b|\bparalysis\b"
        r"|\bstroke\b|\bcva\b|\btia\b|\bunable\s*to\s*ambulate\b"
        r"|\bslurred\s*speech\b|\bunsteady\s*gait\b|\btremor\b|\bvisual\s*changes\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_cefalea"] = cc.str.contains(
        r"\bheadache\b|\bhead\s*pain\b|\bmigraine\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_trauma"] = cc.str.contains(
        r"\bs/p\s*fall\b|\bmvc\b|\bfall\b|\btrauma\b|\binjury\b|\baccident\b"
        r"|\bhead\s*injury\b|\blaceration\b|\bwound\b|\bfracture\b|\bstab\b"
        r"|\bassault\b|\bgsw\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_psiquiatrico"] = cc.str.contains(
        r"\bsi\b|\bsuicidal\b|\bsuicide\b|\banxiety\b|\bpsychiatric\b"
        r"|\bhallucination\b|\bagitation\b|\bdepression\b|\bmania\b"
        r"|\bparanoia\b|\bpsychosis\b|\bhi\b|\binsomnia\b|\baggitat\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_intoxicacion"] = cc.str.contains(
        r"\betoh\b|\bintoxication\b|\boverdose\b|\bingestion\b"
        r"|\bsubstance\b|\bdrug\s*use\b|\balcohol\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_infeccioso"] = cc.str.contains(
        r"\bfever\b|\bili\b|\binfection\b|\bsepsis\b|\bcellulitis\b"
        r"|\bpneumonia\b|\bmeningitis\b|\binfluenza\b|\bcovid\b"
        r"|\babscess\b|\bbites\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_urologico"] = cc.str.contains(
        r"\bdysuria\b|\bhematuria\b|\burinary\s*retention\b|\bflank\s*pain\b"
        r"|\buti\b|\bkidney\b|\brenal\b|\bfoley\b|\burinary\b"
        r"|\bgu\b|\btesticular\b|\bpelvic\s*pain\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_hemorragia_activa"] = cc.str.contains(
        r"\bbrbpr\b|\brectal\s*bleeding\b|\bgib\b|\bvaginal\s*bleeding\b"
        r"|\bepistaxis\b|\bhemoptysis\b|\bbleeding\b|\bhemorrhage\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_alergia_anafilaxia"] = cc.str.contains(
        r"\ballergic\s*reaction\b|\banaphylaxis\b|\ballergy\b|\bangioedema\b"
        r"|\bhives\b|\burticaria\b",
        case=False, regex=True, na=False
    ).astype("int8")

    df["cc_derivacion_urgente"] = cc.str.contains(
        r"\babnormal\s*labs\b|\babnormal\s*ct\b|\babnormal\s*ekg\b"
        r"|\btransfer\b|\breferral\b|\bhyperglycemia\b|\bhypertension\b"
        r"|\bhyponatremia\b|\bhyperkalemia\b|\beta\b",
        case=False, regex=True, na=False
    ).astype("int8")

    logger.info("Bloque 5 completado: 13 síndromes chief complaint")
    return df

# BLOQUE 6 — Historial ED y comorbilidades CCS

def _calcular_frecuentacion(edstays: pd.DataFrame) -> pd.DataFrame:
    """
    Features de frecuentación con operaciones vectorizadas.
    Sin self-join cartesiano — O(n log n) con groupby + shift + rolling.
    39 solapamientos administrativos corregidos a NaN (dias negativos).
    """
    es = edstays.sort_values(["subject_id", "intime"]).copy().reset_index(drop=True)

    es["n_visitas_previas"] = es.groupby("subject_id").cumcount().astype("int16")
    es["primera_visita"]    = (es["n_visitas_previas"] == 0).astype("int8")
    es["outtime_previa"]    = es.groupby("subject_id")["outtime"].shift(1)

    es["dias_desde_ultima_visita"] = (
        (es["intime"] - es["outtime_previa"]).dt.total_seconds() / 86400
    ).round(1)
    # fix solapamientos administrativos
    es.loc[es["dias_desde_ultima_visita"] < 0, "dias_desde_ultima_visita"] = np.nan

    # ventanas temporales con rolling temporal
    es = es.set_index("intime").sort_index()
    es["visitas_ultimo_mes"] = (
        es.groupby("subject_id")["stay_id"]
        .transform(lambda x: x.rolling("30D", closed="left").count())
        .fillna(0).astype("int16")
    )
    es["visitas_ultimo_año"] = (
        es.groupby("subject_id")["stay_id"]
        .transform(lambda x: x.rolling("365D", closed="left").count())
        .fillna(0).astype("int16")
    )
    es = es.reset_index()
    es["frecuentador"] = (es["visitas_ultimo_mes"] >= UMBRAL_FRECUENTADOR).astype("int8")

    logger.info(
        f"Frecuentación: {es['primera_visita'].sum():,} primeras visitas | "
        f"{es['frecuentador'].sum():,} frecuentadores"
    )
    return es

def _calcular_comorbilidades_ccs(
    edstays: pd.DataFrame,
    label_map_ccs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Macro-flags CCS retrospectivas vectorizadas.
    Estrategia: isin() + shift(1) + cummax() — O(n log n), ~1.3s sobre 425K filas.
    Regla lookback: outtime_previa < intime_actual.
    """
    es = edstays.sort_values(["subject_id", "intime"]).copy().reset_index(drop=True)

    for flag, codigos in CCS_MACROFLAGS.items():
        stays_con_flag = label_map_ccs[label_map_ccs["ccs_id"].isin(codigos)]["stay_id"]
        es[f"_{flag}_now"] = es["stay_id"].isin(stays_con_flag).astype("int8")
        es[flag] = es.groupby("subject_id")[f"_{flag}_now"].shift(1).fillna(0)
        es[flag] = es.groupby("subject_id")[flag].cummax().astype("int8")
        es.drop(columns=[f"_{flag}_now"], inplace=True)

    logger.info(f"Comorbilidades CCS: {len(CCS_MACROFLAGS)} macro-flags calculadas")
    return es

def calcular_features_bloque6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orquestador Bloque 6: frecuentación + comorbilidades CCS.
    """
    label_map_ccs = pd.read_parquet(DATA_INTERIM / "label_map_ccs.parquet")

    # Construir edstays desde el df — mismas columnas que el CSV original
    edstays = (
        df[["subject_id", "stay_id", "intime", "outtime"]]
        .drop_duplicates("stay_id")
        .copy()
    )
    # intime y outtime deben ser datetime (loader los parsea, pero por si acaso)
    edstays["intime"]   = pd.to_datetime(edstays["intime"])
    edstays["outtime"]  = pd.to_datetime(edstays["outtime"])

    logger.info(f"edstays construido desde df: {len(edstays):,} filas")

    # frecuentación
    edstays_frec = _calcular_frecuentacion(edstays)
    cols_frec = [
        "stay_id", "n_visitas_previas", "primera_visita",
        "dias_desde_ultima_visita", "visitas_ultimo_mes",
        "visitas_ultimo_año", "frecuentador",
    ]
    df = df.merge(edstays_frec[cols_frec], on="stay_id", how="left")

    # comorbilidades CCS
    edstays_ccs = _calcular_comorbilidades_ccs(edstays, label_map_ccs)
    cols_ccs = ["stay_id"] + list(CCS_MACROFLAGS.keys())
    df = df.merge(edstays_ccs[cols_ccs].drop_duplicates("stay_id"), on="stay_id", how="left")

    logger.info("Bloque 6 completado: frecuentación + comorbilidades CCS")
    return df

# Orquestador principal

def calcular_todas_las_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering sobre el dataset limpio.
    Aplica los 6 bloques en orden y devuelve df con las 88 features definitivas.

    Args:
        df: DataFrame limpio producido por cleaner.limpiar_dataset()

    Returns:
        DataFrame con TODAS_FEATURES + TARGET calculadas.
    """
    logger.info(f"Iniciando feature engineering: {len(df):,} filas | {df.shape[1]} columnas")

    df = calcular_features_bloque1(df)
    df = calcular_features_bloque2(df)
    df = calcular_features_bloque3(df)
    df = calcular_features_bloque4(df)
    df = calcular_features_bloque5(df)
    df = calcular_features_bloque6(df)

    # verificar que todas las features están presentes
    faltantes = [f for f in TODAS_FEATURES if f not in df.columns]
    if faltantes:
        logger.warning(f"Features faltantes ({len(faltantes)}): {faltantes}")
    else:
        logger.success(f"Feature engineering completado: {len(TODAS_FEATURES)} features | 0 faltantes")

    return df

def cargar_features(forzar: bool = False) -> pd.DataFrame:
    """
    Pipeline completo con caché: raw → limpieza → features → dataset_features.parquet.

    Utilidad standalone para obtener las 88 features sin pasar por los notebooks.
    El flujo de entrenamiento habitual (notebooks 06/07) carga normalmente los
    artefactos generados en ``05b_feature_validation.ipynb`` (split temporal),
    mientras que ``05_feature_engineering.ipynb`` exporta el espacio completo
    de características (p. ej. ``dataset_features.parquet``).

    Args:
        forzar: Si True, recalcula aunque exista el caché.

    Returns:
        DataFrame con TODAS_FEATURES + TARGET (88 + 1 columnas).
    """
    from triaje_ia.data.cleaner import limpiar_dataset
    from triaje_ia.data.loader import cargar_dataset_base

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cache = DATA_PROCESSED / "dataset_features.parquet"

    if cache.exists() and not forzar:
        logger.warning(
            f"Cargando desde caché: {cache}. "
            "Usa forzar=True si cambiaste el pipeline de features."
        )
        df = pd.read_parquet(cache)
        logger.success(f"Cargado: {len(df):,} filas | {df.shape[1]} columnas")
        return df

    df = cargar_dataset_base()
    df = limpiar_dataset(df)
    df = calcular_todas_las_features(df)

    df[TODAS_FEATURES + [TARGET]].to_parquet(cache, index=False)
    logger.success(f"Guardado en {cache}: {len(df):,} filas | {len(TODAS_FEATURES)} features")
    return df

# --- Utilidades de Persistencia de Métricas ---

def update_metrics_json(feature, decision_final=None, H=None, eta2=None, V_cramer=None, **extra):
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    # Resolviendo ARTIFACTS_DIR relativo a este archivo (src/triaje_ia/data/features.py)
    # asumiendo que artifacts está en la raíz del proyecto
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
    ARTIFACTS_DIR = ROOT_DIR / "artifacts"
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    METRICS_FILE = ARTIFACTS_DIR / "feature_metrics.json"

    if not METRICS_FILE.exists():
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "features_con_metricas": {}}, f, indent=2, ensure_ascii=False)

    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "features_con_metricas" not in data or not isinstance(data["features_con_metricas"], dict):
        data["features_con_metricas"] = {}

    data.setdefault("version", "1.0")
    data["ultima_actualizacion"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if feature not in data["features_con_metricas"]:
        data["features_con_metricas"][feature] = {
            "decision_final": "pendiente",
            "H": None,
            "eta2": None,
            "V_cramer": None,
        }

    entry = data["features_con_metricas"][feature]

    if decision_final is not None:
        entry["decision_final"] = decision_final
    if H is not None:
        entry["H"] = H
    if eta2 is not None:
        entry["eta2"] = eta2
    if V_cramer is not None:
        entry["V_cramer"] = V_cramer

    for k, v in extra.items():
        if v is None:
            continue
        entry[k] = v

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
