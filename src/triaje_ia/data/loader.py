"""
src/triaje_ia/data/loader.py
─────────────────────────────
Carga y merge de las tablas MIMIC-IV-ED + patients (MIMIC-IV core).
Produce un DataFrame por stay_id con los dos labels y las features
disponibles en el MINUTO 0 de la llegada del paciente.

Arquitectura del problema:

  LABELS:
    acuity            → triage.acuity (1-5)          clasificación multiclase
    label_disposition → edstays.disposition (binario) ingreso vs alta

  FEATURES (solo lo que sabe el enfermero en el minuto 0):
    Demografía:        gender, age, arrival_transport
    Vitales iniciales: temperature, heartrate, resprate, o2sat, sbp, dbp, pain
    Motivo consulta:   chiefcomplaint
    Medicación previa: n_medicamentos + flags binarios desde medrecon.etcdescription
    Diagnóstico principal: ccs_category (ICD-9+10 unificado, refinado en features.py)

  DESCARTADAS EXPLÍCITAMENTE:
    vitalsign → mediciones durante la estancia, no disponibles en minuto 0
    pyxis     → fármacos administrados durante la estancia, no en minuto 0
    diagnosis → diagnóstico final, no disponible en minuto 0
                → icd_code (seq_num=1) mapeado a ccs_category (ICD-9+10 unificado)
                  disponible como contexto; la lógica de morbilidad va en featuring

  EDAD:
    Calculada desde patients.csv.gz (MIMIC-IV core, NO MIMIC-IV-ED)
    Fórmula: anchor_age + (intime.year - anchor_year)
    Descarga: https://physionet.org/content/mimiciv/3.1/hosp/patients.csv.gz
    Destino:  data/raw/patients.csv.gz


"""

import pandas as pd
from loguru import logger

from triaje_ia.config import DATA_RAW, DATA_INTERIM


# Constantes
TABLAS_ED    = ["edstays", "triage", "medrecon"]
TABLA_PATIENTS = DATA_RAW / "patients.csv.gz"

# Valores de disposition que se codifican como ingreso (1)
DISPOSICION_INGRESO: frozenset[str] = frozenset({"ADMITTED", "TRANSFER", "ADMIT"})


# Carga de tablas raw

def cargar_tablas_raw() -> dict[str, pd.DataFrame]:
    """
    Carga las tablas MIMIC-IV-ED necesarias y patients de MIMIC-IV core.

    Returns:
        Diccionario {nombre_tabla: DataFrame} sin ninguna modificación.
    """
    dfs: dict[str, pd.DataFrame] = {}

    # Tablas de MIMIC-IV-ED
    for tabla in TABLAS_ED:
        ruta = DATA_RAW / f"{tabla}.csv.gz"
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encuentra: {ruta}\n"
            )
        dfs[tabla] = pd.read_csv(ruta, low_memory=False)
        logger.info(
            f"{tabla.upper()}: {len(dfs[tabla]):,} filas | "
            f"{dfs[tabla].shape[1]} columnas"
        )

    # patients (MIMIC-IV core) — solo las columnas necesarias para la edad
    if not TABLA_PATIENTS.exists():
        raise FileNotFoundError(
            f"No se encuentra: {TABLA_PATIENTS}\n"
        )

    dfs["patients"] = pd.read_csv(
        TABLA_PATIENTS,
        usecols=["subject_id", "anchor_age", "anchor_year"],
        low_memory=False,
    )
    logger.info(
        f"PATIENTS (MIMIC-IV core): {len(dfs['patients']):,} pacientes"
    )

    # diagnosis (MIMIC-IV-ED) — solo seq_num=1 para ccs_category (solo el diagnóstico principal)
    ruta_diag = DATA_RAW / "diagnosis.csv.gz"
    if ruta_diag.exists():
        dfs["diagnosis"] = pd.read_csv(
            ruta_diag,
            usecols=["stay_id", "icd_code", "icd_version", "seq_num"],
            low_memory=False,
        )
        logger.info(f"DIAGNOSIS: {len(dfs['diagnosis']):,} filas")
    else:
        dfs["diagnosis"] = None
        logger.warning(
            "diagnosis.csv.gz no encontrado"
        )

    return dfs


# Cálculo de edad
def _calcular_edad(
    df: pd.DataFrame,
    df_patients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la edad del paciente en el momento de la visita.

    MIMIC-IV mantiene el intervalo entre anchor_year (año de nacimiento) e intime (año de la visita).
    Fórmula: age = anchor_age + (intime.year - anchor_year)
    """
    df = df.copy()

    if df["intime"].dtype == object:
        df["intime"] = pd.to_datetime(df["intime"], errors="coerce")

    df = df.merge(
        df_patients[["subject_id", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )

    df["age"] = (
        df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])
    ).astype("Int64")  # Int64 soporta NaN en enteros

    # Sanidad: edades fuera de rango fisiológico → NaN
    invalidas = ((df["age"] < 0) | (df["age"] > 120)).sum()
    if invalidas > 0:
        logger.warning(f"Edad: {invalidas:,} valores fuera de [0,120] → NaN")
        df["age"] = df["age"].where((df["age"] >= 0) & (df["age"] <= 120))

    df = df.drop(columns=["anchor_age", "anchor_year"])

    logger.info(
        f"Edad: mediana={df['age'].median():.0f} | "
        f"rango=[{df['age'].min()}, {df['age'].max()}] | "
        f"nulos={df['age'].isna().sum():,}"
    )
    return df


# Agregación de medicación previa
def _agregar_medrecon(df_medrecon: pd.DataFrame) -> pd.DataFrame:
    """Agrega medrecon a una fila por stay_id. NO genera flags med_* (→ features.py)."""
    df_medrecon = df_medrecon.drop_duplicates(subset=["stay_id", "name"])
    agg = (
        df_medrecon.groupby("stay_id")
        .agg(
            n_medicamentos=("name", "size"),
            medicacion_raw=(
                "etcdescription",
                lambda x: ", ".join(x.dropna().astype(str).str.lower()),
            ),
        )
        .reset_index()
    )
    logger.info(f"medrecon agregado: {len(agg):,} stays | columnas: n_medicamentos, medicacion_raw")
    return agg


# Label disposition
def _construir_label_disposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el label binario de disposition.

    1 (ingreso)  → ADMITTED, TRANSFER, ADMIT
    0 (alta)     → HOME, DISCHARGED, LEFT WITHOUT BEING SEEN, etc.
    NaN          → valores nulos (en el futuro se decide qué hacer en cleaner.py)
    """
    df = df.copy()

    disp = df["disposition"].str.upper().str.strip()

    df["label_disposition"] = (
        disp.isin(DISPOSICION_INGRESO).astype("Int8")
    )
    # Los nulos originales deben propagarse como NA, no como 0
    df.loc[disp.isna(), "label_disposition"] = pd.NA

    n_ingreso = df["label_disposition"].eq(1).sum()
    n_alta    = df["label_disposition"].eq(0).sum()
    logger.info(
        f"label_disposition: ingreso={n_ingreso:,} "
        f"({n_ingreso/len(df)*100:.1f}%) | "
        f"alta={n_alta:,} ({n_alta/len(df)*100:.1f}%)"
    )
    return df


# CCS diagnóstico principal
def _agregar_ccs_diagnostico(
    df_diagnosis: pd.DataFrame,
    ccs_icd9: pd.DataFrame,
    ccs_icd10: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mapea el diagnóstico principal (seq_num=1) de cada visita a su
    categoría CCS unificada (ICD-9 e ICD-10 → mismo espacio numérico 1-285).

    Normalización ICD-9: ljust(5) para compatibilidad con dxref2015.
    ICD-10: merge directo sin transformación.

    Produce:
    - ccs_category: Int16 con la categoría CCS, NaN si no matchea
    """
    diag = df_diagnosis[df_diagnosis["seq_num"] == 1].copy()

    # ICD-9: normalizar con ljust(5)
    diag_9 = diag[diag["icd_version"] == 9].copy()
    diag_9["_code_norm"] = diag_9["icd_code"].astype(str).str.strip().str.ljust(5)
    diag_9 = diag_9.merge(
        ccs_icd9[["ICD-9-CM CODE", "CCS CATEGORY"]],
        left_on="_code_norm", right_on="ICD-9-CM CODE", how="left",
    )

    # ICD-10: directo
    diag_10 = diag[diag["icd_version"] == 10].copy()
    diag_10 = diag_10.merge(
        ccs_icd10[["ICD-10-CM CODE", "CCS CATEGORY"]],
        left_on="icd_code", right_on="ICD-10-CM CODE", how="left",
    )

    diag_all = pd.concat([diag_9, diag_10], ignore_index=True)
    diag_all["ccs_category"] = (
        pd.to_numeric(diag_all["CCS CATEGORY"], errors="coerce")
        .astype("Int16")
    )

    result = (
        diag_all[["stay_id", "ccs_category"]]
        .drop_duplicates("stay_id")
    )

    matched_pct = result["ccs_category"].notna().mean() * 100
    logger.info(
        f"CCS diagnóstico principal: {len(result):,} stays | "
        f"match={matched_pct:.1f}% | "
        f"categorías únicas={result['ccs_category'].nunique()}"
    )
    return result


# Merge principal
def construir_dataset_base(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge de las tablas en un DataFrame por stay_id.

    Joins:
    - edstays INNER triage   → 1:1, 425.087 filas sin pérdida
    - LEFT JOIN patients     → edad (todos tienen subject_id)
    - LEFT JOIN medrecon agg → no todos tienen medicación previa
    - LEFT JOIN ccs_category → diagnóstico principal ICD-9/10 → CCS unificado

    Labels en el DataFrame resultante:
    - acuity:            1-5 
    - label_disposition: binario 0/1
    """
    logger.info("Construyendo dataset base...")

    # 1. Base: edstays INNER triage
    df = pd.merge(
        dfs["edstays"],
        dfs["triage"],
        on=["subject_id", "stay_id"],
        how="inner",
        suffixes=("_ed", "_tr"),
    )
    logger.info(f"edstays INNER triage: {len(df):,} filas")

    # Resolver columna gender duplicada: edstays es la canónica
    if "gender_ed" in df.columns:
        df = df.rename(columns={"gender_ed": "gender"})
        df = df.drop(columns=["gender_tr"], errors="ignore")

    # 2. Edad desde patients
    df = _calcular_edad(df, dfs["patients"])

    # 3. Medicación previa
    medrecon_agg = _agregar_medrecon(dfs["medrecon"])
    df = pd.merge(df, medrecon_agg, on="stay_id", how="left")
    logger.info(f"+ medrecon: {len(df):,} filas")

    # 4. Label disposition
    df = _construir_label_disposition(df)

    # 5. CCS del diagnóstico principal (ICD-9 + ICD-10 unificado)
    if dfs.get("diagnosis") is not None:
        ccs_icd9, ccs_icd10 = cargar_mappings_ccs()
        ccs_agg = _agregar_ccs_diagnostico(dfs["diagnosis"], ccs_icd9, ccs_icd10)
        df = pd.merge(df, ccs_agg, on="stay_id", how="left")
        logger.info(
            f"+ ccs_category: {df['ccs_category'].notna().sum():,} stays con categoría"
        )

    # 6. Reducción de tipos de datos: los NaN del LEFT JOIN son ausencia real → 0
    df["n_medicamentos"] = df["n_medicamentos"].fillna(0).astype("int16")
    df["medicacion_raw"] = df["medicacion_raw"].fillna("")

    logger.success(
        f"Dataset base: {len(df):,} filas | {df.shape[1]} columnas"
    )
    return df


# Carga de mappings CCS
def cargar_mappings_ccs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los mappings CCS para su uso en la fase de featuring."""
    ruta_9  = DATA_INTERIM / "ccs_icd9_clean.parquet"
    ruta_10 = DATA_INTERIM / "ccs_icd10_clean.parquet"

    if not ruta_9.exists() or not ruta_10.exists():
        raise FileNotFoundError(
            "Mappings CCS no encontrados en data/interim/. "
            "Ejecuta primero el notebook 01_data_exploration.ipynb."
        )

    return pd.read_parquet(ruta_9), pd.read_parquet(ruta_10)


# Orquestador principal
def cargar_dataset_base(forzar: bool = False) -> pd.DataFrame:
    """
    Pipeline completo: raw → dataset base listo para cleaner.py.
    Usa caché en data/interim/ si existe.

    Args:
        forzar: Recalcula aunque exista el caché.

    Returns:
        DataFrame listo para cleaner.limpiar_dataset().
    """
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    cache = DATA_INTERIM / "dataset_base.parquet"

    if cache.exists() and not forzar:
        # AVISO: si cambiaste parámetros del pipeline-> Usar forzar=True para regenerar.
        logger.warning(
            f"Cargando desde caché: {cache}. "
            "Si cambiaste parámetros del pipeline, usa forzar=True."
        )
        df = pd.read_parquet(cache)
        logger.success(f"Cargado: {len(df):,} filas | {df.shape[1]} columnas")
        return df

    dfs = cargar_tablas_raw()
    df  = construir_dataset_base(dfs)

    df.to_parquet(cache, index=False)
    logger.success(f"Guardado en {cache}")
    return df