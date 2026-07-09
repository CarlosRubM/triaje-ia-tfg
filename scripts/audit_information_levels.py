# ruff: noqa: E501,I001

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama
import pandas as pd


def find_project_root(start: Path) -> Path:
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("No se pudo localizar la raiz del proyecto.")


PROJECT_ROOT = find_project_root(Path.cwd()).resolve()
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from final_prompt_audit import (  # noqa: E402
    DEFAULT_MODEL,
    PROMPTS,
    append_jsonl,
    compare_vector,
    construir_casos_finales,
    load_cache,
    markdown_table,
)
from triaje_ia.inference.predictor import TriajePredictor  # noqa: E402
from triaje_ia.llm.extractor import _completar_datos_explicitos  # noqa: E402
from triaje_ia.llm.normalizer import normalizar_vector_clinico  # noqa: E402
from triaje_ia.llm.schemas import VectorClinico  # noqa: E402
from triaje_ia.llm.validator import validar_vector_clinico  # noqa: E402


OUT_BASE_DIR = PROJECT_ROOT / "reports" / "information_level_audit"
PROMPT_ID = "v3_final"
PROMPT_PATH = PROMPTS[PROMPT_ID]

LEVELS = {
    "L1_minimo": "Edad/sexo + motivo principal.",
    "L2_constantes": "L1 + constantes vitales y dolor si aparece.",
    "L3_contexto": "L2 + duracion, Glasgow/estado y negaciones relevantes.",
    "L4_completo": "Narrativa completa original.",
}

SYMPTOM_ES = {
    "abdominal pain": "dolor abdominal",
    "altered mental status": "alteracion del estado mental",
    "angioedema": "edema de labios o lengua",
    "ankle pain": "dolor de tobillo",
    "ankle sprain": "torcedura de tobillo",
    "aphasia": "dificultad para hablar",
    "arm weakness": "debilidad de brazo",
    "bleeding": "sangrado",
    "bradypnea": "respiracion lenta",
    "chest pain": "dolor toracico",
    "chills": "escalofrios",
    "conjunctivitis": "ojo rojo con leganas",
    "cough": "tos",
    "cyanosis": "cianosis",
    "dental pain": "dolor dental",
    "dizziness": "mareo",
    "dry eye": "ojo seco",
    "dysuria": "disuria",
    "dyspnea": "disnea",
    "ear fullness": "oido taponado",
    "ear pain": "dolor de oido",
    "fever": "fiebre",
    "flank pain": "dolor lumbar",
    "head injury": "golpe en la cabeza",
    "headache": "cefalea",
    "hearing loss": "disminucion de audicion",
    "hematemesis": "hematemesis",
    "hemiparesis": "perdida de fuerza",
    "hypoglycemia": "hipoglucemia",
    "hypoxemia": "baja saturacion",
    "ingrown toenail": "una encarnada",
    "itching": "picor",
    "leg pain": "dolor de pierna",
    "leg redness": "pierna roja",
    "minor burn": "quemadura pequena",
    "minor wound": "corte superficial",
    "nausea": "nauseas",
    "neck stiffness": "rigidez de nuca",
    "pallor": "palidez",
    "pelvic pain": "dolor pelvico",
    "photophobia": "fotofobia",
    "rectal bleeding": "rectorragia",
    "redness": "enrojecimiento",
    "right lower quadrant abdominal pain": "dolor en fosa iliaca derecha",
    "right shoulder pain": "dolor de hombro derecho",
    "scalp wound": "herida en cuero cabelludo",
    "seizure": "convulsion",
    "stridor": "estridor",
    "suicidal ideation": "ideas autoliticas",
    "suprapubic pain": "dolor suprapubico",
    "sweating": "sudoracion",
    "syncope": "sincope",
    "thunderclap headache": "cefalea brusca muy intensa",
    "toe pain": "dolor en dedo del pie",
    "urticaria": "urticaria",
    "urinary retention": "retencion urinaria",
    "vaginal bleeding": "sangrado vaginal",
    "vertigo": "vertigo",
    "vision loss": "perdida de vision",
    "vomiting": "vomitos",
    "wheezing": "pitos respiratorios",
    "wrist pain": "dolor de muneca",
    "deformity": "deformidad",
    "anxiety": "ansiedad",
    "administrative request": "solicitud administrativa",
    "wound check": "revision de herida",
    "suture removal": "retirada de puntos",
    "chronic knee pain": "dolor cronico de rodilla",
    "medication refill": "renovacion de medicacion",
}

CONTEXT_PATTERNS = [
    r"\bGlasgow\s*(?:\d{1,2}|no valorable|15 adaptado)\b",
    r"\bGCS\s*(?:\d{1,2}|no valorable)\b",
    r"\bbuen estado general\b",
    r"\bmal estado general\b",
    r"\bpalid[oa]\b",
    r"\bsudoros[oa]\b",
    r"\bconfus[oa]\b",
    r"\bsomnolient[oa]\b",
    r"\bdesorientad[oa]\b",
    r"\borientad[oa]\b",
    r"\bagitad[oa]\b",
    r"\bansios[oa]\b",
]

NEGATION_PATTERN = re.compile(
    r"\b(?:sin|niega|no refiere|no presenta|no)\s+[^.]+", re.IGNORECASE
)


@dataclass(frozen=True)
class InformationLevelCase:
    case_id: str
    original_case_id: str
    info_level: str
    info_level_description: str
    familia_clinica: str
    esi_referencia: int
    dificultad: str
    foco_auditoria: str
    narrativa: str
    narrativa_base: str
    vector_esperado: dict[str, Any]
    razon_referencia: str


def sexo_text(sexo: str) -> str:
    if sexo == "M":
        return "Varon"
    if sexo == "F":
        return "Mujer"
    return "Paciente"


def translate_symptom(symptom: str) -> str:
    return SYMPTOM_ES.get(symptom.lower().strip(), symptom.replace("_", " "))


def chief_complaint(vector: dict[str, Any], fallback: str) -> str:
    sintomas = vector.get("sintomas_presentes") or []
    if sintomas:
        translated = [translate_symptom(s) for s in sintomas[:3]]
        return ", ".join(translated)
    return fallback


def vitals_sentence(vector: dict[str, Any]) -> str:
    parts: list[str] = []
    sis = vector.get("presion_sistolica")
    dia = vector.get("presion_diastolica")
    if sis is not None and dia is not None:
        parts.append(f"TA {sis}/{dia}")
    elif sis is None and dia is None:
        pass
    if vector.get("frecuencia_cardiaca") is not None:
        parts.append(f"FC {vector['frecuencia_cardiaca']}")
    if vector.get("frecuencia_respiratoria") is not None:
        parts.append(f"FR {vector['frecuencia_respiratoria']}")
    if vector.get("saturacion_oxigeno") is not None:
        parts.append(f"SatO2 {vector['saturacion_oxigeno']}%")
    if vector.get("temperatura") is not None:
        parts.append(f"T {vector['temperatura']}")
    if vector.get("nivel_dolor") is not None:
        parts.append(f"dolor {vector['nivel_dolor']}/10")
    return ", ".join(parts)


def extract_context(original: str, vector: dict[str, Any]) -> str:
    chunks: list[str] = []
    duration = vector.get("duracion_sintomas")
    if duration:
        chunks.append(f"Evolucion: {duration}.")

    for pattern in CONTEXT_PATTERNS:
        match = re.search(pattern, original, flags=re.IGNORECASE)
        if match:
            value = match.group(0).strip()
            if value.lower() not in {chunk.lower().strip(".") for chunk in chunks}:
                chunks.append(f"{value}.")

    for match in NEGATION_PATTERN.finditer(original):
        value = match.group(0).strip()
        if len(value) <= 90:
            chunks.append(f"{value}.")

    if not chunks:
        return ""
    seen: set[str] = set()
    unique: list[str] = []
    for chunk in chunks:
        key = chunk.lower()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return " ".join(unique[:5])


def build_level_narrative(base_case: Any, level: str) -> str:
    vector = base_case.vector_esperado
    intro = (
        f"{sexo_text(vector['sexo'])} {vector['edad']}a. "
        f"Motivo: {chief_complaint(vector, base_case.foco_auditoria)}."
    )
    if level == "L1_minimo":
        return intro

    vitals = vitals_sentence(vector)
    l2 = intro if not vitals else f"{intro} {vitals}."
    if level == "L2_constantes":
        return l2

    context = extract_context(base_case.narrativa, vector)
    l3 = l2 if not context else f"{l2} {context}"
    if level == "L3_contexto":
        return l3

    if level == "L4_completo":
        return base_case.narrativa

    raise ValueError(f"Nivel no soportado: {level}")


def build_information_cases(limit: int | None = None) -> list[InformationLevelCase]:
    base_cases = construir_casos_finales()
    if limit is not None:
        base_cases = base_cases[:limit]

    rows: list[InformationLevelCase] = []
    for base_case in base_cases:
        for level, description in LEVELS.items():
            rows.append(
                InformationLevelCase(
                    case_id=f"{base_case.case_id}_{level}",
                    original_case_id=base_case.case_id,
                    info_level=level,
                    info_level_description=description,
                    familia_clinica=base_case.familia_clinica,
                    esi_referencia=base_case.esi_referencia,
                    dificultad=base_case.dificultad,
                    foco_auditoria=base_case.foco_auditoria,
                    narrativa=build_level_narrative(base_case, level),
                    narrativa_base=base_case.narrativa,
                    vector_esperado=base_case.vector_esperado,
                    razon_referencia=base_case.razon_referencia,
                )
            )
    return rows


def extract_with_v3_final(narrativa: str, *, prompt_text: str, model: str) -> tuple[VectorClinico, VectorClinico, float]:
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"Narrativa:\n{narrativa}"},
        ],
        format=VectorClinico.model_json_schema(),
        options={"temperature": 0.0},
    )
    elapsed = time.perf_counter() - start
    raw_vector = normalizar_vector_clinico(
        VectorClinico.model_validate_json(response.message.content)
    )
    final_vector = normalizar_vector_clinico(
        _completar_datos_explicitos(narrativa, raw_vector)
    )
    return raw_vector, final_vector, elapsed


def process_case(
    case: InformationLevelCase,
    *,
    prompt_text: str,
    model: str,
    predictor: TriajePredictor | None,
) -> dict[str, Any]:
    base = {
        "prompt_id": PROMPT_ID,
        "prompt_path": str(PROMPT_PATH.relative_to(PROJECT_ROOT)),
        "model": model,
        **asdict(case),
        "vector_esperado_json": json.dumps(case.vector_esperado, ensure_ascii=False),
    }
    try:
        raw_vector, final_vector, elapsed = extract_with_v3_final(
            case.narrativa,
            prompt_text=prompt_text,
            model=model,
        )
        alertas = validar_vector_clinico(final_vector, case.narrativa)
        comparison = compare_vector(case.vector_esperado, final_vector)
        row = {
            **base,
            "ok": True,
            "error": "",
            "elapsed_seconds": round(elapsed, 3),
            "n_alertas_validador": len(alertas),
            "alertas_validador": " | ".join(str(alerta) for alerta in alertas),
            "raw_vector_json": json.dumps(raw_vector.model_dump(), ensure_ascii=False),
            "final_vector_json": json.dumps(final_vector.model_dump(), ensure_ascii=False),
            **comparison,
        }
        if predictor is not None:
            result = predictor.predict(final_vector, case.narrativa)
            probas = [float(x) for x in result.probas]
            delta = result.clase_predicha - case.esi_referencia
            row.update(
                {
                    "clase_predicha": result.clase_predicha,
                    "confianza": result.confianza,
                    "alerta_a1_activada": result.alerta_a1_activada,
                    "p_esi_1": probas[0],
                    "p_esi_2": probas[1],
                    "p_esi_3": probas[2],
                    "p_esi_4": probas[3],
                    "p_esi_5": probas[4],
                    "delta_vs_referencia": delta,
                    "abs_delta_vs_referencia": abs(delta),
                    "discrepancia_severa": abs(delta) >= 2,
                    "subtriaje": delta > 0,
                    "sobretriaje": delta < 0,
                    "fallo_seguridad_a1": case.esi_referencia == 1
                    and result.clase_predicha != 1
                    and not result.alerta_a1_activada,
                    "sobretriaje_leve_a1": case.esi_referencia >= 4
                    and result.clase_predicha == 1,
                }
            )
        return row
    except Exception as exc:
        return {**base, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level, group in results.groupby("info_level", sort=False):
        ok = group[group["ok"] == True].copy()  # noqa: E712
        row: dict[str, Any] = {
            "info_level": level,
            "description": LEVELS[level],
            "n_cases": int(len(group)),
            "n_ok": int(len(ok)),
            "n_failed": int((group["ok"] != True).sum()),  # noqa: E712
        }
        if not ok.empty:
            row.update(
                {
                    "mean_elapsed_seconds": round(float(ok["elapsed_seconds"].mean()), 3),
                    "mean_scalar_match_rate": round(float(ok["scalar_match_rate"].mean()), 3),
                    "mean_symptom_recall": round(float(ok["symptom_recall"].mean()), 3),
                    "empty_symptoms": int(ok["empty_symptoms"].sum()),
                    "validator_alerts_total": int(ok["n_alertas_validador"].sum()),
                    "symptom_extra_total": int(ok["symptom_extra_count"].sum()),
                }
            )
            if "clase_predicha" in ok:
                row.update(
                    {
                        "mean_abs_delta_vs_reference": round(
                            float(ok["abs_delta_vs_referencia"].mean()), 3
                        ),
                        "severe_discrepancies": int(ok["discrepancia_severa"].sum()),
                        "safety_failures_a1": int(ok["fallo_seguridad_a1"].sum()),
                        "low_acuity_predicted_a1": int(ok["sobretriaje_leve_a1"].sum()),
                        "a1_alerts": int(ok["alerta_a1_activada"].sum()),
                        "pred_esi_1": int((ok["clase_predicha"] == 1).sum()),
                        "pred_esi_2": int((ok["clase_predicha"] == 2).sum()),
                        "pred_esi_3": int((ok["clase_predicha"] == 3).sum()),
                        "pred_esi_4": int((ok["clase_predicha"] == 4).sum()),
                        "pred_esi_5": int((ok["clase_predicha"] == 5).sum()),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def build_transition_summary(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[(results["ok"] == True) & results["abs_delta_vs_referencia"].notna()].copy()  # noqa: E712
    if ok.empty:
        return pd.DataFrame()
    pivot = ok.pivot_table(
        index=["original_case_id", "familia_clinica", "foco_auditoria", "esi_referencia"],
        columns="info_level",
        values="abs_delta_vs_referencia",
        aggfunc="first",
    ).reset_index()
    for level in LEVELS:
        if level not in pivot.columns:
            pivot[level] = pd.NA
    pivot["delta_L1_to_L4"] = pivot["L4_completo"] - pivot["L1_minimo"]
    pivot["mejora_L1_to_L4"] = pivot["delta_L1_to_L4"] < 0
    pivot["empeora_L1_to_L4"] = pivot["delta_L1_to_L4"] > 0
    pivot["igual_L1_to_L4"] = pivot["delta_L1_to_L4"] == 0
    return pivot


def write_summary_report(path: Path, summary: pd.DataFrame, transitions: pd.DataFrame, results: pd.DataFrame) -> None:
    lines = [
        "# Auditoria de calidad de informacion con v3_final",
        "",
        "Compara 4 niveles de narrativa sobre los mismos 50 casos sinteticos.",
        "El modelo, el prompt, los artefactos y la politica A1 permanecen congelados.",
        "",
        "## Resumen por nivel",
        "",
        markdown_table(summary),
        "",
    ]

    if not transitions.empty:
        lines.extend(
            [
                "## Cambio L1 a L4",
                "",
                f"- Mejoran: {int(transitions['mejora_L1_to_L4'].sum())} casos.",
                f"- Empeoran: {int(transitions['empeora_L1_to_L4'].sum())} casos.",
                f"- Igual: {int(transitions['igual_L1_to_L4'].sum())} casos.",
                "",
            ]
        )
        changed = transitions[transitions["delta_L1_to_L4"] != 0].copy()
        if changed.empty:
            lines.append("No hay cambios ordinales entre L1 y L4.")
        else:
            cols = [
                "original_case_id",
                "familia_clinica",
                "foco_auditoria",
                "esi_referencia",
                "L1_minimo",
                "L2_constantes",
                "L3_contexto",
                "L4_completo",
                "delta_L1_to_L4",
            ]
            lines.append(markdown_table(changed[cols]))

    failed = results[results["ok"] != True]  # noqa: E712
    lines.extend(["", "## Fallos de ejecucion", ""])
    if failed.empty:
        lines.append("No hubo fallos de ejecucion.")
    else:
        lines.append(markdown_table(failed[["case_id", "info_level", "error"]]))

    ok = results[results["ok"] == True].copy()  # noqa: E712
    if "discrepancia_severa" in ok:
        severe = ok[ok["discrepancia_severa"] == True].copy()  # noqa: E712
        lines.extend(["", "## Discrepancias severas", ""])
        if severe.empty:
            lines.append("No hubo discrepancias severas.")
        else:
            cols = [
                "original_case_id",
                "info_level",
                "familia_clinica",
                "esi_referencia",
                "clase_predicha",
                "confianza",
                "foco_auditoria",
            ]
            lines.append(markdown_table(severe[cols].head(80)))

    path.write_text("\n".join(lines), encoding="utf-8")


def write_errors_report(path: Path, results: pd.DataFrame, transitions: pd.DataFrame) -> None:
    lines = [
        "# Patrones de error por cantidad de informacion",
        "",
        "La comparacion ayuda a separar errores por falta de datos de errores persistentes del modelo.",
        "",
    ]

    ok = results[results["ok"] == True].copy()  # noqa: E712
    extraction_flags = ok[
        (ok["symptom_recall"] < 0.5)
        | (ok["scalar_match_rate"] < 1.0)
        | (ok["n_alertas_validador"] > 0)
    ].copy()
    lines.extend(["## Extracciones a revisar", ""])
    if extraction_flags.empty:
        lines.append("No se detectaron extracciones llamativas.")
    else:
        cols = [
            "original_case_id",
            "info_level",
            "familia_clinica",
            "foco_auditoria",
            "scalar_match_rate",
            "symptom_recall",
            "symptom_missing",
            "symptom_extra",
            "alertas_validador",
        ]
        lines.append(markdown_table(extraction_flags[cols].head(80)))

    if "discrepancia_severa" in ok:
        persistent = (
            ok.groupby(["original_case_id", "familia_clinica", "foco_auditoria"], as_index=False)
            .agg(
                niveles_con_discrepancia=("discrepancia_severa", "sum"),
                peor_delta=("abs_delta_vs_referencia", "max"),
                media_delta_abs=("abs_delta_vs_referencia", "mean"),
            )
            .sort_values(
                ["niveles_con_discrepancia", "peor_delta", "media_delta_abs"],
                ascending=[False, False, False],
            )
        )
        lines.extend(["", "## Fallos persistentes del modelo", ""])
        lines.append(markdown_table(persistent.head(20)))

    if not transitions.empty:
        worsened = transitions[transitions["empeora_L1_to_L4"] == True].copy()  # noqa: E712
        improved = transitions[transitions["mejora_L1_to_L4"] == True].copy()  # noqa: E712
        lines.extend(["", "## Casos que mejoran al completar la nota", ""])
        lines.append(
            "No hay mejoras L1-L4."
            if improved.empty
            else markdown_table(
                improved[
                    [
                        "original_case_id",
                        "familia_clinica",
                        "foco_auditoria",
                        "L1_minimo",
                        "L4_completo",
                        "delta_L1_to_L4",
                    ]
                ]
            )
        )
        lines.extend(["", "## Casos que empeoran al completar la nota", ""])
        lines.append(
            "No hay empeoramientos L1-L4."
            if worsened.empty
            else markdown_table(
                worsened[
                    [
                        "original_case_id",
                        "familia_clinica",
                        "foco_auditoria",
                        "L1_minimo",
                        "L4_completo",
                        "delta_L1_to_L4",
                    ]
                ]
            )
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_nursing_checklist(path: Path, summary: pd.DataFrame, transitions: pd.DataFrame) -> None:
    best_level = ""
    if not summary.empty and "mean_abs_delta_vs_reference" in summary:
        best = summary.sort_values("mean_abs_delta_vs_reference").iloc[0]
        best_level = f"En esta auditoria, el nivel con menor error ordinal medio fue `{best['info_level']}`."

    lines = [
        "# Checklist para escribir la narrativa de triaje",
        "",
        "Guia derivada de la auditoria de niveles de informacion con `v3_final`.",
        "No sustituye criterio clinico ni convierte el prototipo en herramienta asistencial real.",
        "",
        best_level,
        "",
        "## Datos imprescindibles",
        "",
        "- Edad y sexo.",
        "- Motivo principal en una frase concreta.",
        "- Tiempo de evolucion.",
        "- TA, FC, FR, SatO2 y temperatura.",
        "- Dolor 0-10 si hay dolor.",
        "- Estado mental: orientado, confuso, somnoliento o Glasgow si se conoce.",
        "",
        "## Datos que conviene anadir",
        "",
        "- Disnea, dolor toracico, sangrado, sincope, focalidad neurologica o convulsion si aparecen.",
        "- Negaciones relevantes: sin disnea, sin fiebre, sin dolor toracico, sin perdida de conciencia.",
        "- Antecedentes que cambian riesgo: EPOC, diabetes, embarazo, cirrosis, epilepsia, anticoagulacion.",
        "- Medicacion critica: anticoagulantes, insulina, beta-bloqueantes, inhaladores, benzodiacepinas.",
        "",
        "## Formato recomendado",
        "",
        "```text",
        "Mujer 72a. Motivo: disnea desde ayer. EPOC. TA 118/70, FC 110, FR 28, SatO2 89%, T 38.7, Glasgow 15. Niega dolor toracico. Usa inhaladores.",
        "```",
        "",
        "## Ejemplos de calidad de nota",
        "",
        "Pobre:",
        "",
        "```text",
        "Mujer 72a. Motivo: disnea.",
        "```",
        "",
        "Aceptable:",
        "",
        "```text",
        "Mujer 72a. Motivo: disnea. TA 118/70, FC 110, FR 28, SatO2 89%, T 38.7.",
        "```",
        "",
        "Optima:",
        "",
        "```text",
        "Mujer 72a. Motivo: fiebre, tos y disnea progresiva. EPOC. TA 118/70, FC 110, FR 28, SatO2 89%, T 38.7, Glasgow 15. Usa inhaladores.",
        "```",
        "",
        "## Regla practica",
        "",
        "La nota debe permitir reconstruir gravedad fisiologica y contexto clinico sin inferir datos. Si un dato no se midio, no lo inventes; si se midio, escribelo.",
    ]
    path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")


def write_cases(path: Path, cases: list[InformationLevelCase]) -> None:
    path.write_text(
        json.dumps([asdict(case) for case in cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita si v3_final funciona mejor con notas de triaje mas completas."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cases-only", action="store_true")
    parser.add_argument("--no-predict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "cases": out_dir / "audit_cases_information_levels.json",
        "results_csv": out_dir / "audit_results.csv",
        "results_parquet": out_dir / "audit_results.parquet",
        "summary_csv": out_dir / "audit_summary.csv",
        "summary_json": out_dir / "audit_summary.json",
        "cache": out_dir / "audit_cache.jsonl",
        "report": out_dir / "audit_summary.md",
        "errors": out_dir / "audit_errors.md",
        "checklist": out_dir / "nursing_checklist.md",
        "transitions": out_dir / "audit_transitions.csv",
    }

    if args.force and paths["cache"].exists():
        paths["cache"].unlink()

    cases = build_information_cases(args.limit)
    write_cases(paths["cases"], cases)
    if args.cases_only:
        print(f"Casos guardados en: {paths['cases']}")
        return

    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    predictor = None if args.no_predict else TriajePredictor()
    cache = load_cache(paths["cache"])
    results: list[dict[str, Any]] = []

    total = len(cases)
    for current, case in enumerate(cases, start=1):
        cache_key = f"{PROMPT_ID}::{case.case_id}"
        if cache_key in cache:
            row = cache[cache_key]
            print(f"{current:03d}/{total} {case.case_id} cache")
        else:
            print(f"{current:03d}/{total} {case.case_id}")
            row = process_case(
                case,
                prompt_text=prompt_text,
                model=args.model,
                predictor=predictor,
            )
            append_jsonl(paths["cache"], row)
        results.append(row)

    results_df = pd.DataFrame(results)
    summary_df = build_summary(results_df)
    transitions_df = build_transition_summary(results_df)

    results_df.to_csv(paths["results_csv"], index=False, encoding="utf-8")
    results_df.to_parquet(paths["results_parquet"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False, encoding="utf-8")
    transitions_df.to_csv(paths["transitions"], index=False, encoding="utf-8")
    paths["summary_json"].write_text(
        json.dumps(summary_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_report(paths["report"], summary_df, transitions_df, results_df)
    write_errors_report(paths["errors"], results_df, transitions_df)
    write_nursing_checklist(paths["checklist"], summary_df, transitions_df)

    print(f"\nResultados guardados en: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
