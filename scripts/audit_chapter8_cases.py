# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ollama
import pandas as pd


def find_project_root(start: Path) -> Path:
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("No se pudo localizar la raiz del proyecto.")


PROJECT_ROOT = find_project_root(Path.cwd()).resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from triaje_ia.inference.predictor import TriajePredictor
from triaje_ia.llm.extractor import (  # noqa: E402
    SYSTEM_PROMPT,
    _completar_datos_explicitos,
)
from triaje_ia.llm.normalizer import normalizar_vector_clinico  # noqa: E402
from triaje_ia.llm.schemas import VectorClinico  # noqa: E402
from triaje_ia.llm.validator import validar_vector_clinico  # noqa: E402
from triaje_ia.ui.clinical_form import (  # noqa: E402
    TriageFormData,
    entrada_minima_completa,
    generar_narrativa_triaje_texto_libre,
    validar_entrada_triaje,
)


DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
OUT_DIR = PROJECT_ROOT / "reports" / "chapter8_case_studies"


@dataclass(frozen=True)
class Chapter8Case:
    case_id: str
    title: str
    intended_profile: str
    form: TriageFormData
    source_note: str


def _case(
    case_id: str,
    title: str,
    intended_profile: str,
    form: TriageFormData,
) -> Chapter8Case:
    return Chapter8Case(
        case_id=case_id,
        title=title,
        intended_profile=intended_profile,
        form=form,
        source_note=(
            "Caso sintetico demostrativo del capitulo 8; no procede de pacientes "
            "reales ni de historias clinicas externas."
        ),
    )


def build_cases() -> list[Chapter8Case]:
    return [
        _case(
            "caso_1",
            "Compromiso respiratorio y cardiovascular",
            "Prioridad maxima esperada por alteracion respiratoria y hemodinamica.",
            TriageFormData(
                edad=78,
                sexo="M",
                metodo_llegada="ambulancia",
                motivo_consulta="Disnea intensa y dolor toracico",
                duracion_sintomas="6 horas",
                sintomas_adicionales=(
                    "Dolor toracico opresivo, sudoracion y mal estado general"
                ),
                presion_sistolica=92,
                presion_diastolica=58,
                frecuencia_cardiaca=118,
                frecuencia_respiratoria=30,
                saturacion_oxigeno=88.0,
                temperatura=37.8,
                nivel_dolor=8,
                antecedentes=(
                    "Insuficiencia cardiaca, diabetes tipo 2, fibrilacion auricular"
                ),
                medicacion=(
                    "Anticoagulante, furosemida, betabloqueante, insulina"
                ),
            ),
        ),
        _case(
            "caso_2",
            "Dolor abdominal con vomitos y estrenimiento",
            "Prioridad intermedia esperada por sintomas abdominales y recursos probables.",
            TriageFormData(
                edad=34,
                sexo="F",
                metodo_llegada="autonomo",
                motivo_consulta=(
                    "Dolor abdominal generalizado con vomitos y estrenimiento"
                ),
                duracion_sintomas="Desde esta manana",
                sintomas_adicionales=(
                    "Dolor abdominal generalizado, vomitos y estrenimiento. "
                    "Antecedente de laminectomia. Ultima regla dentro de los "
                    "ultimos 28 dias."
                ),
                presion_sistolica=132,
                presion_diastolica=80,
                frecuencia_cardiaca=102,
                frecuencia_respiratoria=16,
                saturacion_oxigeno=99.0,
                temperatura=36.5,
                nivel_dolor=6,
                antecedentes="Laminectomia",
                medicacion="",
            ),
        ),
        _case(
            "caso_3",
            "Ojo rojo y picor ocular",
            "Baja prioridad esperada por cuadro leve, sin signos de alarma y constantes normales.",
            TriageFormData(
                edad=28,
                sexo="F",
                metodo_llegada="autonomo",
                motivo_consulta="Ojo rojo y picor ocular",
                duracion_sintomas="2 dias",
                sintomas_adicionales=(
                    "Ojo rojo con picor y lagrimeo leve. Sin dolor intenso, "
                    "sin perdida de vision, sin traumatismo y sin fiebre."
                ),
                presion_sistolica=116,
                presion_diastolica=72,
                frecuencia_cardiaca=74,
                frecuencia_respiratoria=16,
                saturacion_oxigeno=99.0,
                temperatura=36.4,
                nivel_dolor=1,
                antecedentes="",
                medicacion="",
            ),
        ),
    ]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _call_ollama_raw(narrativa: str, model: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Narrativa:\n{narrativa}"},
        ],
        format=VectorClinico.model_json_schema(),
        options={"temperature": 0.0},
    )
    elapsed = time.perf_counter() - t0
    return response.message.content, elapsed


def _extract_with_evidence(case: Chapter8Case, model: str) -> dict[str, Any]:
    narrativa = generar_narrativa_triaje_texto_libre(case.form)
    input_warnings = validar_entrada_triaje(case.form)

    raw_json, llm_seconds = _call_ollama_raw(narrativa, model)

    t0 = time.perf_counter()
    vector_pydantic = VectorClinico.model_validate_json(raw_json)
    pydantic_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    vector_normalized = normalizar_vector_clinico(vector_pydantic)
    vector_repaired = normalizar_vector_clinico(
        _completar_datos_explicitos(narrativa, vector_normalized)
    )
    normalization_seconds = time.perf_counter() - t0

    alerts = validar_vector_clinico(vector_repaired, narrativa)

    return {
        "case_id": case.case_id,
        "title": case.title,
        "intended_profile": case.intended_profile,
        "source_note": case.source_note,
        "input": asdict(case.form),
        "entrada_minima_completa": entrada_minima_completa(case.form),
        "input_warnings": input_warnings,
        "narrativa": narrativa,
        "llm_raw_content": raw_json,
        "llm_raw_json": json.loads(raw_json),
        "vector_pydantic": vector_pydantic.model_dump(),
        "vector_normalized": vector_normalized.model_dump(),
        "vector_final_review": vector_repaired.model_dump(),
        "review_changes": [],
        "validation_alerts": [
            {
                "campo": alert.campo,
                "mensaje": alert.mensaje,
                "nivel": alert.nivel.value,
                "sugerencia": alert.sugerencia,
            }
            for alert in alerts
        ],
        "latency": {
            "llm_seconds": llm_seconds,
            "pydantic_seconds": pydantic_seconds,
            "normalization_repair_seconds": normalization_seconds,
        },
    }


def _run_prediction(
    evidence: dict[str, Any],
    predictor: TriajePredictor,
    include_shap: bool,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    vector = VectorClinico.model_validate(evidence["vector_final_review"])
    narrativa = evidence["narrativa"]

    t0 = time.perf_counter()
    result = predictor.predict(vector, narrativa)
    prediction_seconds = time.perf_counter() - t0

    probas = [float(x) for x in result.probas]
    prediction = {
        "case_id": evidence["case_id"],
        "clase_predicha": int(result.clase_predicha),
        "confianza": float(result.confianza),
        "alerta_a1_activada": bool(result.alerta_a1_activada),
        "n_features": len(result.feature_names),
        "p_acuity_1": probas[0],
        "p_acuity_2": probas[1],
        "p_acuity_3": probas[2],
        "p_acuity_4": probas[3],
        "p_acuity_5": probas[4],
        "probas_sum": float(sum(probas)),
    }

    features = result.X.copy()
    features.insert(0, "case_id", evidence["case_id"])

    shap_rows: list[dict[str, Any]] = []
    shap_seconds = 0.0
    if include_shap:
        t0 = time.perf_counter()
        explanation = predictor.explain(result)
        shap_seconds = time.perf_counter() - t0
        for direction, factors in (
            ("positiva", explanation.top_positivas),
            ("negativa", explanation.top_negativas),
        ):
            for rank, factor in enumerate(factors, start=1):
                shap_rows.append({
                    "case_id": evidence["case_id"],
                    "clase_explicada": explanation.clase_explicada,
                    "direccion": direction,
                    "rank": rank,
                    "feature": factor.nombre,
                    "valor": factor.valor,
                    "shap_value": factor.shap_value,
                    "abs_shap": abs(factor.shap_value),
                })

    latency_update = {
        "prediction_seconds": prediction_seconds,
        "shap_seconds": shap_seconds,
    }
    return prediction, features, shap_rows, latency_update


def _render_narratives_md(evidences: list[dict[str, Any]]) -> str:
    parts = ["# Narrativas estructuradas de los casos del capitulo 8", ""]
    for item in evidences:
        parts.extend([
            f"## {item['case_id']} - {item['title']}",
            "",
            "```text",
            item["narrativa"],
            "```",
            "",
        ])
    return "\n".join(parts)


def _render_report_md(
    evidences: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> str:
    pred_by_case = {row["case_id"]: row for row in predictions}
    lines = [
        "# Auditoria de casos del capitulo 8",
        "",
        "Este informe se genera ejecutando el flujo local del prototipo sobre los tres casos sinteticos usados en el capitulo 8.",
        "",
        "## Configuracion",
        "",
        f"- LLM local: `{DEFAULT_MODEL}`",
        "- Prompt: `prompts/extractor_system_v3_final.txt`",
        "- Temperatura: `0.0`",
        "- Prediccion: `TriajePredictor` con modelo activo de `models/active_model.json`",
        "- Decision: `argmax` + alerta A1 configurada en el predictor",
        "",
        "## Resumen de predicciones",
        "",
        "| Caso | Clase | P(A1) | P(A2) | P(A3) | P(A4) | P(A5) | Alerta A1 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in evidences:
        pred = pred_by_case[item["case_id"]]
        lines.append(
            "| {case} | {clase} | {p1:.4f} | {p2:.4f} | {p3:.4f} | {p4:.4f} | {p5:.4f} | {alerta} |".format(
                case=item["case_id"],
                clase=pred["clase_predicha"],
                p1=pred["p_acuity_1"],
                p2=pred["p_acuity_2"],
                p3=pred["p_acuity_3"],
                p4=pred["p_acuity_4"],
                p5=pred["p_acuity_5"],
                alerta="Si" if pred["alerta_a1_activada"] else "No",
            )
        )
    lines.extend(["", "## Trazabilidad por caso", ""])
    for item in evidences:
        pred = pred_by_case[item["case_id"]]
        lines.extend([
            f"### {item['case_id']} - {item['title']}",
            "",
            f"- Perfil previsto: {item['intended_profile']}",
            f"- Entrada minima completa: {item['entrada_minima_completa']}",
            f"- Alertas de validacion: {len(item['validation_alerts'])}",
            f"- Predictores finales usados: {pred['n_features']}",
            f"- Clase predicha: Acuity {pred['clase_predicha']}",
            f"- Probabilidad principal: {pred['confianza']:.4f}",
            f"- Alerta A1: {'Si' if pred['alerta_a1_activada'] else 'No'}",
            "",
        ])
    return "\n".join(lines)


def run(model: str, out_dir: Path, include_shap: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases()

    evidences: list[dict[str, Any]] = []
    for case in cases:
        evidences.append(_extract_with_evidence(case, model))

    t0 = time.perf_counter()
    predictor = TriajePredictor()
    predictor_load_seconds = time.perf_counter() - t0

    predictions: list[dict[str, Any]] = []
    features_frames: list[pd.DataFrame] = []
    shap_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []

    for evidence in evidences:
        prediction, features, shap_case_rows, latency_update = _run_prediction(
            evidence, predictor, include_shap=include_shap
        )
        predictions.append(prediction)
        features_frames.append(features)
        shap_rows.extend(shap_case_rows)
        latency_rows.append({
            "case_id": evidence["case_id"],
            "llm_seconds": evidence["latency"]["llm_seconds"],
            "pydantic_seconds": evidence["latency"]["pydantic_seconds"],
            "normalization_repair_seconds": evidence["latency"]["normalization_repair_seconds"],
            "predictor_load_seconds_first_run": predictor_load_seconds,
            **latency_update,
        })

    _write_json(out_dir / "chapter8_cases_inputs.json", [
        {
            "case_id": case.case_id,
            "title": case.title,
            "intended_profile": case.intended_profile,
            "source_note": case.source_note,
            "input": asdict(case.form),
        }
        for case in cases
    ])
    (out_dir / "chapter8_cases_narratives.md").write_text(
        _render_narratives_md(evidences),
        encoding="utf-8",
    )
    _write_json(out_dir / "chapter8_cases_llm_raw.json", [
        {
            "case_id": item["case_id"],
            "llm_raw_content": item["llm_raw_content"],
            "llm_raw_json": item["llm_raw_json"],
        }
        for item in evidences
    ])
    _write_json(out_dir / "chapter8_cases_vectors.json", [
        {
            "case_id": item["case_id"],
            "vector_pydantic": item["vector_pydantic"],
            "vector_normalized": item["vector_normalized"],
            "validation_alerts": item["validation_alerts"],
        }
        for item in evidences
    ])
    _write_json(out_dir / "chapter8_cases_review_vectors.json", [
        {
            "case_id": item["case_id"],
            "vector_final_review": item["vector_final_review"],
            "review_changes": item["review_changes"],
        }
        for item in evidences
    ])

    pd.concat(features_frames, ignore_index=True).to_csv(
        out_dir / "chapter8_cases_features.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(predictions).to_csv(
        out_dir / "chapter8_cases_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(shap_rows).to_csv(
        out_dir / "chapter8_cases_shap_top.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(latency_rows).to_csv(
        out_dir / "chapter8_cases_latency.csv",
        index=False,
        encoding="utf-8",
    )
    (out_dir / "chapter8_cases_report.md").write_text(
        _render_report_md(evidences, predictions),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera evidencias reproducibles para los casos del capitulo 8."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Omite SHAP si solo se desea auditar extraccion y prediccion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(model=args.model, out_dir=args.out_dir, include_shap=not args.no_shap)


if __name__ == "__main__":
    main()
