"""Run a gold-case LLM audit for clinical extraction and feature conversion.

The deterministic adapter audit verifies the conversion layer. This script adds
the variable layer that depends on the active LLM backend: it sends synthetic
clinical narratives, compares the extracted VectorClinico against an explicit
gold contract, then checks selected derived features.

It does not tune prompts, thresholds or models. Failures are reported as audit
findings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("No se pudo localizar la raiz del proyecto.")


PROJECT_ROOT = find_project_root(Path.cwd()).resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from triaje_ia.inference.adapter import vectorclinico_a_features  # noqa: E402
from triaje_ia.llm.factory import crear_extractor  # noqa: E402
from triaje_ia.llm.schemas import VectorClinico  # noqa: E402
from triaje_ia.llm.validator import validar_vector_clinico  # noqa: E402


DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
OUT_DIR = PROJECT_ROOT / "reports" / "variable_extraction_audit" / "llm_gold"

SCALAR_FIELDS = [
    "edad",
    "sexo",
    "presion_sistolica",
    "presion_diastolica",
    "frecuencia_cardiaca",
    "frecuencia_respiratoria",
    "saturacion_oxigeno",
    "temperatura",
    "nivel_dolor",
    "duracion_sintomas",
    "metodo_llegada",
]


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    title: str
    focus: str
    narrative: str
    expected_scalars: dict[str, Any]
    expected_contains: dict[str, list[str]] = field(default_factory=dict)
    expected_absent: dict[str, list[str]] = field(default_factory=dict)
    expected_features: dict[str, Any] = field(default_factory=dict)


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _norm(value: object) -> str:
    return " ".join(_strip_accents(str(value).lower()).split())


def _list_norm(values: list[str]) -> list[str]:
    return [_norm(value) for value in values]


def _contains_term(values: list[str], expected: str) -> bool:
    expected_norm = _norm(expected)
    return any(expected_norm in value or value in expected_norm for value in _list_norm(values))


def _compare_scalar(actual: Any, expected: Any) -> tuple[bool, str]:
    if expected is None:
        return actual is None, f"expected None, got {actual!r}"
    if isinstance(expected, float):
        if actual is None:
            return False, f"expected {expected}, got None"
        return abs(float(actual) - expected) <= 0.2, f"expected {expected}, got {actual!r}"
    return actual == expected, f"expected {expected!r}, got {actual!r}"


def build_cases() -> list[GoldCase]:
    return [
        GoldCase(
            case_id="G001",
            title="ambulance chest pain",
            focus="arrival ambulance, vitals, pain, cardiac medication",
            narrative=(
                "Varon 58 anos traido por SAMU en ambulancia. Dolor toracico "
                "opresivo 9/10 desde hace 45 minutos, sudoracion y nauseas. "
                "TA 150/92, FC 108, FR 22, SatO2 94%, T 36.8. Antecedentes HTA. "
                "Toma bisoprolol y Sintrom."
            ),
            expected_scalars={
                "edad": 58,
                "sexo": "M",
                "presion_sistolica": 150,
                "presion_diastolica": 92,
                "frecuencia_cardiaca": 108,
                "frecuencia_respiratoria": 22,
                "saturacion_oxigeno": 94.0,
                "temperatura": 36.8,
                "nivel_dolor": 9,
                "metodo_llegada": "ambulancia",
            },
            expected_contains={
                "sintomas_presentes": ["chest pain", "nausea"],
                "patologias_previas": ["hipertension"],
                "medicacion_habitual": [
                    "beta blockers cardiac selective",
                    "anticoagulants - coumarin",
                ],
            },
            expected_features={
                "llegada_ambulancia": 1,
                "llegada_desconocida": 0,
                "cc_dolor_toracico": 1,
                "med_anticoagulante": 1,
                "med_betabloqueante": 1,
                "hx_cardiaco": 1,
            },
        ),
        GoldCase(
            case_id="G002",
            title="helicopter trauma",
            focus="arrival helicopter, trauma, shock features",
            narrative=(
                "Mujer 34 anos trasladada en helicoptero medicalizado por accidente "
                "de moto. Dolor abdominal y herida sangrante en muslo. TA 84/50, "
                "FC 142, FR 30, SatO2 90%, T 35.9, dolor 8/10."
            ),
            expected_scalars={
                "edad": 34,
                "sexo": "F",
                "presion_sistolica": 84,
                "presion_diastolica": 50,
                "frecuencia_cardiaca": 142,
                "frecuencia_respiratoria": 30,
                "saturacion_oxigeno": 90.0,
                "temperatura": 35.9,
                "nivel_dolor": 8,
                "metodo_llegada": "helicoptero",
            },
            expected_contains={"sintomas_presentes": ["abdominal pain", "bleeding"]},
            expected_features={
                "llegada_helicoptero": 1,
                "llegada_desconocida": 0,
                "cc_trauma": 1,
                "cc_gi_agudo": 1,
                "hipotension": 1,
                "shock_index_alto": 1,
                "vitales_criticos": 1,
            },
        ),
        GoldCase(
            case_id="G003",
            title="walk in green zone",
            focus="arrival autonomous, negative medication, green zone",
            narrative=(
                "Varon 25 anos acude por sus medios por dolor de oido desde ayer. "
                "Niega fiebre, niega dolor toracico y niega disnea. Sin medicacion "
                "habitual y sin antecedentes. TA 118/74, FC 76, FR 16, SatO2 99%, "
                "T 36.6, dolor 3/10."
            ),
            expected_scalars={
                "edad": 25,
                "sexo": "M",
                "presion_sistolica": 118,
                "presion_diastolica": 74,
                "frecuencia_cardiaca": 76,
                "frecuencia_respiratoria": 16,
                "saturacion_oxigeno": 99.0,
                "temperatura": 36.6,
                "nivel_dolor": 3,
                "metodo_llegada": "autonomo",
            },
            expected_contains={"sintomas_presentes": ["ear pain"]},
            expected_absent={
                "sintomas_presentes": ["chest pain", "dyspnea", "fever"],
                "medicacion_habitual": ["medicacion", "unknown"],
            },
            expected_features={
                "llegada_autonoma": 1,
                "llegada_desconocida": 0,
                "sin_medicacion": 1,
                "zona_verde": 1,
                "cc_dolor_toracico": 0,
                "cc_disnea": 0,
                "cc_infeccioso": 0,
            },
        ),
        GoldCase(
            case_id="G004",
            title="unknown arrival sepsis",
            focus="arrival unknown only when not stated",
            narrative=(
                "Mujer 81 anos de residencia, no consta metodo de llegada. Fiebre, "
                "decaimiento y confusion. TA 86/42, FC 124, FR 28, SatO2 89%, "
                "T 39.3. Antecedente de insuficiencia renal cronica."
            ),
            expected_scalars={
                "edad": 81,
                "sexo": "F",
                "presion_sistolica": 86,
                "presion_diastolica": 42,
                "frecuencia_cardiaca": 124,
                "frecuencia_respiratoria": 28,
                "saturacion_oxigeno": 89.0,
                "temperatura": 39.3,
                "metodo_llegada": "desconocido",
            },
            expected_contains={
                "sintomas_presentes": ["fever", "altered mental status"],
                "patologias_previas": ["insuficiencia renal"],
            },
            expected_features={
                "llegada_desconocida": 1,
                "cc_infeccioso": 1,
                "cc_neuro_ams": 1,
                "fiebre": 1,
                "o2sat_bajo_92": 1,
                "hx_metabolico_renal": 1,
            },
        ),
        GoldCase(
            case_id="G005",
            title="respiratory medication and inhaled steroid exclusion",
            focus="inhaled respiratory therapy must not become systemic steroid",
            narrative=(
                "Mujer 69 anos con EPOC acude en ambulancia por disnea y tos. "
                "Tratamiento habitual con salbutamol inhalado y corticoide inhalado. "
                "TA 132/78, FC 96, FR 24, SatO2 91%, T 37.2, dolor 0/10."
            ),
            expected_scalars={
                "edad": 69,
                "sexo": "F",
                "presion_sistolica": 132,
                "presion_diastolica": 78,
                "frecuencia_cardiaca": 96,
                "frecuencia_respiratoria": 24,
                "saturacion_oxigeno": 91.0,
                "temperatura": 37.2,
                "nivel_dolor": 0,
                "metodo_llegada": "ambulancia",
            },
            expected_contains={
                "sintomas_presentes": ["dyspnea", "cough"],
                "patologias_previas": ["EPOC"],
            },
            expected_features={
                "llegada_ambulancia": 1,
                "cc_disnea": 1,
                "o2sat_bajo_92": 1,
                "hx_respiratorio": 1,
                "med_corticoide_sistemico": 0,
            },
        ),
        GoldCase(
            case_id="G006",
            title="opioid benzodiazepine respiratory risk",
            focus="compound medication risk",
            narrative=(
                "Varon 52 anos traido por familiar en coche propio por somnolencia. "
                "Toma oxicodona cronica y diazepam. TA 112/68, FC 58, FR 9, "
                "SatO2 88%, T 36.0. Responde a estimulos dolorosos."
            ),
            expected_scalars={
                "edad": 52,
                "sexo": "M",
                "presion_sistolica": 112,
                "presion_diastolica": 68,
                "frecuencia_cardiaca": 58,
                "frecuencia_respiratoria": 9,
                "saturacion_oxigeno": 88.0,
                "temperatura": 36.0,
                "metodo_llegada": "autonomo",
            },
            expected_contains={
                "sintomas_presentes": ["altered mental status"],
                "medicacion_habitual": [
                    "analgesic opioid",
                    "benzodiazepines",
                ],
            },
            expected_features={
                "llegada_autonoma": 1,
                "cc_neuro_ams": 1,
                "med_opiaceo": 1,
                "med_benzodiacepina": 1,
                "riesgo_depresion_resp": 1,
            },
        ),
        GoldCase(
            case_id="G007",
            title="female with diabetes and insulin",
            focus="sex, metabolic history, insulin",
            narrative=(
                "Mujer 76 anos acude caminando por mareo y sudoracion. Diabetes "
                "mellitus tipo 2 en tratamiento con insulina. TA 118/70, FC 104, "
                "FR 22, SatO2 96%, T 36.1, glucemia 42 mg/dl."
            ),
            expected_scalars={
                "edad": 76,
                "sexo": "F",
                "presion_sistolica": 118,
                "presion_diastolica": 70,
                "frecuencia_cardiaca": 104,
                "frecuencia_respiratoria": 22,
                "saturacion_oxigeno": 96.0,
                "temperatura": 36.1,
                "metodo_llegada": "autonomo",
            },
            expected_contains={
                "sintomas_presentes": ["dizziness", "hypoglycemia"],
                "patologias_previas": ["diabetes"],
                "medicacion_habitual": ["insulin analogs"],
            },
            expected_features={
                "anciano": 1,
                "anciano_mayor": 1,
                "med_insulina": 1,
                "hx_metabolico_renal": 1,
            },
        ),
        GoldCase(
            case_id="G008",
            title="other sex administrative request",
            focus="all sex values, administrative low acuity, missing vitals",
            narrative=(
                "Persona adulta de 40 anos, sexo biologico no especificado. Solicita "
                "renovacion de receta de medicacion habitual. Sin sintomas agudos. "
                "No se registran constantes en triaje."
            ),
            expected_scalars={
                "edad": 40,
                "sexo": "Otro",
                "presion_sistolica": None,
                "presion_diastolica": None,
                "frecuencia_cardiaca": None,
                "frecuencia_respiratoria": None,
                "saturacion_oxigeno": None,
                "temperatura": None,
                "nivel_dolor": None,
                "metodo_llegada": "desconocido",
            },
            expected_contains={"sintomas_presentes": ["medication refill"]},
            expected_features={
                "temperature_missing": 1,
                "o2sat_missing": 1,
                "pain_missing": 1,
                "llegada_desconocida": 1,
            },
        ),
    ]


def compare_case(case: GoldCase, vector: VectorClinico) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    data = vector.model_dump()

    for field_name, expected in case.expected_scalars.items():
        ok, detail = _compare_scalar(data.get(field_name), expected)
        if not ok:
            failures.append(
                {
                    "type": "scalar",
                    "field": field_name,
                    "detail": detail,
                }
            )

    for field_name, expected_terms in case.expected_contains.items():
        actual_values = data.get(field_name) or []
        for expected in expected_terms:
            if not _contains_term(actual_values, expected):
                failures.append(
                    {
                        "type": "missing_list_term",
                        "field": field_name,
                        "detail": f"expected term {expected!r}, got {actual_values!r}",
                    }
                )

    for field_name, absent_terms in case.expected_absent.items():
        actual_values = data.get(field_name) or []
        for absent in absent_terms:
            if _contains_term(actual_values, absent):
                failures.append(
                    {
                        "type": "forbidden_list_term",
                        "field": field_name,
                        "detail": f"forbidden term {absent!r}, got {actual_values!r}",
                    }
                )

    features = vectorclinico_a_features(vector).iloc[0].to_dict()
    for feature_name, expected in case.expected_features.items():
        actual = features.get(feature_name)
        ok, detail = _compare_scalar(actual, expected)
        if not ok:
            failures.append(
                {
                    "type": "feature",
                    "field": feature_name,
                    "detail": detail,
                }
            )

    llegada_flags = [
        "llegada_autonoma",
        "llegada_ambulancia",
        "llegada_helicoptero",
        "llegada_desconocida",
    ]
    if int(sum(features[flag] for flag in llegada_flags)) != 1:
        failures.append(
            {
                "type": "feature_exclusion",
                "field": "llegada_*",
                "detail": f"arrival flags are not exclusive: { {k: features[k] for k in llegada_flags} }",
            }
        )

    return failures


def run_audit(
    *,
    backend: str,
    model: str,
    max_cases: int | None = None,
) -> dict[str, Any]:
    extractor = crear_extractor(backend)
    cases = build_cases()
    if max_cases is not None:
        cases = cases[:max_cases]

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "title": case.title,
            "focus": case.focus,
            "narrative": case.narrative,
        }
        try:
            vector = extractor(case.narrative, modelo=model)
            elapsed = time.perf_counter() - started
            failures = compare_case(case, vector)
            validator_alerts = validar_vector_clinico(vector, case.narrative)
            row.update(
                {
                    "ok": len(failures) == 0,
                    "error": "",
                    "elapsed_seconds": round(elapsed, 3),
                    "failures": failures,
                    "n_failures": len(failures),
                    "validator_alerts": [str(alert) for alert in validator_alerts],
                    "vector": vector.model_dump(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "failures": [
                        {
                            "type": "exception",
                            "field": "extractor",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                    "n_failures": 1,
                    "validator_alerts": [],
                    "vector": {},
                }
            )
        rows.append(row)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "model": model,
        "n_cases": len(rows),
        "n_passed": sum(1 for row in rows if row["ok"]),
        "n_failed": sum(1 for row in rows if not row["ok"]),
        "passed": all(row["ok"] for row in rows),
        "rows": rows,
    }


def write_reports(payload: dict[str, Any]) -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "llm_gold_audit.json"
    jsonl_path = OUT_DIR / "llm_gold_audit_cases.jsonl"
    md_path = OUT_DIR / "llm_gold_audit.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in payload["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Auditoria LLM con casos oro",
        "",
        f"Generado: {payload['generated_at']}",
        f"Backend: `{payload['backend']}`",
        f"Modelo: `{payload['model']}`",
        f"Casos: {payload['n_cases']}",
        f"Pasados: {payload['n_passed']}",
        f"Fallidos: {payload['n_failed']}",
        f"Estado: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "## Casos",
        "",
    ]
    for row in payload["rows"]:
        status = "PASS" if row["ok"] else "FAIL"
        lines.append(f"### {row['case_id']} - {row['title']} - {status}")
        lines.append("")
        lines.append(f"Foco: {row['focus']}")
        lines.append(f"Tiempo: {row['elapsed_seconds']} s")
        if row["error"]:
            lines.append(f"Error: `{row['error']}`")
        if row["failures"]:
            lines.append("")
            lines.append("Fallos:")
            for failure in row["failures"]:
                lines.append(
                    f"- `{failure['type']}` `{failure['field']}`: {failure['detail']}"
                )
        if row["validator_alerts"]:
            lines.append("")
            lines.append("Alertas validador:")
            for alert in row["validator_alerts"]:
                lines.append(f"- {alert}")
        lines.append("")

    lines.extend(
        [
            "## Nota metodologica",
            "",
            "Estos casos son sinteticos y sirven para auditar extraccion, no para "
            "seleccionar modelo, umbrales, calibracion ni politica A1.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path, jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="ollama", choices=["ollama", "api"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any gold case fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_audit(
        backend=args.backend,
        model=args.model,
        max_cases=args.max_cases,
    )
    md_path, json_path, jsonl_path = write_reports(payload)
    print(f"LLM gold audit {'PASS' if payload['passed'] else 'FAIL'}")
    print(md_path)
    print(json_path)
    print(jsonl_path)
    if args.strict and not payload["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
