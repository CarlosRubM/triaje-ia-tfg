"""Generate a deterministic audit report for the clinical variable contract.

This script does not call the LLM and does not touch model thresholds. It
documents the current structured contract and verifies the adapter-level checks
that are stable enough to run offline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from triaje_ia.data.features import (
    FEATURES_BINARIAS,
    FEATURES_CATEGORICAS,
    FEATURES_CONTINUAS,
    TODAS_FEATURES,
)
from triaje_ia.inference.adapter import vectorclinico_a_features
from triaje_ia.llm.schemas import VectorClinico


REPORT_DIR = Path("reports/variable_extraction_audit")
MARKDOWN_PATH = REPORT_DIR / "variable_contract_audit.md"
JSON_PATH = REPORT_DIR / "variable_contract_audit.json"


def _vector(**kwargs) -> VectorClinico:
    defaults = {
        "edad": 45,
        "sexo": "M",
        "sintomas_presentes": ["checkup"],
        "patologias_previas": [],
        "medicacion_habitual": [],
        "presion_sistolica": 120,
        "presion_diastolica": 80,
        "frecuencia_cardiaca": 80,
        "frecuencia_respiratoria": 16,
        "saturacion_oxigeno": 98.0,
        "temperatura": 37.0,
        "nivel_dolor": 0,
        "metodo_llegada": "desconocido",
    }
    defaults.update(kwargs)
    return VectorClinico(**defaults)


def _row(**kwargs):
    return vectorclinico_a_features(_vector(**kwargs)).iloc[0]


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_audit() -> dict[str, object]:
    checks: list[dict[str, object]] = []

    catalog = FEATURES_CONTINUAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS
    checks.append(
        _check(
            "catalogo_88_features",
            catalog == TODAS_FEATURES and len(set(catalog)) == 88,
            f"{len(catalog)} features catalogadas",
        )
    )

    llegada_flags = [
        "llegada_autonoma",
        "llegada_ambulancia",
        "llegada_helicoptero",
        "llegada_desconocida",
    ]
    llegada_cases = {
        "ambulancia": "llegada_ambulancia",
        "helicoptero": "llegada_helicoptero",
        "autonomo": "llegada_autonoma",
        "otro": "llegada_autonoma",
        "desconocido": "llegada_desconocida",
    }
    for metodo, expected_flag in llegada_cases.items():
        row = _row(metodo_llegada=metodo)
        checks.append(
            _check(
                f"llegada_{metodo}",
                row[expected_flag] == 1 and int(row[llegada_flags].sum()) == 1,
                f"{metodo} -> {expected_flag}",
            )
        )

    row = _row(medicacion_habitual=[])
    med_cols = [c for c in TODAS_FEATURES if c.startswith("med_")]
    risk_cols = [
        "alto_riesgo_sangrado",
        "alto_riesgo_delirium",
        "riesgo_depresion_resp",
    ]
    checks.append(
        _check(
            "sin_medicacion_excluye_riesgo_farmacologico",
            row["sin_medicacion"] == 1 and int(row[med_cols + risk_cols].sum()) == 0,
            "sin medicacion mantiene todas las flags farmacologicas a cero",
        )
    )

    row = _row()
    checks.append(
        _check(
            "zona_verde_excluye_criticos",
            row["zona_verde"] == 1
            and row["n_vitales_anomalos"] == 0
            and row["vitales_criticos"] == 0
            and row["qsofa_positivo"] == 0,
            "constantes normales activan zona verde sin criticos",
        )
    )

    row = _row(frecuencia_respiratoria=22, presion_sistolica=100)
    checks.append(
        _check(
            "qsofa_dos_criterios",
            row["qsofa"] == 2 and row["qsofa_positivo"] == 1,
            "FR 22 y PAS 100 activan qSOFA positivo",
        )
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_features": len(TODAS_FEATURES),
        "features_continuas": FEATURES_CONTINUAS,
        "features_binarias": FEATURES_BINARIAS,
        "features_categoricas": FEATURES_CATEGORICAS,
        "vectorclinico_fields": list(VectorClinico.model_fields.keys()),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    return payload


def write_report(payload: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Auditoria determinista de variables",
        "",
        f"Generado: {payload['generated_at']}",
        f"Features tabulares: {payload['n_features']}",
        f"Estado: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "## Campos VectorClinico",
        "",
        *[f"- `{field}`" for field in payload["vectorclinico_fields"]],
        "",
        "## Checks",
        "",
    ]
    for check in payload["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- `{status}` {check['name']}: {check['detail']}")
    lines.extend(
        [
            "",
            "## Nota",
            "",
            "Esta auditoria no ejecuta el LLM. La validacion con LLM debe hacerse "
            "con casos oro clinicos y sin usar el conjunto test para ajustar prompts, "
            "features, umbrales o modelos.",
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_audit()
    write_report(result)
    print(f"Audit {'PASS' if result['passed'] else 'FAIL'}")
    print(MARKDOWN_PATH)
