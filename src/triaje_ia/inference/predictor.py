"""
src/triaje_ia/inference/predictor.py
──────────────────────────────────────
Encapsula todo el stack ML de producción en un único objeto TriajePredictor.

Cambiar de modelo = editar models/active_model.json + copiar artefactos.
app.py y el resto del código no cambian.

Flujo:
    TriajePredictor.predict(vector, narrativa)
        └── _build_features()  →  X (DataFrame)
              ├── adapter 88f → seleccionar features tabulares finales
              ├── añadir columnas _valor si el modelo final las espera
              └── Bio_ClinicalBERT → SVD 15f
        └── clf.predict_proba()
        └── decisión final por argmax + alerta A1 si procede
        └── TriajeResult
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from triaje_ia.config import MODELS_DIR
from triaje_ia.llm.schemas import VectorClinico
from triaje_ia.ml.explicabilidad import (
    ExplicacionSHAP,
    crear_explainer,
    explicar_prediccion,
)

# ── Singleton perezoso para Bio_ClinicalBERT ────────────────────────────────
_BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
_bert_cache: dict = {}
_POLITICAS_PERMITIDAS = {"argmax", "argmax_with_a1_warning"}


def _get_bert_model() -> tuple:
    """Carga Bio_ClinicalBERT la primera vez; reutiliza el singleton después."""
    if _bert_cache:
        return _bert_cache["tokenizer"], _bert_cache["model"], _bert_cache["device"]
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(_BERT_MODEL_NAME)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _bert_cache.update({"tokenizer": tokenizer, "model": model, "device": device})
        return tokenizer, model, device
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar Bio_ClinicalBERT ({_BERT_MODEL_NAME}). "
            "Instala transformers y torch: pip install transformers torch"
        ) from exc


def _get_cls_embedding(text: str) -> np.ndarray:
    """Devuelve el vector CLS (768-dim) de Bio_ClinicalBERT para un texto."""
    import torch

    tokenizer, model, device = _get_bert_model()
    enc = tokenizer(
        [text], padding=True, truncation=True, max_length=32, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
    return out.last_hidden_state[:, 0, :].cpu().numpy()[0]  # (768,)


@dataclass(frozen=True)
class TriajeResult:
    probas: np.ndarray
    clase_predicha: int
    confianza: float
    threshold_a1_activado: bool
    X: pd.DataFrame
    feature_names: list[str]
    alertas_dominio: tuple[str, ...]


class TriajePredictor:
    """
    Predictor de triaje que carga su configuración desde active_model.json.

    Uso:
        predictor = TriajePredictor()
        result = predictor.predict(vector, narrativa)
        shap_exp = predictor.explain(result)
    """

    def __init__(self, config_path: Path = MODELS_DIR / "active_model.json"):
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        arts = cfg["artifacts"]

        self._clf = joblib.load(MODELS_DIR / arts["classifier"])
        feature_payload = json.loads(
            (MODELS_DIR / arts["feature_list"]).read_text(encoding="utf-8")
        )
        self._feature_names: list[str] = (
            feature_payload.get("features") or feature_payload["todas_features"]
        )

        threshold_payload = (
            json.loads((MODELS_DIR / arts["thresholds"]).read_text(encoding="utf-8"))
            if arts.get("thresholds")
            else None
        )
        self._decision_policy = "argmax"
        self._warning_threshold_a1 = 0.40
        self._class_weights: Optional[np.ndarray] = None
        if isinstance(threshold_payload, dict):
            self._decision_policy = str(threshold_payload.get("policy", "argmax"))
            self._warning_threshold_a1 = float(
                threshold_payload.get(
                    "warning_threshold_a1",
                    threshold_payload.get("threshold_a1", self._warning_threshold_a1),
                )
            )
            if "class_weights" in threshold_payload:
                self._class_weights = np.asarray(
                    threshold_payload["class_weights"], dtype=float
                )
            minimum_threshold_a1 = threshold_payload.get("minimum_allowed_threshold_a1")
            if minimum_threshold_a1 is not None and (
                self._warning_threshold_a1 < float(minimum_threshold_a1)
            ):
                raise ValueError(
                    "warning_threshold_a1 no puede ser menor que "
                    "minimum_allowed_threshold_a1."
                )
        elif threshold_payload is not None:
            self._class_weights = np.asarray(threshold_payload, dtype=float)
        if self._decision_policy not in _POLITICAS_PERMITIDAS:
            raise ValueError(
                "Politica de decision no soportada: "
                f"{self._decision_policy!r}. "
                f"Permitidas: {sorted(_POLITICAS_PERMITIDAS)}"
            )

        self._svd = (
            joblib.load((MODELS_DIR / arts["bert_svd"]).resolve())
            if arts.get("bert_svd")
            else None
        )
        self._assumptions: dict = (
            json.loads(
                (MODELS_DIR / arts["production_assumptions"]).read_text(
                    encoding="utf-8"
                )
            )
            if arts.get("production_assumptions")
            else {}
        )

        # Listas derivadas de feature_names (precalculadas una vez)
        self._bert_cols: list[str] = [
            f for f in self._feature_names if f.startswith("bert_svd_")
        ]
        if self._svd is not None:
            n_componentes_svd = getattr(self._svd, "n_components", None)
            if n_componentes_svd is not None and len(self._bert_cols) != int(
                n_componentes_svd
            ):
                raise ValueError(
                    "Incompatibilidad entre feature_list.json y bert_svd.joblib: "
                    f"{len(self._bert_cols)} columnas BERT frente a "
                    f"{n_componentes_svd} componentes SVD."
                )
        self._tabular_features: list[str] = [
            f for f in self._feature_names if not f.startswith("bert_svd_")
        ]
        # Medianas de imputación para las 3 cols _valor (de production_assumptions)
        self._medians: dict = self._assumptions.get("imputation_medians", {})

    def predict(self, vector: VectorClinico, narrativa: str) -> TriajeResult:
        x = self._build_features(vector, narrativa)
        probas = self._clf.predict_proba(x)[0]

        clase = int(np.argmax(probas)) + 1
        threshold_activado = bool(
            clase != 1 and probas[0] >= self._warning_threshold_a1
        )

        alertas: tuple[str, ...] = (
            (self._assumptions["mensaje_ui"],)
            if self._assumptions.get("mensaje_ui")
            else ()
        )

        return TriajeResult(
            probas=probas,
            clase_predicha=clase,
            confianza=float(probas[clase - 1]),
            threshold_a1_activado=threshold_activado,
            X=x,
            feature_names=list(x.columns),
            alertas_dominio=alertas,
        )

    def explain(
        self, result: TriajeResult, clase: Optional[int] = None
    ) -> ExplicacionSHAP:
        """Genera explicación SHAP. Llamar desde app.py en try/except."""
        target = clase if clase is not None else result.clase_predicha
        explainer = crear_explainer(self._clf, clase=target)
        return explicar_prediccion(
            explainer, result.X, clase_predicha=target, modelo=self._clf
        )

    def _build_features(self, vector: VectorClinico, narrativa: str) -> pd.DataFrame:
        """
        Construye el vector final usado por el modelo congelado:
          1. Adapter 88f → seleccionar las features tabulares finales.
          2. Añadir columnas _valor si el modelo las espera.
          3. Bio_ClinicalBERT → SVD 15 cols
          Orden final = self._feature_names.
        Si self._svd es None (modelo placeholder), devuelve solo las tabulares.
        """
        from triaje_ia.data.features import TODAS_FEATURES
        from triaje_ia.inference.adapter import (
            _celsius_a_fahrenheit,
            vectorclinico_a_features,
        )

        # ── 1. Features tabulares del adapter (88 cols) ──────────────────────
        df_88 = vectorclinico_a_features(vector)

        # Features tabulares finales que vienen del adapter.
        cols_de_adapter = [f for f in self._tabular_features if f in TODAS_FEATURES]
        df = df_88[cols_de_adapter].copy()

        # ── 2. Cols _valor (vitales continuos, imputados con medianas train) ──
        _valor_cols = {
            "temperature_valor": _celsius_a_fahrenheit(vector.temperatura),
            "heartrate_valor":   vector.frecuencia_cardiaca,
            "dbp_valor":         vector.presion_diastolica,
        }
        for col, raw_val in _valor_cols.items():
            if col in self._tabular_features:
                median = self._medians.get(col)
                df[col] = float(raw_val) if raw_val is not None else median

        # ── 3. BERT embeddings → SVD ─────────────────────────────────────────
        if self._svd is not None:
            texto = ", ".join(vector.sintomas_presentes).strip()
            if not texto and narrativa.strip():
                logging.warning(
                    "BERT usa la narrativa como fallback porque el LLM no "
                    "extrajo sintomas_presentes."
                )
                texto = narrativa.strip()
            emb_768 = _get_cls_embedding(texto)
            emb_svd = self._svd.transform(emb_768.reshape(1, -1))
            df_bert = pd.DataFrame(
                emb_svd, columns=self._bert_cols, index=df.index
            )
            df = pd.concat(
                [df.reset_index(drop=True), df_bert.reset_index(drop=True)], axis=1
            )

        return df[self._feature_names]
