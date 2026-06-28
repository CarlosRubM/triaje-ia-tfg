"""Presentation helpers for SHAP output in the Streamlit UI.

The model and SHAP computation stay in the ML layer. This module only translates
feature ids into readable labels and groups signed SHAP values for display.
"""

from __future__ import annotations

from dataclasses import dataclass

from triaje_ia.ui.clinical_display import display_feature_name


@dataclass(frozen=True)
class ShapFactor:
    nombre: str
    valor: float


def clean_feature_name(feature: str) -> str:
    """Return a clinical, readable label for a model feature name."""
    return display_feature_name(feature)


def split_shap_factors(
    feature_names: list[str] | tuple[str, ...],
    shap_values: list[float] | tuple[float, ...],
    *,
    top_n: int = 3,
) -> tuple[list[ShapFactor], list[ShapFactor]]:
    """Split signed SHAP values into positive and negative display factors."""
    factors = [
        ShapFactor(clean_feature_name(str(name)), float(value))
        for name, value in zip(feature_names, shap_values, strict=False)
    ]
    positives = sorted(
        (factor for factor in factors if factor.valor > 0),
        key=lambda item: abs(item.valor),
        reverse=True,
    )[:top_n]
    negatives = sorted(
        (factor for factor in factors if factor.valor < 0),
        key=lambda item: abs(item.valor),
        reverse=True,
    )[:top_n]
    return positives, negatives
