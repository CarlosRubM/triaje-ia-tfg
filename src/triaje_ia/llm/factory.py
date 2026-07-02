"""
src/triaje_ia/llm/factory.py
──────────────────────────────
Factory del extractor local de VectorClinico.

Uso:
    from triaje_ia.llm.factory import crear_extractor
    extraer = crear_extractor("ollama")
    vector = extraer(narrativa, modelo="qwen2.5")
"""

from __future__ import annotations

from typing import Callable

from loguru import logger

from triaje_ia.llm.schemas import VectorClinico


def crear_extractor(
    backend: str = "ollama",
) -> Callable[..., VectorClinico]:
    """
    Devuelve la función de extracción local con Ollama.

    Args:
        backend: debe ser "ollama".

    Returns:
        Callable con firma (narrativa, modelo=...) -> VectorClinico
    """
    if backend != "ollama":
        raise ValueError(f"Backend no reconocido: '{backend}'. Usa 'ollama'.")

    from triaje_ia.llm.extractor import extraer_vector_clinico

    logger.info("Backend LLM: Ollama (local)")
    return extraer_vector_clinico
