"""
src/triaje_ia/llm/factory.py
──────────────────────────────
Factory para seleccionar el extractor de VectorClinico.

Patrón strategy: misma interfaz, dos implementaciones:
  - "ollama": extracción local con Ollama (desarrollo, privacidad)
  - "api":    extracción via Groq/OpenAI (deploy cloud, rapidez)

Uso:
    from triaje_ia.llm.factory import crear_extractor
    extraer = crear_extractor("ollama")  # o "api"
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
    Devuelve la función de extracción apropiada.

    Args:
        backend: "ollama" para local, "api" para Groq/OpenAI.

    Returns:
        Callable con firma (narrativa, modelo=...) -> VectorClinico
    """
    if backend == "ollama":
        from triaje_ia.llm.extractor import extraer_vector_clinico
        logger.info("Backend LLM: Ollama (local)")
        return extraer_vector_clinico
    elif backend == "api":
        from triaje_ia.llm.extractor_api import extraer_vector_clinico_api
        logger.info("Backend LLM: API cloud (Groq/OpenAI)")
        return extraer_vector_clinico_api
    else:
        raise ValueError(
            f"Backend no reconocido: '{backend}'. Usa 'ollama' o 'api'."
        )
