import ollama
from loguru import logger

from triaje_ia.config import PROMPTS_DIR
from triaje_ia.llm.normalizer import normalizar_vector_clinico
from triaje_ia.llm.schemas import VectorClinico

SYSTEM_PROMPT = (PROMPTS_DIR / "extractor_system_v2.txt").read_text(encoding="utf-8")

def extraer_vector_clinico(narrativa: str, modelo: str = "qwen2.5") -> VectorClinico:
    """
    Convierte texto libre del paciente en un vector clínico estructurado.

    Args:
        narrativa: Descripción del caso en lenguaje natural
        modelo: Modelo Ollama a usar

    Returns:
        VectorClinico validado por Pydantic

    Raises:
        ValidationError: Si el LLM devuelve JSON incompatible con el esquema
    """
    logger.info(f"Iniciando extracción con modelo '{modelo}'")

    respuesta = ollama.chat(
        model=modelo,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Narrativa:\n{narrativa}"},
        ],
        format=VectorClinico.model_json_schema(),
        options={"temperature": 0.0},  # Determinismo total, sin creatividad
    )

    vector = normalizar_vector_clinico(
        VectorClinico.model_validate_json(respuesta.message.content)
    )
    logger.success(f"Extraídos {len(vector.sintomas_presentes)} síntomas")
    return vector
