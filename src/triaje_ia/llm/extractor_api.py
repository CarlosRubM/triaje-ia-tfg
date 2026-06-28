"""
src/triaje_ia/llm/extractor_api.py
────────────────────────────────────
Extractor de VectorClinico via API cloud (Groq / OpenAI compatible).

Usa la misma interfaz que extractor.py (Ollama local) pero conecta
con APIs HTTP para despliegues en cloud donde no hay GPU local.

Groq es gratuito y ofrece ~1-2s de latencia con llama-3.3-70b-versatile.
"""

import os

from loguru import logger

from triaje_ia.llm.extractor import SYSTEM_PROMPT, _completar_datos_explicitos
from triaje_ia.llm.normalizer import normalizar_vector_clinico
from triaje_ia.llm.schemas import VectorClinico


def extraer_vector_clinico_api(
    narrativa: str,
    modelo: str = "llama-3.3-70b-versatile",
    provider: str = "groq",
) -> VectorClinico:
    """
    Extrae VectorClinico usando una API cloud (Groq por defecto).

    Requiere variable de entorno GROQ_API_KEY (o OPENAI_API_KEY
    si provider='openai').

    Args:
        narrativa: historia clínica en texto libre.
        modelo: modelo a usar en la API.
        provider: 'groq' u 'openai'.

    Returns:
        VectorClinico validado por Pydantic.

    Raises:
        EnvironmentError: si no hay API key configurada.
        ValidationError: si el JSON no cumple el esquema.
    """
    if provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY", "")
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = "https://api.openai.com/v1"
    else:
        raise ValueError(f"Provider no soportado: {provider}")

    if not api_key:
        raise EnvironmentError(
            f"No se encontró la API key para {provider}. "
            f"Configura la variable de entorno "
            f"{'GROQ_API_KEY' if provider == 'groq' else 'OPENAI_API_KEY'}."
        )

    # Importar httpx bajo demanda para no forzar dependencia
    try:
        import httpx
    except ImportError:
        raise ImportError(
            "httpx es necesario para el extractor API. "
            "Instálalo con: uv add httpx"
        )

    logger.info(f"Extracción API: {provider} / {modelo}")

    schema = VectorClinico.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "VectorClinico",
            "schema": schema,
            "strict": True,
        },
    }

    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Narrativa:\n{narrativa}"},
        ],
        "temperature": 0.0,
        "response_format": response_format,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30.0,
    )

    if hasattr(resp, "status_code") and resp.status_code != 200:
        raise ConnectionError(
            f"API {provider} devolvió {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    vector = normalizar_vector_clinico(VectorClinico.model_validate_json(content))
    vector = normalizar_vector_clinico(_completar_datos_explicitos(narrativa, vector))
    logger.success(
        f"Extracción API completada: {len(vector.sintomas_presentes)} síntomas"
    )
    return vector
