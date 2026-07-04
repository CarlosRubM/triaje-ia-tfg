import re

import ollama
from loguru import logger

from triaje_ia.config import PROMPTS_DIR
from triaje_ia.llm.normalizer import normalizar_vector_clinico
from triaje_ia.llm.schemas import VectorClinico

SYSTEM_PROMPT = (PROMPTS_DIR / "extractor_system_v3_final.txt").read_text(
    encoding="utf-8"
)


_PATRON_TENSION = re.compile(
    r"\b(?:ta|pa|tensi[oó]n|presi[oó]n arterial)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
    re.IGNORECASE,
)
_PATRON_FC = re.compile(
    r"\b(?:fc|pulso|frecuencia cardiaca)\s*[:=]?\s*(\d{2,3})\b",
    re.IGNORECASE,
)
_PATRON_FR = re.compile(
    r"\b(?:fr|frecuencia respiratoria|respiraci[oó]n)\s*[:=]?\s*(\d{1,2})\b",
    re.IGNORECASE,
)
_PATRON_SATURACION = re.compile(
    r"\b(?:sat(?:o2)?|spo2|saturaci[oó]n|ox[ií]geno)\s*[:=]?\s*(\d{2,3})(?:\s*%)?",
    re.IGNORECASE,
)
_PATRON_TEMPERATURA = re.compile(
    r"\b(?:tª|t\.?|temp(?:eratura)?)\s*[:=]?\s*(\d{2}(?:[,.]\d)?)\b",
    re.IGNORECASE,
)

_PATRON_GLUCEMIA = re.compile(
    r"\b(?:glucemia|glucose|glucosa)\D{0,20}(\d{2,3})\s*(?:mg/dl)?\b",
    re.IGNORECASE,
)
_PATRON_ESTADO_MENTAL_GUIADO = re.compile(
    r"\b(somnolient[ao]|bajo nivel de conciencia|"
    r"alteraci[oÃ³]n del nivel de consciencia|"
    r"alteraci[oÃ³]n del nivel de conciencia)\b",
    re.IGNORECASE,
)
_PATRONES_SINTOMAS_EXPLICITOS = [
    (
        re.compile(r"\b(herida sangrante|sangrado|sangrante|hemorragia)\b", re.IGNORECASE),
        "bleeding",
    ),
    (
        re.compile(r"\b(accidente|trauma|politrauma|ca[ií]da|moto)\b", re.IGNORECASE),
        "trauma",
    ),
    (
        re.compile(
            r"\b(traumatismo craneal|trauma craneal|golpe en (?:la )?cabeza|tce)\b",
            re.IGNORECASE,
        ),
        "head injury",
    ),
    (
        re.compile(r"\b(herida en cuero cabelludo|scalp wound)\b", re.IGNORECASE),
        "scalp wound",
    ),
    (
        re.compile(r"\bfractura abierta\b", re.IGNORECASE),
        "open fracture",
    ),
    (
        re.compile(r"\bbradipnea\b", re.IGNORECASE),
        "bradypnea",
    ),
    (
        re.compile(
            r"\b(somnolencia|somnoliento|obnubilad|confusi[oó]n|confuso|"
            r"responde solo al dolor|bajo nivel de consciencia)\b",
            re.IGNORECASE,
        ),
        "altered mental status",
    ),
]

_PATRONES_METODO_LLEGADA = [
    (
        re.compile(
            r"\b(helic[oó]ptero|hems|heli|evacuaci[oó]n a[eé]rea)\b",
            re.IGNORECASE,
        ),
        "helicoptero",
    ),
    (
        re.compile(
            r"\b(ambulancia|uvi m[oó]vil|samu|061|ems)\b",
            re.IGNORECASE,
        ),
        "ambulancia",
    ),
    (
        re.compile(
            r"\b(acude caminando|por sus medios|coche propio|"
            r"tra[ií]do por familiar|acude solo|acude acompa[nñ]ado)\b",
            re.IGNORECASE,
        ),
        "autonomo",
    ),
]

_MOTIVOS_LEVES_EXPLICITOS = [
    (
        re.compile(
            r"\b(receta|renovaci[oó]n.*medicaci[oó]n|"
            r"medicaci[oó]n habitual.*acab[oó])\b",
            re.IGNORECASE,
        ),
        "medication refill",
    ),
    (
        re.compile(
            r"\b(justificante|informe de baja|parte m[eé]dico)\b",
            re.IGNORECASE,
        ),
        "administrative request",
    ),
    (
        re.compile(
            r"\b(revisi[oó]n.*herida|herida quir[uú]rgica limpia|cura)\b",
            re.IGNORECASE,
        ),
        "wound check",
    ),
    (
        re.compile(r"\b(retirada de puntos|quitar puntos)\b", re.IGNORECASE),
        "suture removal",
    ),
    (
        re.compile(
            r"\b(dolor.*rodilla.*cr[oó]nic|rodilla.*cr[oó]nic)\b",
            re.IGNORECASE,
        ),
        "chronic knee pain",
    ),
    (re.compile(r"\b(ojo seco|sequedad ocular)\b", re.IGNORECASE), "dry eye"),
    (
        re.compile(
            r"\b(o[ií]do taponado|sensaci[oó]n.*o[ií]do.*taponado)\b",
            re.IGNORECASE,
        ),
        "ear fullness",
    ),
    (
        re.compile(r"\b(corte superficial|herida superficial)\b", re.IGNORECASE),
        "minor wound",
    ),
]


def _float_texto(valor: str) -> float:
    return float(valor.replace(",", "."))


def _add_unique(values: list[str], value: str) -> list[str]:
    normalized = {item.strip().lower() for item in values}
    if value.lower() in normalized:
        return values
    return [*values, value]


def _completar_datos_explicitos(
    narrativa: str, vector: VectorClinico
) -> VectorClinico:
    """
    Recupera datos literales que el LLM puede omitir en casos leves.

    No interpreta gravedad ni inventa valores: solo copia constantes escritas en
    la narrativa y motivos administrativos explícitos cuando sintomas_presentes
    queda vacío.
    """
    cambios: dict[str, object] = {}

    if vector.presion_sistolica is None or vector.presion_diastolica is None:
        if match := _PATRON_TENSION.search(narrativa):
            cambios.setdefault("presion_sistolica", int(match.group(1)))
            cambios.setdefault("presion_diastolica", int(match.group(2)))

    if vector.frecuencia_cardiaca is None:
        if match := _PATRON_FC.search(narrativa):
            cambios["frecuencia_cardiaca"] = int(match.group(1))

    if vector.frecuencia_respiratoria is None:
        if match := _PATRON_FR.search(narrativa):
            cambios["frecuencia_respiratoria"] = int(match.group(1))

    if vector.saturacion_oxigeno is None:
        if match := _PATRON_SATURACION.search(narrativa):
            cambios["saturacion_oxigeno"] = float(match.group(1))

    if vector.temperatura is None:
        if match := _PATRON_TEMPERATURA.search(narrativa):
            cambios["temperatura"] = _float_texto(match.group(1))

    sintomas = list(vector.sintomas_presentes)
    for patron, sintoma in _PATRONES_SINTOMAS_EXPLICITOS:
        if patron.search(narrativa):
            sintomas = _add_unique(sintomas, sintoma)

    if _PATRON_ESTADO_MENTAL_GUIADO.search(narrativa):
        sintomas = _add_unique(sintomas, "altered mental status")

    if match := _PATRON_GLUCEMIA.search(narrativa):
        if int(match.group(1)) < 70:
            sintomas = _add_unique(sintomas, "hypoglycemia")

    if sintomas != list(vector.sintomas_presentes):
        cambios["sintomas_presentes"] = sintomas

    if vector.metodo_llegada == "desconocido":
        for patron, metodo in _PATRONES_METODO_LLEGADA:
            if patron.search(narrativa):
                cambios["metodo_llegada"] = metodo
                break

    if not vector.sintomas_presentes:
        for patron, motivo in _MOTIVOS_LEVES_EXPLICITOS:
            if patron.search(narrativa):
                cambios["sintomas_presentes"] = [motivo]
                break

    if not cambios:
        return vector
    return vector.model_copy(update=cambios)


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
        options={"temperature": 0.0, "num_ctx": 8192},  # Determinismo total, sin creatividad
    )

    vector = normalizar_vector_clinico(
        VectorClinico.model_validate_json(respuesta.message.content)
    )
    vector = normalizar_vector_clinico(_completar_datos_explicitos(narrativa, vector))
    logger.success(f"Extraídos {len(vector.sintomas_presentes)} síntomas")
    return vector
