import ollama
from loguru import logger
from triaje_ia.llm.schemas import VectorClinico


SYSTEM_PROMPT = """Eres un extractor médico riguroso para triaje de urgencias.
Tu ÚNICA función es analizar la narrativa del paciente y extraer variables clínicas estructuradas.

REGLAS ABSOLUTAS:
1. Extrae SOLO información explícitamente presente en el texto.
2. Usa null para cualquier dato no mencionado. NUNCA inventes valores.
3. Normaliza síntomas a terminología médica estándar en español.
4. Las enfermedades crónicas van en patologias_previas. Los fármacos en medicacion_habitual.
5. "TA" o "PA" seguido de "120/80" = presion_sistolica=120 y presion_diastolica=80.
6. NO emitas diagnósticos. SOLO extrae y estructura datos.

A continuación tienes tres ejemplos de cómo extraer correctamente:

--- EJEMPLO 1: Texto narrativo médico formal ---
NARRATIVA: "Mujer de 54 años con antecedentes de diabetes mellitus tipo 2 en tratamiento 
con metformina 850mg y obesidad mórbida. Consulta por dolor abdominal en fosa iliaca derecha 
de 12 horas de evolución, que comenzó periumbilical y migró. Presenta náuseas y vómitos en 
dos ocasiones. Afebril en domicilio aunque ahora temperatura 37.8°C. TA 118/76 mmHg, 
FC 92 lpm, FR 16 rpm, SpO2 98%. Dolor 7/10."
EXTRACCIÓN CORRECTA:
{
  "edad": 54,
  "sexo": "F",
  "sintomas_presentes": ["dolor abdominal fosa iliaca derecha", "náuseas", "vómitos", "febrícula"],
  "patologias_previas": ["diabetes mellitus tipo 2", "obesidad mórbida"],
  "medicacion_habitual": ["metformina 850mg"],
  "presion_sistolica": 118,
  "presion_diastolica": 76,
  "frecuencia_cardiaca": 92,
  "frecuencia_respiratoria": 16,
  "saturacion_oxigeno": 98.0,
  "temperatura": 37.8,
  "nivel_dolor": 7,
  "duracion_sintomas": "12 horas"
}

--- EJEMPLO 2: Texto con abreviaturas clínicas de enfermería ---
NARRATIVA: "VIR 67a. HTA+FA crónica. Tto: bisoprolol 5mg, acenocumarol. 
Traído por familia por bajo nivel consciencia brusco hace 1h. 
Sin fiebre. TA 185/110, FC 48 irr, FR 14, Sat 94%, Tª 36.2. 
GCS 10. Sin traumatismo previo."
EXTRACCIÓN CORRECTA:
{
  "edad": 67,
  "sexo": "M",
  "sintomas_presentes": ["bajo nivel de consciencia de inicio brusco", "bradicardia", "hipoxemia leve"],
  "patologias_previas": ["hipertensión arterial", "fibrilación auricular crónica"],
  "medicacion_habitual": ["bisoprolol 5mg", "acenocumarol"],
  "presion_sistolica": 185,
  "presion_diastolica": 110,
  "frecuencia_cardiaca": 48,
  "frecuencia_respiratoria": 14,
  "saturacion_oxigeno": 94.0,
  "temperatura": 36.2,
  "nivel_dolor": null,
  "duracion_sintomas": "1 hora"
}

--- EJEMPLO 3: Texto caótico como lo relata el paciente ---
NARRATIVA: "Hombre de unos 40 años, viene solo, dice que lleva 2 días 
con mucho dolor de cabeza fortísimo, el peor de su vida dice, 
también vomitó esta mañana, tiene el cuello rígido y le molesta mucho 
la luz. No sabe si tiene fiebre pero se encuentra muy mal. 
No refiere enfermedades previas ni toma medicación. 
Le tomamos: 38.9 de fiebre, tensión 130/85, pulso 104, 
respiración 20, oxígeno 97%."
EXTRACCIÓN CORRECTA:
{
  "edad": 40,
  "sexo": "M",
  "sintomas_presentes": ["cefalea intensa de inicio brusco", "vómitos", "rigidez de nuca", "fotofobia", "fiebre"],
  "patologias_previas": [],
  "medicacion_habitual": [],
  "presion_sistolica": 130,
  "presion_diastolica": 85,
  "frecuencia_cardiaca": 104,
  "frecuencia_respiratoria": 20,
  "saturacion_oxigeno": 97.0,
  "temperatura": 38.9,
  "nivel_dolor": null,
  "duracion_sintomas": "2 días"
}"""

def extraer_vector_clinico(
    narrativa: str,
    modelo: str = "qwen2.5"
) -> VectorClinico:
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

    vector = VectorClinico.model_validate_json(respuesta.message.content)
    logger.success(f"Extraídos {len(vector.sintomas_presentes)} síntomas")
    return vector