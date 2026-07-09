# Auditoria de casos del capitulo 8

Este informe se genera ejecutando el flujo local del prototipo sobre los tres casos sinteticos usados en el capitulo 8.

## Configuracion

- LLM local: `llama3.1:8b-instruct-q4_K_M`
- Prompt: `prompts/extractor_system_v3_final.txt`
- Temperatura: `0.0`
- Prediccion: `TriajePredictor` con modelo activo de `models/active_model.json`
- Decision: `argmax` + alerta A1 configurada en el predictor

## Resumen de predicciones

| Caso | Clase | P(A1) | P(A2) | P(A3) | P(A4) | P(A5) | Alerta A1 |
|---|---:|---:|---:|---:|---:|---:|---|
| caso_1 | 1 | 0.8898 | 0.1072 | 0.0028 | 0.0001 | 0.0000 | No |
| caso_2 | 3 | 0.0069 | 0.1650 | 0.8178 | 0.0100 | 0.0003 | No |
| caso_3 | 4 | 0.0114 | 0.0693 | 0.2771 | 0.5905 | 0.0517 | No |

## Trazabilidad por caso

### caso_1 - Compromiso respiratorio y cardiovascular

- Perfil previsto: Prioridad maxima esperada por alteracion respiratoria y hemodinamica.
- Entrada minima completa: True
- Alertas de validacion: 0
- Predictores finales usados: 79
- Clase predicha: Acuity 1
- Probabilidad principal: 0.8898
- Alerta A1: No

### caso_2 - Dolor abdominal con vomitos y estrenimiento

- Perfil previsto: Prioridad intermedia esperada por sintomas abdominales y recursos probables.
- Entrada minima completa: True
- Alertas de validacion: 0
- Predictores finales usados: 79
- Clase predicha: Acuity 3
- Probabilidad principal: 0.8178
- Alerta A1: No

### caso_3 - Ojo rojo y picor ocular

- Perfil previsto: Baja prioridad esperada por cuadro leve, sin signos de alarma y constantes normales.
- Entrada minima completa: True
- Alertas de validacion: 0
- Predictores finales usados: 79
- Clase predicha: Acuity 4
- Probabilidad principal: 0.5905
- Alerta A1: No
