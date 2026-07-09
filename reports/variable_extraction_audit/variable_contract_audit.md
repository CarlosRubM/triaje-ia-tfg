# Auditoria determinista de variables

Generado: 2026-06-06T00:13:54
Features tabulares: 88
Estado: PASS

## Campos VectorClinico

- `edad`
- `sexo`
- `sintomas_presentes`
- `patologias_previas`
- `medicacion_habitual`
- `presion_sistolica`
- `presion_diastolica`
- `frecuencia_cardiaca`
- `frecuencia_respiratoria`
- `saturacion_oxigeno`
- `temperatura`
- `nivel_dolor`
- `duracion_sintomas`
- `metodo_llegada`

## Checks

- `PASS` catalogo_88_features: 88 features catalogadas
- `PASS` llegada_ambulancia: ambulancia -> llegada_ambulancia
- `PASS` llegada_helicoptero: helicoptero -> llegada_helicoptero
- `PASS` llegada_autonomo: autonomo -> llegada_autonoma
- `PASS` llegada_otro: otro -> llegada_autonoma
- `PASS` llegada_desconocido: desconocido -> llegada_desconocida
- `PASS` sin_medicacion_excluye_riesgo_farmacologico: sin medicacion mantiene todas las flags farmacologicas a cero
- `PASS` zona_verde_excluye_criticos: constantes normales activan zona verde sin criticos
- `PASS` qsofa_dos_criterios: FR 22 y PAS 100 activan qSOFA positivo

## Nota

Esta auditoria no ejecuta el LLM. La validacion con LLM debe hacerse con casos oro clinicos y sin usar el conjunto test para ajustar prompts, features, umbrales o modelos.
