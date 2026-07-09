# AGENTS.md

## 1. Para qué sirve este archivo

Este archivo resume el contexto técnico del proyecto y las reglas básicas que se
han seguido al trabajar sobre el código, notebooks y artefactos. Está pensado
como guía de mantenimiento para no romper el flujo final del TFG.

Durante el desarrollo se han usado herramientas de asistencia a la programación
para revisar código, ordenar tareas, detectar riesgos y documentar decisiones.
La responsabilidad de las decisiones finales del proyecto sigue siendo del
autor.

## 2. Resumen del proyecto

`triaje-ia-tfg` es un prototipo académico de apoyo a la decisión clínica para
triaje de urgencias.

El sistema combina:

- un LLM local con Ollama para extraer información clínica desde texto libre;
- un `VectorClinico` estructurado con Pydantic;
- un adaptador que transforma ese vector en features tabulares;
- un modelo LightGBM final con features tabulares y componentes BERT/SVD;
- una alerta conservadora para posibles casos Acuity 1;
- una explicación SHAP en la interfaz.

El proyecto no es un producto clínico ni debe usarse para decisiones reales.

## 3. PROHIBIDO

No hacer nunca sin confirmación explícita:

- usar test para escoger modelos, features, umbrales o calibración;
- cambiar la política A1 o sus umbrales sin justificar impacto clínico;
- borrar notebooks, datos, modelos o artefactos dudosos;
- limpiar outputs de notebooks por defecto;
- subir `.env`, datos MIMIC, modelos pesados o cachés;
- imprimir secretos o contenido de `.env`;
- introducir rutas absolutas;
- modificar prompts LLM como si fueran texto normal sin validar la extracción;
- hacer refactors grandes si el objetivo era corregir un bug pequeño;
- mezclar cambios de código, notebooks, reports y documentación en el mismo
  commit si pueden separarse.

## 3. Flujo real de la aplicación

```text
Texto libre del caso clínico
  -> llm/factory.py          configura el extractor local
  -> llm/extractor.py        extrae JSON clínico y crea VectorClinico
  -> llm/normalizer.py       normaliza síntomas y medicación frecuentes
  -> llm/validator.py        avisa de incoherencias semánticas
  -> inference/adapter.py    convierte VectorClinico a features tabulares
  -> inference/predictor.py  añade BERT/SVD y llama al modelo final
  -> LightGBM final          estima probabilidades ESI 1-5
  -> política de decisión    argmax + alerta A1 conservadora
  -> ml/explicabilidad.py    genera explicación SHAP
  -> ui/app.py               muestra resultado en Streamlit
```

La app principal está en:

```text
src/triaje_ia/ui/app.py
```

## 4. Estructura importante

| Ruta | Papel |
|---|---|
| `src/triaje_ia/data/` | Carga, limpieza y generación offline de features. |
| `src/triaje_ia/llm/` | Extracción, normalización y validación del texto libre. |
| `src/triaje_ia/inference/` | Adaptador y predictor usados por la app. |
| `src/triaje_ia/ml/` | Pipeline, decisión y explicabilidad. |
| `src/triaje_ia/ui/app.py` | Interfaz Streamlit. |
| `notebooks/` | Desarrollo metodológico del TFG. |
| `models/` | Configuración del modelo activo y modelo final si está disponible localmente. |
| `reports/` | Métricas y figuras finales. |
| `prompts/` | Prompts del extractor LLM. |
| `tests/` | Tests unitarios. |

## 5. Notebooks que forman la línea final

Los notebooks finales que deben mantenerse como referencia principal son:

```text
notebooks/1_data_understanding/01_data_exploration.ipynb
notebooks/1_data_understanding/02_loader_validation.ipynb
notebooks/2_data_preparation/03_cleaning_analysis.ipynb
notebooks/2_data_preparation/04_cleaner_validation.ipynb
notebooks/2_data_preparation/05_feature_engineering.ipynb
notebooks/2_data_preparation/05b_feature_validation.ipynb
notebooks/3_modeling/06_nlp_feature_extraction.ipynb
notebooks/3_modeling/07_model_training.ipynb
notebooks/4_evaluation/08_model_evaluation.ipynb
notebooks/4_evaluation/09_llm_extraction_validation.ipynb
```

Además, se conservan como evidencia complementaria citada en la memoria:

```text
notebooks/3_modeling/07b_lgbm_bert_optuna_tuning.ipynb
notebooks/3_modeling/07c_lgbm_bert_tail_class_tuning.ipynb
notebooks/4_evaluation/10_llm_production_flow_audit_v3.ipynb
notebooks/4_evaluation/11_llm_minimum_information_audit_v3.ipynb
```

Los notebooks `07b` y `07c` documentan alternativas evaluadas y descartadas.
Los notebooks `10` y `11` son auditorías del flujo final y de la información
mínima; no constituyen validación clínica ni sustituyen la línea principal.

Los notebooks antiguos de pruebas se han movido fuera de la línea final del
proyecto. No deben volver a mezclarse con el flujo principal salvo para consultar
contexto histórico.

## 6. Comandos básicos

El entorno se define en:

```text
pyproject.toml
uv.lock
```

La versión objetivo es Python `>=3.11`. Usar `uv` como gestor de dependencias;
no mantener dependencias manuales fuera de `pyproject.toml`.

Instalar dependencias:

```bash
uv sync
```

Instalar dependencias de desarrollo:

```bash
uv sync --group dev
```

Ejecutar tests:

```bash
uv run pytest tests/ -v
```

Ejecutar la app:

```bash
uv run streamlit run src/triaje_ia/ui/app.py
```

LLM local con Ollama:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve
```

## 7. Artefactos del modelo final

Configuración versionable:

```text
models/active_model.json
models/feature_list.json
models/thresholds.json
models/model_training_metadata.json
models/production_assumptions.json
```

Artefactos finales congelados que se versionan como excepción para que un clon
pueda ejecutar la app:

```text
models/lgbm_bert_final.joblib
data/processed/bert_svd.joblib
models/artifact_manifest.json
```

La excepción se limita a estos dos binarios y a los hashes registrados en el
manifiesto. Otros modelos serializados y datos procesados siguen sin subirse.

Resultados finales útiles para la memoria:

```text
reports/final_evaluation/
reports/figures/final_evaluation/
```

## 8. Reglas metodológicas que no se deben romper

- No usar test para elegir modelos, features, thresholds o calibración.
- Las decisiones de selección se hacen con train, validación cruzada u OOF.
- Los transformadores se ajustan solo con train.
- Test temporal se reserva para evaluación final.
- No usar información futura del episodio como feature de minuto 0.
- No usar diagnóstico final ni CCS del episodio actual como predictor.
- Si se transforma test, debe ser con transformadores ya ajustados.
- No copiar resultados antiguos en notebooks finales sin recalcularlos.

## 9. Política A1

La configuración final de la app usa `argmax` como decisión principal.

Además, muestra una alerta cuando:

```text
P(Acuity 1) >= 0.40
```

y la clase sugerida por `argmax` no es ESI 1.

Esta alerta no cambia automáticamente la predicción. Se usa como mecanismo de
seguridad visual para que un caso con probabilidad relevante de Acuity 1 no pase
desapercibido.

No modificar esta lógica sin:

- explicar el motivo;
- revisar impacto clínico;
- actualizar tests;
- comprobar que la app sigue mostrando la alerta correctamente.

## 10. Reglas al tocar código

- Leer primero el módulo afectado y sus tests.
- Hacer cambios pequeños y fáciles de revisar.
- No refactorizar por estética si no hay una razón clara.
- No cambiar comportamiento observable salvo bug o mejora justificada.
- No introducir dependencias nuevas sin necesidad real.
- No imprimir secretos ni leer `.env`.
- No guardar datos sensibles en logs.
- Mantener rutas relativas basadas en `config.py`.
- Si se toca producción, ejecutar tests.

## 11. Reglas al tocar prompts LLM

Los prompts están en:

```text
prompts/
```

Cambiar un prompt puede modificar la extracción clínica sin provocar un error de
Python. Por eso, si se toca un prompt:

- revisar `src/triaje_ia/llm/schemas.py`;
- probar varios casos clínicos representativos;
- comprobar que el JSON extraído sigue cumpliendo el contrato;
- revisar alertas de `validator.py`;
- no cambiar a la vez prompt, schema y modelo salvo que sea imprescindible.

## 12. Reglas al tocar notebooks

- No quitar outputs por defecto; en este proyecto interesa conservar resultados
  visibles.
- Mantener narrativa clara y defendible.
- Evitar experimentos históricos dentro de notebooks finales.
- Si un notebook genera artefactos, debe explicar qué genera y para qué.
- Si se usa test, debe quedar claro que es solo evaluación o transformación
  final, no selección.

## 13. Archivos que no deben subirse

No subir al repositorio normal:

```text
data/raw/
data/interim/
data/processed/* salvo data/processed/bert_svd.joblib
models/*.joblib salvo models/lgbm_bert_final.joblib
models/*.pkl
.env
.venv/
.cursor/
.codegraph/
_archive_before_delivery/
__pycache__/
```

Si algo no está claro, no borrarlo directamente. Primero moverlo a archivo local
o dejarlo en revisión.

## 14. Criterio de cierre de un cambio

Un cambio se considera razonablemente cerrado si:

- los tests relevantes pasan;
- no se han añadido rutas absolutas;
- no se han añadido secretos;
- el cambio no introduce leakage;
- la app sigue funcionando si afecta a inferencia o UI;
- el commit queda separado por tema y con un mensaje comprensible.

Este archivo no sustituye a la memoria del TFG. Solo deja por escrito el modo de
trabajo seguido para mantener el repositorio ordenado y evitar cambios
metodológicamente peligrosos.
