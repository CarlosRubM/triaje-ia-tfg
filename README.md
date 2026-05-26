# trIAje - sistema híbrido LLM + ML para triaje de urgencias

Trabajo Fin de Grado del Grado en Ingeniería Informática.

Autor: Carlos Rubio  
Repositorio: [CarlosRubM/triaje-ia-tfg](https://github.com/CarlosRubM/triaje-ia-tfg)

Este proyecto desarrolla un prototipo académico de apoyo a la decisión clínica
para triaje de urgencias. La idea principal es combinar texto libre escrito por
el usuario con un modelo de machine learning entrenado sobre MIMIC-IV-ED para
estimar el nivel de urgencia ESI/Acuity de 1 a 5.

La escala ESI/Acuity ordena la urgencia de 1 a 5: el nivel 1 representa los
casos más críticos y el nivel 5 los menos urgentes. Es un problema difícil
porque las clases están desbalanceadas y porque los pacientes críticos son pocos
en proporción, pero son los más sensibles desde el punto de vista clínico.

El sistema no está pensado para uso clínico real. Es un trabajo académico y sus
predicciones deben interpretarse solo como una demostración técnica.

## Idea general

El flujo completo es:

```text
Historia clínica en texto libre
  -> extracción LLM a VectorClinico
  -> validación y normalización del vector
  -> conversión a features tabulares
  -> incorporación de componentes BERT/SVD
  -> LightGBM final
  -> predicción ESI + alerta conservadora A1
  -> explicación SHAP en la interfaz
```

El LLM no decide el nivel de triaje. Su papel es extraer información clínica
estructurada desde texto libre. La predicción final la realiza un modelo
LightGBM entrenado y evaluado de forma separada.

## Arquitectura del proyecto

```text
triaje-ia-tfg/
├── src/triaje_ia/                  # Código fuente del sistema
│   ├── data/                       # Carga, limpieza y generación de features
│   ├── llm/                        # Extracción clínica desde texto libre
│   ├── inference/                  # Adaptador y predictor final
│   ├── ml/                         # Pipeline ML, decisión y explicabilidad
│   └── ui/                         # Aplicación Streamlit
├── notebooks/                      # Desarrollo metodológico del TFG
│   ├── 1_data_understanding/
│   ├── 2_data_preparation/
│   ├── 3_modeling/
│   └── 4_evaluation/
├── prompts/                        # Prompts usados por el extractor LLM
├── models/                         # Configuración del modelo activo
├── reports/                        # Resultados y figuras finales
├── tests/                          # Tests unitarios
├── pyproject.toml
└── .env.example
```

### Código fuente

| Ruta | Papel |
|---|---|
| `src/triaje_ia/config.py` | Define rutas comunes del proyecto. |
| `src/triaje_ia/data/loader.py` | Carga y unión de tablas de MIMIC-IV-ED. |
| `src/triaje_ia/data/cleaner.py` | Limpieza del dataset antes del modelado. |
| `src/triaje_ia/data/features.py` | Generación de las features tabulares base. |
| `src/triaje_ia/llm/schemas.py` | Esquema Pydantic del `VectorClinico`. |
| `src/triaje_ia/llm/extractor.py` | Extracción local con Ollama. |
| `src/triaje_ia/llm/extractor_api.py` | Extracción alternativa mediante API. |
| `src/triaje_ia/llm/factory.py` | Selección del backend LLM. |
| `src/triaje_ia/llm/normalizer.py` | Normalización ligera de síntomas y medicación. |
| `src/triaje_ia/llm/validator.py` | Alertas de coherencia sobre el vector extraído. |
| `src/triaje_ia/inference/adapter.py` | Conversión de `VectorClinico` a features ML. |
| `src/triaje_ia/inference/predictor.py` | Carga del modelo final e inferencia. |
| `src/triaje_ia/ml/pipeline.py` | Pipeline de preprocesado/modelado. |
| `src/triaje_ia/ml/decision.py` | Políticas de decisión y alerta A1. |
| `src/triaje_ia/ml/explicabilidad.py` | Explicaciones SHAP. |
| `src/triaje_ia/ui/app.py` | Aplicación Streamlit. |

## Notebooks finales

Los notebooks que quedan en el proyecto forman el flujo defendible del TFG:

| Notebook | Rol |
|---|---|
| `notebooks/1_data_understanding/01_data_exploration.ipynb` | Exploración inicial del dataset. |
| `notebooks/1_data_understanding/02_loader_validation.ipynb` | Validación de la carga de datos. |
| `notebooks/2_data_preparation/03_cleaning_analysis.ipynb` | Análisis de limpieza. |
| `notebooks/2_data_preparation/04_cleaner_validation.ipynb` | Validación del cleaner. |
| `notebooks/2_data_preparation/05_feature_engineering.ipynb` | Construcción de features tabulares. |
| `notebooks/2_data_preparation/05b_feature_validation.ipynb` | Validación y ampliación final de features. |
| `notebooks/3_modeling/06_nlp_feature_extraction.ipynb` | Features LLM y embeddings BERT/SVD. |
| `notebooks/3_modeling/07_model_training.ipynb` | Entrenamiento, CV, OOF y congelación del modelo final. |
| `notebooks/4_evaluation/08_model_evaluation.ipynb` | Evaluación final sobre test temporal. |
| `notebooks/4_evaluation/09_llm_extraction_validation.ipynb` | Validación local de extracción LLM con casos clínicos. |

Los notebooks históricos de prueba se han retirado de la línea final del
proyecto para que el repositorio muestre solo el flujo principal.

## Modelo final

La configuración activa está en:

```text
models/active_model.json
```

El modelo final es:

```text
LightGBM + features tabulares finales + componentes BERT/SVD
```

Artefactos principales:

| Artefacto | Ruta |
|---|---|
| Clasificador final | `models/lgbm_bert_final.joblib` |
| Lista exacta de variables | `models/feature_list.json` |
| Umbral de alerta y política | `models/thresholds.json` |
| Metadata de entrenamiento | `models/model_training_metadata.json` |
| Supuestos de producción | `models/production_assumptions.json` |
| Reductor BERT/SVD | `data/processed/bert_svd.joblib` |

El modelo usa 79 variables finales. El entrenamiento se realizó sobre train con
validación cruzada agrupada y predicciones OOF. El test temporal queda reservado
para la evaluación final del notebook 08.

Las variables finales combinan constantes vitales, flags clínicos derivados,
medicación y antecedentes, motivo de consulta, variables de llegada y
componentes semánticos BERT/SVD del texto.

La decisión final de la app usa `argmax`. Además, se muestra una alerta clínica
conservadora si `P(Acuity 1) >= 0.40` y la clase sugerida no es ESI 1. Esa alerta
no cambia automáticamente la predicción: sirve como aviso de seguridad.

## Datos usados

El modelado se realiza sobre MIMIC-IV-ED con separación temporal train/test:

| Partición | Episodios |
|---|---:|
| Train | 334.480 |
| Test temporal | 83.620 |

Distribución de clases en el test temporal:

| Clase | Porcentaje |
|---|---:|
| Acuity 1 | 5,6 % |
| Acuity 2 | 33,7 % |
| Acuity 3 | 54,3 % |
| Acuity 4 | 6,3 % |
| Acuity 5 | 0,2 % |

## Resultados principales

Resultados finales sobre test temporal (`n = 83.620`), generados por
`notebooks/4_evaluation/08_model_evaluation.ipynb`:

| Métrica | Valor |
|---|---:|
| Macro F1 | 0.560 |
| Weighted F1 | 0.697 |
| Balanced Accuracy | 0.576 |
| Precision Acuity 1 | 0.658 |
| Recall Acuity 1 | 0.691 |
| F1 Acuity 1 | 0.674 |
| Precision Acuity 2 | 0.665 |
| Recall Acuity 2 | 0.668 |
| F1 Acuity 2 | 0.667 |
| AUPRC Acuity 1 | 0.705 |
| AUPRC Acuity 2 | 0.726 |

Como referencia interna de desarrollo, el baseline `DummyClassifier`
estratificado obtuvo una Macro F1 OOF de 0.200, la regresión logística base
0.372 y LightGBM con las 46 variables base 0.487. El modelo final
LightGBM+BERT/SVD alcanzó 0.568 en OOF y 0.560 en el test temporal.

Archivos de resultados:

```text
reports/final_evaluation/final_metrics.json
reports/final_evaluation/classification_report.csv
reports/final_evaluation/safety_analysis.csv
reports/final_evaluation/auprc_metrics.json
reports/figures/final_evaluation/confusion_matrix.png
reports/figures/final_evaluation/confusion_matrix_normalized.png
reports/figures/final_evaluation/shap_summary.png
```

## Cómo probar el proyecto

### Requisitos

- Python 3.11 o superior
- `uv`
- Ollama si se quiere usar extracción local
- Artefactos del modelo final y datos procesados necesarios

Instalación:

```bash
uv sync
```

Para desarrollo y tests:

```bash
uv sync --group dev
```

### Configuración del LLM

El proyecto incluye `.env.example`. Para usar configuración local:

```bash
cp .env.example .env
```

Backend local recomendado:

```text
LLM_BACKEND=ollama
```

Modelo usado en las pruebas locales:

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama serve
```

También existe backend API (`LLM_BACKEND=api`) preparado para Groq/OpenAI, pero
la vía local con Ollama es la opción principal para este prototipo.

### Ejecutar la aplicación

```bash
uv run streamlit run src/triaje_ia/ui/app.py
```

Para que la app funcione deben existir:

```text
models/lgbm_bert_final.joblib
models/active_model.json
models/feature_list.json
models/thresholds.json
models/model_training_metadata.json
models/production_assumptions.json
data/processed/bert_svd.joblib
```

Los JSON de configuración sí están pensados para ir en el repositorio. El modelo
`.joblib` y los artefactos pesados de `data/processed/` deben aportarse aparte
si se clona el proyecto desde cero.

### Ejecutar tests

```bash
uv run pytest tests/ -v
```

## Qué incluye y qué no incluye Git

Se versionan:

- código fuente
- notebooks finales
- tests
- prompts
- JSON pequeños de configuración del modelo
- métricas y figuras finales de evaluación

No se versionan:

- datos originales de MIMIC-IV-ED
- parquets intermedios pesados
- caches de BERT/LLM
- modelos `.joblib`
- archivos `.env`
- material histórico archivado antes de la entrega

Esto evita subir datos o binarios pesados y mantiene el repositorio centrado en
el código, la metodología y los resultados finales.

## Limitaciones

- El sistema es académico y no debe usarse para decisiones clínicas reales.
- El entrenamiento se basa en MIMIC-IV-ED, por lo que puede existir diferencia
  entre el entorno de entrenamiento y el uso sobre texto libre en la app.
- La extracción LLM puede cometer errores u omisiones, aunque se aplican reglas
  de validación y normalización.
- Algunas variables históricas de urgencias no están disponibles en producción
  desde texto libre y se imputan con supuestos conservadores documentados.
- La explicación SHAP es descriptiva; no sustituye la valoración clínica.

## Autoría

Proyecto desarrollado por Carlos Rubio como Trabajo Fin de Grado.

Dataset de referencia: MIMIC-IV-ED, sujeto a las condiciones de acceso y uso de
PhysioNet.
