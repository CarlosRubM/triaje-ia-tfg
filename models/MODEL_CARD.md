# Model Card — Sistema de Triaje IA

## Modelo elegido

<!-- Completar en F8: nombre del modelo ganador, hiperparametros clave -->

## Dataset

- **Fuente**: MIMIC-IV-ED (Medical Information Mart for Intensive Care, Emergency Department)
- **Poblacion**: Episodios de urgencias con acuity level registrado (ESI 1-5)
- **Target**: `acuity` — nivel de urgencia clinica (1=critico, 5=no urgente)

## Features

- **Total**: 88 features derivadas de signos vitales, datos demograficos, motivo de consulta, medicacion habitual, historial de visitas y comorbilidades CCS.
- **Descripcion detallada**: ver `src/triaje_ia/data/features.py` y `notebooks/2_preparacion_datos/05_ingenieria_features.ipynb`.

## Metricas

<!-- Completar en F8 con los resultados finales de 06b -->

| Metrica         | Valor |
|-----------------|-------|
| Accuracy        |       |
| Macro F1        |       |
| AUC-ROC (macro) |       |
| Critical miss   |       |

## Politica de umbral

- **Politica oficial**: `threshold_a1` — si `P(acuity=1) >= 0.20`, se asigna acuity 1 independientemente del argmax.
- **Justificacion**: minimizar falsos negativos en pacientes criticos tiene mayor coste clinico que los falsos positivos.

## Limitaciones conocidas

- `ccs_category` se extrae de diagnosticos registrados durante o tras el episodio, no en el minuto 0. En un sistema de produccion real esto constituye data leakage temporal. El modelo es valido para investigacion y demostracion, no para despliegue clinico real.
- El modelo no ha sido validado externamente en poblaciones distintas a MIMIC-IV-ED.
- La conversion de temperatura Celsius a Fahrenheit en el adapter introduce una dependencia de unidades que debe auditarse en cualquier integracion futura.

## Uso previsto

Demostracion academica de un sistema hibrido LLM + ML para soporte a la decision clinica en triaje de urgencias. TFG — curso 2024/2025.

## Uso no previsto

Este modelo **no debe usarse** para tomar decisiones clinicas reales sobre pacientes. No esta validado, auditado ni certificado para uso clinico.
