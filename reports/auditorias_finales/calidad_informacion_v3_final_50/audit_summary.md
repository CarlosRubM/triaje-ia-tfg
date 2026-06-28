# Auditoria de calidad de informacion con v3_final

Compara 4 niveles de narrativa sobre los mismos 50 casos sinteticos.
El modelo, el prompt, los artefactos y la politica A1 permanecen congelados.

## Resumen por nivel

| info_level    | description                                            | n_cases | n_ok | n_failed | mean_elapsed_seconds | mean_scalar_match_rate | mean_symptom_recall | empty_symptoms | validator_alerts_total | symptom_extra_total | mean_abs_delta_vs_reference | severe_discrepancies | safety_failures_a1 | low_acuity_predicted_a1 | a1_alerts | pred_esi_1 | pred_esi_2 | pred_esi_3 | pred_esi_4 | pred_esi_5 |
| ------------- | ------------------------------------------------------ | ------- | ---- | -------- | -------------------- | ---------------------- | ------------------- | -------------- | ---------------------- | ------------------- | --------------------------- | -------------------- | ------------------ | ----------------------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| L1_minimo     | Edad/sexo + motivo principal.                          | 50      | 50   | 0        | 6.828                | 0.293                  | 0.773               | 0              | 28                     | 16                  | 1.64                        | 27                   | 0                  | 14                      | 1         | 46         | 1          | 0          | 0          | 3          |
| L2_constantes | L1 + constantes vitales y dolor si aparece.            | 50      | 50   | 0        | 7.34                 | 1.0                    | 0.735               | 0              | 3                      | 18                  | 0.5                         | 6                    | 1                  | 0                       | 0         | 14         | 12         | 15         | 6          | 3          |
| L3_contexto   | L2 + duracion, Glasgow/estado y negaciones relevantes. | 50      | 50   | 0        | 7.362                | 1.0                    | 0.755               | 1              | 6                      | 13                  | 0.56                        | 7                    | 1                  | 0                       | 0         | 14         | 13         | 15         | 6          | 2          |
| L4_completo   | Narrativa completa original.                           | 50      | 50   | 0        | 7.586                | 1.0                    | 0.64                | 0              | 7                      | 22                  | 0.54                        | 6                    | 1                  | 0                       | 0         | 15         | 11         | 14         | 9          | 1          |

## Cambio L1 a L4

- Mejoran: 30 casos.
- Empeoran: 3 casos.
- Igual: 17 casos.

| original_case_id | familia_clinica    | foco_auditoria                    | esi_referencia | L1_minimo | L2_constantes | L3_contexto | L4_completo | delta_L1_to_L4 |
| ---------------- | ------------------ | --------------------------------- | -------------- | --------- | ------------- | ----------- | ----------- | -------------- |
| F006             | endocrino          | hipoglucemia grave                | 1              | 0         | 1             | 1           | 1           | 1              |
| F011             | dolor_toracico     | dolor toracico tiempo-dependiente | 2              | 1         | 0             | 0           | 0           | -1             |
| F014             | dolor_abdominal    | abdomen agudo                     | 2              | 1         | 0             | 0           | 0           | -1             |
| F016             | trauma             | trauma craneal anticoagulado      | 2              | 1         | 0             | 0           | 0           | -1             |
| F018             | embarazo           | hemorragia obstetrica             | 2              | 1         | 0             | 0           | 0           | -1             |
| F019             | alergia            | reaccion alergica con edema       | 2              | 1         | 0             | 0           | 0           | -1             |
| F020             | cefalea            | cefalea trueno                    | 2              | 1         | 0             | 0           | 0           | -1             |
| F021             | dolor_toracico     | dolor toracico estable            | 3              | 2         | 1             | 1           | 1           | -1             |
| F022             | digestivo          | dolor FID                         | 3              | 2         | 0             | 0           | 0           | -2             |
| F023             | urinario           | pielonefritis posible             | 3              | 2         | 0             | 0           | 0           | -2             |
| F024             | syncope            | sincope recuperado                | 3              | 2         | 1             | 1           | 1           | -1             |
| F025             | trauma             | fractura probable                 | 3              | 2         | 0             | 0           | 0           | -2             |
| F027             | ginecologia        | dolor pelvico                     | 3              | 2         | 0             | 0           | 0           | -2             |
| F028             | neurologia         | vertigo intenso                   | 3              | 2         | 0             | 0           | 0           | -2             |
| F029             | digestivo          | rectorragia con anticoagulacion   | 3              | 2         | 1             | 1           | 1           | -1             |
| F031             | urologia           | retencion urinaria                | 3              | 2         | 0             | 0           | 0           | -2             |
| F032             | piel               | celulitis con fiebre              | 3              | 2         | 0             | 0           | 0           | -2             |
| F034             | trauma             | esguince leve                     | 4              | 3         | 1             | 0           | 0           | -3             |
| F035             | herida             | herida simple                     | 4              | 3         | 1             | 1           | 0           | -3             |
| F036             | otorrino           | otalgia                           | 4              | 3         | 1             | 1           | 1           | -2             |
| F037             | dental             | dolor dental                      | 4              | 3         | 0             | 0           | 0           | -3             |
| F038             | oftalmologia       | conjuntivitis                     | 4              | 3         | 0             | 1           | 1           | -2             |
| F039             | piel               | una encarnada                     | 4              | 3         | 1             | 1           | 1           | -2             |
| F040             | piel               | quemadura pequena                 | 4              | 3         | 0             | 0           | 0           | -3             |
| F041             | gastro             | nauseas leves                     | 4              | 3         | 1             | 1           | 1           | -2             |
| F042             | musculoesqueletico | dolor hombro no traumatico        | 4              | 3         | 0             | 0           | 0           | -3             |
| F044             | administrativo     | informe de baja                   | 5              | 4         | 0             | 0           | 1           | -3             |
| F045             | cura               | revision de herida                | 5              | 0         | 1             | 1           | 1           | 1              |
| F046             | cura               | retirada de puntos                | 5              | 0         | 1             | 1           | 1           | 1              |
| F047             | oftalmologia       | ojo seco                          | 5              | 4         | 2             | 2           | 2           | -2             |
| F048             | otorrino           | oido taponado                     | 5              | 4         | 2             | 2           | 2           | -2             |
| F049             | musculoesqueletico | dolor cronico rodilla             | 5              | 4         | 2             | 2           | 2           | -2             |
| F050             | administrativo     | certificado                       | 5              | 4         | 0             | 3           | 1           | -3             |

## Fallos de ejecucion

No hubo fallos de ejecucion.

## Discrepancias severas

| original_case_id | info_level    | familia_clinica    | esi_referencia | clase_predicha | confianza           | foco_auditoria                  |
| ---------------- | ------------- | ------------------ | -------------- | -------------- | ------------------- | ------------------------------- |
| F021             | L1_minimo     | dolor_toracico     | 3              | 1              | 0.9529754286586469  | dolor toracico estable          |
| F022             | L1_minimo     | digestivo          | 3              | 1              | 0.8829310997889243  | dolor FID                       |
| F023             | L1_minimo     | urinario           | 3              | 1              | 0.7440627089820698  | pielonefritis posible           |
| F024             | L1_minimo     | syncope            | 3              | 1              | 0.90120695566157    | sincope recuperado              |
| F025             | L1_minimo     | trauma             | 3              | 1              | 0.8895033401425834  | fractura probable               |
| F026             | L1_minimo     | respiratorio       | 3              | 1              | 0.8246593629307308  | neumonia estable                |
| F026             | L2_constantes | respiratorio       | 3              | 1              | 0.8517043946020577  | neumonia estable                |
| F026             | L3_contexto   | respiratorio       | 3              | 1              | 0.8517043946020577  | neumonia estable                |
| F026             | L4_completo   | respiratorio       | 3              | 1              | 0.8498368774674373  | neumonia estable                |
| F027             | L1_minimo     | ginecologia        | 3              | 1              | 0.7163964313820691  | dolor pelvico                   |
| F028             | L1_minimo     | neurologia         | 3              | 1              | 0.8572642139661827  | vertigo intenso                 |
| F029             | L1_minimo     | digestivo          | 3              | 1              | 0.8704913623329695  | rectorragia con anticoagulacion |
| F030             | L1_minimo     | pediatria          | 3              | 1              | 0.6339026800247961  | fiebre pediatrica               |
| F030             | L2_constantes | pediatria          | 3              | 1              | 0.6308485808716507  | fiebre pediatrica               |
| F030             | L3_contexto   | pediatria          | 3              | 1              | 0.6308485808716507  | fiebre pediatrica               |
| F030             | L4_completo   | pediatria          | 3              | 1              | 0.6308485808716507  | fiebre pediatrica               |
| F031             | L1_minimo     | urologia           | 3              | 1              | 0.7938827683219176  | retencion urinaria              |
| F032             | L1_minimo     | piel               | 3              | 1              | 0.8215787060206378  | celulitis con fiebre            |
| F033             | L1_minimo     | oftalmologia       | 3              | 1              | 0.938191270059765   | perdida visual                  |
| F033             | L2_constantes | oftalmologia       | 3              | 1              | 0.594007590719871   | perdida visual                  |
| F033             | L3_contexto   | oftalmologia       | 3              | 1              | 0.594007590719871   | perdida visual                  |
| F033             | L4_completo   | oftalmologia       | 3              | 1              | 0.279607067652091   | perdida visual                  |
| F034             | L1_minimo     | trauma             | 4              | 1              | 0.5646694207691656  | esguince leve                   |
| F035             | L1_minimo     | herida             | 4              | 1              | 0.9603878970392412  | herida simple                   |
| F036             | L1_minimo     | otorrino           | 4              | 1              | 0.6393180574015144  | otalgia                         |
| F037             | L1_minimo     | dental             | 4              | 1              | 0.6217836858019659  | dolor dental                    |
| F038             | L1_minimo     | oftalmologia       | 4              | 1              | 0.9103337769619108  | conjuntivitis                   |
| F039             | L1_minimo     | piel               | 4              | 1              | 0.7746353939440221  | una encarnada                   |
| F040             | L1_minimo     | piel               | 4              | 1              | 0.9500623915226557  | quemadura pequena               |
| F041             | L1_minimo     | gastro             | 4              | 1              | 0.827640480882254   | nauseas leves                   |
| F042             | L1_minimo     | musculoesqueletico | 4              | 1              | 0.745134800554842   | dolor hombro no traumatico      |
| F044             | L1_minimo     | administrativo     | 5              | 1              | 0.6291796448118263  | informe de baja                 |
| F047             | L1_minimo     | oftalmologia       | 5              | 1              | 0.9503277554928702  | ojo seco                        |
| F047             | L2_constantes | oftalmologia       | 5              | 3              | 0.2694653606755024  | ojo seco                        |
| F047             | L3_contexto   | oftalmologia       | 5              | 3              | 0.2694653606755024  | ojo seco                        |
| F047             | L4_completo   | oftalmologia       | 5              | 3              | 0.2694653606755024  | ojo seco                        |
| F048             | L1_minimo     | otorrino           | 5              | 1              | 0.8973323463523892  | oido taponado                   |
| F048             | L2_constantes | otorrino           | 5              | 3              | 0.28635528905000035 | oido taponado                   |
| F048             | L3_contexto   | otorrino           | 5              | 3              | 0.28635528905000035 | oido taponado                   |
| F048             | L4_completo   | otorrino           | 5              | 3              | 0.28635528905000035 | oido taponado                   |
| F049             | L1_minimo     | musculoesqueletico | 5              | 1              | 0.8912750711823549  | dolor cronico rodilla           |
| F049             | L2_constantes | musculoesqueletico | 5              | 3              | 0.5502320480284285  | dolor cronico rodilla           |
| F049             | L3_contexto   | musculoesqueletico | 5              | 3              | 0.5502320480284285  | dolor cronico rodilla           |
| F049             | L4_completo   | musculoesqueletico | 5              | 3              | 0.5502320480284285  | dolor cronico rodilla           |
| F050             | L1_minimo     | administrativo     | 5              | 1              | 0.6290866234428045  | certificado                     |
| F050             | L3_contexto   | administrativo     | 5              | 2              | 0.4372116167732299  | certificado                     |