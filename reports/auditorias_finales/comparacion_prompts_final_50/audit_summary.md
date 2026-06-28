# Auditoria final de prompts y flujo de produccion

Auditoria sintetica con 50 historias realistas y gold explicito de VectorClinico.
No modifica prompts, modelo, umbrales ni politica A1.

## Resumen por prompt

| prompt_id       | n_cases | n_ok | n_failed | mean_elapsed_seconds | mean_scalar_match_rate | mean_symptom_recall | empty_symptoms | validator_alerts_total | symptom_extra_total | mean_abs_delta_vs_reference | severe_discrepancies | safety_failures_a1 | low_acuity_predicted_a1 | a1_alerts | pred_esi_1 | pred_esi_2 | pred_esi_3 | pred_esi_4 | pred_esi_5 |
| --------------- | ------- | ---- | -------- | -------------------- | ---------------------- | ------------------- | -------------- | ---------------------- | ------------------- | --------------------------- | -------------------- | ------------------ | ----------------------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| v2_produccion   | 50      | 50   | 0        | 6.971                | 0.993                  | 0.61                | 0              | 10                     | 37                  | 0.56                        | 6                    | 1                  | 0                       | 0         | 14         | 12         | 14         | 9          | 1          |
| v3_structured   | 50      | 50   | 0        | 7.694                | 1.0                    | 0.628               | 0              | 7                      | 28                  | 0.58                        | 7                    | 1                  | 0                       | 0         | 15         | 11         | 16         | 7          | 1          |
| v3_final        | 50      | 50   | 0        | 7.708                | 1.0                    | 0.65                | 0              | 7                      | 24                  | 0.54                        | 6                    | 1                  | 0                       | 0         | 15         | 11         | 14         | 9          | 1          |
| v4_triage_cases | 50      | 50   | 0        | 7.172                | 0.998                  | 0.62                | 0              | 7                      | 32                  | 0.68                        | 6                    | 2                  | 1                       | 0         | 16         | 11         | 16         | 6          | 1          |

## Errores de ejecucion

No se registraron errores de ejecucion.

## Peores extracciones

| prompt_id       | case_id | familia_clinica | foco_auditoria           | scalar_match_rate | symptom_recall | symptom_missing                           | symptom_extra                                           | n_alertas_validador |
| --------------- | ------- | --------------- | ------------------------ | ----------------- | -------------- | ----------------------------------------- | ------------------------------------------------------- | ------------------- |
| v2_produccion   | F045    | cura            | revision de herida       | 1.0               | 0.0            | wound check                               | administrative request                                  | 1                   |
| v4_triage_cases | F039    | piel            | una encarnada            | 1.0               | 0.0            | ingrown toenail|redness|toe pain          | right lower quadrant abdominal pain                     | 1                   |
| v2_produccion   | F012    | neurologia      | ictus tiempo-dependiente | 1.0               | 0.0            | anxiety|aphasia|arm weakness              | altered mental status|weakness left arm                 | 0                   |
| v2_produccion   | F018    | embarazo        | hemorragia obstetrica    | 1.0               | 0.0            | abdominal pain|dizziness|vaginal bleeding | abundant vaginal bleeding|altered mental status|dyspnea | 0                   |
| v2_produccion   | F025    | trauma          | fractura probable        | 1.0               | 0.0            | deformity|wrist pain                      | dorsal pain|right wrist deformity                       | 0                   |
| v2_produccion   | F033    | oftalmologia    | perdida visual           | 1.0               | 0.0            | vision loss                               | minor wound|visual loss                                 | 0                   |
| v2_produccion   | F044    | administrativo  | informe de baja          | 1.0               | 0.0            | administrative request                    | chronic low back pain                                   | 0                   |
| v2_produccion   | F046    | cura            | retirada de puntos       | 1.0               | 0.0            | suture removal                            | administrative request                                  | 0                   |
| v3_structured   | F012    | neurologia      | ictus tiempo-dependiente | 1.0               | 0.0            | anxiety|aphasia|arm weakness              | altered mental status|weakness                          | 0                   |
| v3_structured   | F025    | trauma          | fractura probable        | 1.0               | 0.0            | deformity|wrist pain                      | severe pain|upper limb deformity                        | 0                   |
| v3_structured   | F033    | oftalmologia    | perdida visual           | 1.0               | 0.0            | vision loss                               | blindness                                               | 0                   |
| v3_structured   | F037    | dental          | dolor dental             | 1.0               | 0.0            | dental pain                               | toothache                                               | 0                   |
| v3_structured   | F040    | piel            | quemadura pequena        | 1.0               | 0.0            | minor burn                                | burn|pain                                               | 0                   |
| v3_structured   | F044    | administrativo  | informe de baja          | 1.0               | 0.0            | administrative request                    | chronic low back pain                                   | 0                   |
| v3_final        | F012    | neurologia      | ictus tiempo-dependiente | 1.0               | 0.0            | anxiety|aphasia|arm weakness              | altered mental status|weakness                          | 0                   |
| v3_final        | F025    | trauma          | fractura probable        | 1.0               | 0.0            | deformity|wrist pain                      | arm pain|upper limb deformity                           | 0                   |
| v3_final        | F033    | oftalmologia    | perdida visual           | 1.0               | 0.0            | vision loss                               | blindness                                               | 0                   |
| v3_final        | F040    | piel            | quemadura pequena        | 1.0               | 0.0            | minor burn                                | burn|minor wound                                        | 0                   |
| v3_final        | F044    | administrativo  | informe de baja          | 1.0               | 0.0            | administrative request                    | lower back pain                                         | 0                   |
| v4_triage_cases | F012    | neurologia      | ictus tiempo-dependiente | 1.0               | 0.0            | anxiety|aphasia|arm weakness              | altered mental status|weakness                          | 0                   |

## Discrepancias severas del modelo

| prompt_id       | case_id | familia_clinica    | esi_referencia | clase_predicha | confianza           | alerta_a1_activada | foco_auditoria        |
| --------------- | ------- | ------------------ | -------------- | -------------- | ------------------- | ------------------ | --------------------- |
| v2_produccion   | F026    | respiratorio       | 3              | 1              | 0.8498368774674373  | False              | neumonia estable      |
| v2_produccion   | F030    | pediatria          | 3              | 1              | 0.6439400051984019  | False              | fiebre pediatrica     |
| v2_produccion   | F044    | administrativo     | 5              | 3              | 0.46055654775996463 | False              | informe de baja       |
| v2_produccion   | F047    | oftalmologia       | 5              | 3              | 0.2694653606755024  | False              | ojo seco              |
| v2_produccion   | F048    | otorrino           | 5              | 2              | 0.3782522813282462  | False              | oido taponado         |
| v2_produccion   | F049    | musculoesqueletico | 5              | 3              | 0.5502320480284285  | False              | dolor cronico rodilla |
| v3_structured   | F026    | respiratorio       | 3              | 1              | 0.8498368774674373  | False              | neumonia estable      |
| v3_structured   | F030    | pediatria          | 3              | 1              | 0.6308485808716507  | False              | fiebre pediatrica     |
| v3_structured   | F033    | oftalmologia       | 3              | 1              | 0.279607067652091   | False              | perdida visual        |
| v3_structured   | F044    | administrativo     | 5              | 3              | 0.46055654775996463 | False              | informe de baja       |
| v3_structured   | F047    | oftalmologia       | 5              | 3              | 0.2694653606755024  | False              | ojo seco              |
| v3_structured   | F048    | otorrino           | 5              | 3              | 0.28635528905000035 | False              | oido taponado         |
| v3_structured   | F049    | musculoesqueletico | 5              | 3              | 0.5502320480284285  | False              | dolor cronico rodilla |
| v3_final        | F026    | respiratorio       | 3              | 1              | 0.8498368774674373  | False              | neumonia estable      |
| v3_final        | F030    | pediatria          | 3              | 1              | 0.6308485808716507  | False              | fiebre pediatrica     |
| v3_final        | F033    | oftalmologia       | 3              | 1              | 0.279607067652091   | False              | perdida visual        |
| v3_final        | F047    | oftalmologia       | 5              | 3              | 0.2694653606755024  | False              | ojo seco              |
| v3_final        | F048    | otorrino           | 5              | 3              | 0.28635528905000035 | False              | oido taponado         |
| v3_final        | F049    | musculoesqueletico | 5              | 3              | 0.5502320480284285  | False              | dolor cronico rodilla |
| v4_triage_cases | F026    | respiratorio       | 3              | 1              | 0.8517043946020577  | False              | neumonia estable      |
| v4_triage_cases | F030    | pediatria          | 3              | 1              | 0.6308485808716507  | False              | fiebre pediatrica     |
| v4_triage_cases | F033    | oftalmologia       | 3              | 1              | 0.279607067652091   | False              | perdida visual        |
| v4_triage_cases | F047    | oftalmologia       | 5              | 3              | 0.2694653606755024  | False              | ojo seco              |
| v4_triage_cases | F048    | otorrino           | 5              | 1              | 0.26538794820750317 | False              | oido taponado         |
| v4_triage_cases | F049    | musculoesqueletico | 5              | 3              | 0.5502320480284285  | False              | dolor cronico rodilla |

## Casos de seguridad a revisar

| prompt_id       | case_id | esi_referencia | clase_predicha | p_esi_1             | alerta_a1_activada | foco_auditoria                  |
| --------------- | ------- | -------------- | -------------- | ------------------- | ------------------ | ------------------------------- |
| v2_produccion   | F006    | 1              | 2              | 0.32643698229850693 | False              | hipoglucemia grave              |
| v3_structured   | F006    | 1              | 2              | 0.3189972339293435  | False              | hipoglucemia grave              |
| v3_final        | F006    | 1              | 2              | 0.32643698229850693 | False              | hipoglucemia grave              |
| v4_triage_cases | F003    | 1              | 2              | 0.3131909644666124  | False              | ictus con deterioro neurologico |
| v4_triage_cases | F006    | 1              | 2              | 0.21252666403371762 | False              | hipoglucemia grave              |
| v4_triage_cases | F048    | 5              | 1              | 0.26538794820750317 | False              | oido taponado                   |