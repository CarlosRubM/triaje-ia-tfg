# Tuning orientado a ESI 4/5

Busqueda nocturna sobre LightGBM+BERT/SVD usando solo train y validacion OOF agrupada por paciente. El test temporal no se ha usado.

## Comparacion OOF

```csv
modelo,score_compuesto,macro_f1,weighted_f1,balanced_accuracy,f1_a1,f1_a2,f1_a3,f1_a4,f1_a5,recall_a4,precision_a4,recall_a5,precision_a5,infratriaje_total,sobretriaje_total,infratriaje_critico_a1,delta_score_compuesto_vs_actual,delta_macro_f1_vs_actual,delta_f1_a4_vs_actual,delta_f1_a5_vs_actual,delta_f1_a1_vs_actual,delta_f1_a2_vs_actual,delta_infratriaje_critico_a1_vs_actual
lightgbm_final_bert_actual_oof,0.511160,0.567940,0.692585,0.586992,0.669482,0.662288,0.740868,0.500692,0.266368,0.615103,0.422167,0.255365,0.278363,0.170082,0.140908,0.018097,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
tail_tuned_training_argmax_oof,0.514454,0.570332,0.692279,0.588298,0.669389,0.659944,0.742032,0.497942,0.282353,0.603701,0.423713,0.283262,0.281450,0.171912,0.139049,0.018713,0.003294,0.002392,-0.002750,0.015985,-0.000092,-0.002345,0.000616
tail_tuned_policy_posthoc_oof,0.515828,0.571364,0.691226,0.588948,0.671576,0.660727,0.739312,0.498063,0.287140,0.622461,0.415105,0.277897,0.297018,0.175747,0.136803,0.019062,0.004669,0.003424,-0.002628,0.020771,0.002095,-0.001561,0.000966
```

## Mejor configuracion de entrenamiento

- Trial: 84
- Score compuesto: 0.514454
- Macro F1: 0.570332
- F1 A4: 0.497942
- F1 A5: 0.282353
- F1 A1: 0.669389
- F1 A2: 0.659944
- Infratriaje critico A1: 0.018713

## Mejor politica post-hoc OOF

- Multiplicadores: [0.9839, 1.031, 1.0332, 1.0925, 1.0009]
- Score compuesto: 0.515828
- Macro F1: 0.571364
- F1 A4: 0.498063
- F1 A5: 0.287140

## Decision metodologica

El candidato no cumple todos los criterios OOF de adopcion. Se documenta como experimento de sensibilidad y se mantiene el modelo congelado actual.

No se sobrescribe `models/lgbm_bert_final.joblib`. La politica post-hoc es diagnostica y no modifica la politica A1 final.

## Artefactos

- Trials: `reports\hyperparameter_tuning\lgbm_bert_tail_tuning_trials.csv`
- Parametros: `reports\hyperparameter_tuning\lgbm_bert_tail_tuning_best_params.json`
- Comparacion: `reports\hyperparameter_tuning\lgbm_bert_tail_tuning_oof_comparison.csv`
- Resumen: `reports\hyperparameter_tuning\lgbm_bert_tail_tuning_summary.md`

## Matrices de confusion OOF

### Baseline actual

,1,2,3,4,5
1,13318,4565,1391,94,3
2,5766,72856,31238,1381,21
3,1258,30936,129753,17603,125
4,58,367,8052,14295,468
5,15,27,164,488,238


### Mejor entrenamiento tail

,1,2,3,4,5
1,13112,4663,1501,93,2
2,5496,72312,32037,1397,20
3,1123,30522,130752,17136,142
4,60,362,8278,14030,510
5,14,25,173,456,264


### Mejor politica post-hoc

,1,2,3,4,5
1,12995,4765,1501,108,2
2,5236,72488,31999,1521,18
3,1032,30514,129730,18277,122
4,52,366,7885,14466,471
5,14,24,158,477,259
