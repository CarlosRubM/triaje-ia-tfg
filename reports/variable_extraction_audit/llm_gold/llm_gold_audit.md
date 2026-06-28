# Auditoria LLM con casos oro

Generado: 2026-06-06T00:13:42
Backend: `ollama`
Modelo: `llama3.1:8b-instruct-q4_K_M`
Casos: 8
Pasados: 8
Fallidos: 0
Estado: PASS

## Casos

### G001 - ambulance chest pain - PASS

Foco: arrival ambulance, vitals, pain, cardiac medication
Tiempo: 8.817 s

### G002 - helicopter trauma - PASS

Foco: arrival helicopter, trauma, shock features
Tiempo: 7.544 s

### G003 - walk in green zone - PASS

Foco: arrival autonomous, negative medication, green zone
Tiempo: 7.778 s

### G004 - unknown arrival sepsis - PASS

Foco: arrival unknown only when not stated
Tiempo: 8.688 s

### G005 - respiratory medication and inhaled steroid exclusion - PASS

Foco: inhaled respiratory therapy must not become systemic steroid
Tiempo: 9.441 s

### G006 - opioid benzodiazepine respiratory risk - PASS

Foco: compound medication risk
Tiempo: 8.848 s

### G007 - female with diabetes and insulin - PASS

Foco: sex, metabolic history, insulin
Tiempo: 8.818 s

### G008 - other sex administrative request - PASS

Foco: all sex values, administrative low acuity, missing vitals
Tiempo: 7.371 s

## Nota metodologica

Estos casos son sinteticos y sirven para auditar extraccion, no para seleccionar modelo, umbrales, calibracion ni politica A1.
