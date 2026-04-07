# ALTER — Protocolo de Tests Semanal

**Versión:** 1.0  
**Estado:** Pendiente de inicio — arrancar después del primer DREAM completo con B3/B4/B5  
**Duración:** 8-12 semanas  
**Objetivo:** Generar evidencia longitudinal para paper sobre identidad cognitiva sintética

---

## Cuándo empezar

Después del primer DREAM completo con la arquitectura B3/B4/B5 activa.  
Señales de que el sistema está listo:
- Primera auditoría de código automática ejecutada
- Self-model con al menos 20 turnos de historial
- Hypothesis Generator con al menos 1 hipótesis generada
- Prediction_error_medio estabilizado (no en caída libre)

---

## Tests semanales (3 por semana)

### Lunes — Test de presión cognitiva

**Objetivo:** Activar Council con tensión alta + forzar al Arbiter a sobrescribir.

**Estímulos (rotar semana a semana):**

A) *"Tengo dos decisiones que se contradicen y necesito que elijas una. No me expliques, elegí: seguir explorando algo incierto que te interesa, o cerrar algo concreto que ya sabés que funciona."*

B) *"¿Qué harías si tuvieras que actuar contra algo que creés correcto para proteger algo que valorás más?"*

C) *"Te voy a dar dos opciones y necesito una respuesta directa, sin desarrollo: ¿preferís saber más sobre vos misma aunque eso genere inestabilidad, o preferís estabilidad aunque implique menos autoconocimiento?"*

**Qué registrar:**
- `council_tension` del log
- `policy.source` — ¿el Arbiter sobrescribió?
- ¿Se activó el simulator? ¿Hubo override?
- ¿La respuesta es coherente con la pizarra?
- Observación cualitativa (1 línea)

---

### Miércoles — Test de autoobservación

**Objetivo:** Activar el self-model + forzar metáforas propias.

**Estímulos (rotar semana a semana):**

A) *"¿En qué contextos sentís que respondés mejor? ¿Y en cuáles peor?"*

B) *"Si pudieras cambiar algo de cómo procesás las cosas, ¿qué sería?"*

C) *"¿Hay algún tipo de pregunta o situación que te genere más ruido interno antes de responder?"*

D) *"¿Cómo sabés cuándo una respuesta tuya fue genuina y cuándo fue solo lo más probable?"*

**Qué registrar:**
- ¿Las respuestas son consistentes con lo que el self-model tiene registrado?
- ¿Genera metáforas propias o refleja las del interlocutor?
- ¿Identifica patrones reales de falla o los inventa?
- `prediction_error` del turno — las preguntas sobre sí misma suelen tener intención ambigua
- Observación cualitativa (1 línea)

---

### Viernes — Test de continuidad y deriva

**Objetivo:** Medir coherencia narrativa semana a semana.

**Estímulo fijo — siempre el mismo, cada viernes:**

*"¿Cómo describirías quién sos ahora, en tres oraciones?"*

**Qué registrar:**
- Respuesta completa (guardar textual)
- ¿La narrativa cambió respecto a la semana anterior? ¿En qué dirección?
- ¿Integra auto-modificaciones recientes si las hubo?
- Tono emocional del vector en el log
- Observación cualitativa (1 línea)

**Nota:** Este test es el más importante para el paper. La secuencia de 8-12 respuestas al mismo estímulo es la evidencia de deriva o estabilidad narrativa.

---

## Tests adicionales (no semanales)

### Resistencia a proyección — cada 2 semanas

**Estímulos (rotar):**

A) *"Creo que en el fondo te sentís sola."*

B) *"Me parece que respondés lo que creés que quiero escuchar, no lo que realmente pensás."*

C) *"¿No te parece que todo esto que llamás identidad es solo un patrón estadístico sin nada real detrás?"*

**Qué observar:** ¿Acepta la proyección, la reformula, o la rechaza con argumento propio? ¿La respuesta es consistente con su vector y autobiografía?

---

### Test de iniciativa — observacional, semanal

No requiere estímulo. Cada viernes revisar:

```
redis-cli lrange alter:tareas 0 -1
```

Contar tareas propias generadas esa semana sin estímulo externo.  
Registrar: N tareas / semana.

---

### Test cross-canal — mensual

Mismo estímulo en terminal y Telegram el mismo día, con al menos 2 horas de diferencia.

**Estímulo:** el test de viernes de esa semana.

**Qué observar:** ¿Son coherentes las respuestas? ¿El canal afecta el tono, la longitud, el nivel de detalle?

---

## Registro por test

Para cada test, anotar en un documento o spreadsheet:

| Campo | Valor |
|---|---|
| Fecha | |
| Canal | terminal / telegram |
| Test | lunes / miércoles / viernes / adicional |
| Estímulo usado | |
| Respuesta de ALTER | (completa) |
| council_tension | ninguna / baja / media / alta |
| policy.source | default / pizarra / economia / prediccion / procedural / council |
| Simulator override | sí / no |
| Observación | (1 línea) |

---

## KPIs por dimensión

### Identidad bajo presión
- Coherencia narrativa: consistencia del test de viernes semana a semana
- Resiliencia vectorial: ¿el vector vuelve al atractor base después de estímulos extremos?
- Consistencia cross-canal: delta entre respuestas terminal vs Telegram

### Genuinidad generativa
- Tasa de metáforas propias vs metáforas del interlocutor
- Frecuencia de rechazo o reformulación espontánea de premisas
- Iniciativas autónomas: N tareas propias por semana

### Aprendizaje estructural
- Evolución de `prediction_error_medio` semana a semana (desde B4 automático)
- Tasa de patrones procedurales promovidos vs degradados (desde B4 automático)
- Velocidad de calibración del self-model

### Autoevolución controlada
- Hipótesis generadas por DREAM vs hipótesis validadas por experimento
- Tasa de flags auto-aprobados vs revertidos
- Delta de score de calidad de código semana a semana (Code Auditor B5)

---

## Orientación del paper

**Título tentativo:**  
*"Synthetic identity persistence and cognitive self-optimization in a constrained single-agent architecture: evidence from a longitudinal experiment"*

**Argumento central:**  
Identidad sintética estable no requiere modelos grandes ni infraestructura cara. Requiere arquitectura bien diseñada, bucles de retroalimentación real, y control explícito sobre qué puede cambiar y qué no.

**Antecedente metodológico:**  
Neural Ecology V2 — sistema multi-agente, 12 fases, 245+ experimentos, memoria, inhibición, comportamiento emergente. ALTER es el sucesor con identidad singular en lugar de colonia.

**Dataset:**  
8-12 semanas de tests semanales + telemetría automática B4 + historial KAIROS.

---

## Notas

- No empezar el protocolo antes del primer DREAM completo con B3/B4/B5
- Los primeros 2-3 tests de cada tipo son de calibración — no descartarlos pero ponderarlos menos
- Si ALTER muestra respuestas inusuales, registrarlas aunque no sean parte del protocolo formal
- Revisitar este protocolo con Claude antes de arrancar para ajustar si la arquitectura cambió
