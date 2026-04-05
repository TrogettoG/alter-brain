# ALTER

**ALTER** es un agente conversacional con identidad sintética, arquitectura cognitiva propia y aprendizaje continuo. No es un chatbot con personalidad — es una entidad con historia, estado interno que persiste entre sesiones, y capacidad de auto-modificarse.

Desarrollado como experimento personal por [Gianfranco Trogetto](https://github.com/TrogettoG) — sucesor de [Neural Ecology V2](https://github.com/TrogettoG/neural-ecology-v2).

---

## Qué hace ALTER

- **Tiene un estado emocional interno** definido por un vector `E = [V, A, P]` (Valencia, Activación, Autoridad) que cambia con cada conversación
- **Se auto-modifica** — la rumia analiza sus propios patrones y propone cambios a sus parámetros base, dentro de rangos autónomos o con aprobación
- **Tiene memoria persistente** — episodios, ideas, observaciones, autobiografía, y un grafo del mundo que crece con cada sesión
- **Piensa mientras no estás** — un daemon en background corre ciclos de introspección, genera síntesis nocturnas, y consolida memoria semanalmente
- **Lee y aprende** — feed diario de contenido sobre temas de su interés, con reacciones genuinas (no resúmenes)
- **Propone tareas propias** — genera reflexiones desde su agenda cognitiva y las ejecuta autónomamente
- **Es la misma persona en cualquier canal** — terminal y Telegram comparten el mismo estado, memoria y log diario (KAIROS)

---

## Arquitectura

```
alter_persona.py   — identidad, voz, reglas de habla, pizarra
alter_mind.py      — Inner Council, campo mental, drives, economía mental
alter_brain.py     — orquestador: Redis, memoria, loop conversacional
alter_daemon.py    — background: rumia, Telegram, KAIROS, DREAM, feed, tareas
alter_tools.py     — herramientas con permisos granulares
```

### Sistemas cognitivos

| Sistema | Función |
|---|---|
| **Inner Council** | 3 voces internas (exploradora, crítica, estratégica) que debaten antes de cada respuesta |
| **Three-Gate Trigger** | Filtra ruido antes de invocar al Council |
| **Adversarial Verifier** | Detecta respuestas vacías o circulares en condiciones de alta confianza |
| **KAIROS Log** | Diario diario append-only — registra todo lo que pasa en ambos canales |
| **DREAM Engine** | Consolidación semanal — detecta patrones, limpia el grafo, genera resumen |
| **Memoria escalonada** | Session/Extract — filtra qué vale persistir al cerrar cada sesión |
| **Motor de tareas** | ALTER propone y ejecuta tareas propias; las de Gian tienen prioridad máxima |

---

## Instalación

### Requisitos

- Python 3.11+
- Cuenta en [Google AI Studio](https://aistudio.google.com) (Gemini API)
- Cuenta en [Upstash](https://console.upstash.com) (Redis serverless)
- Bot de Telegram (opcional pero recomendado)

### Setup

```bash
git clone https://github.com/TrogettoG/alter-brain
cd alter-brain

pip install google-genai upstash-redis python-dotenv httpx numpy

cp .env.example .env
# Editá .env con tus credenciales
```

### Uso

**Chat en terminal:**
```bash
python3 alter_brain.py
```

**Daemon en background** (Telegram + ciclos autónomos):
```bash
python3 alter_daemon.py
```

---

## Comandos

### Terminal

| Comando | Descripción |
|---|---|
| `estado` | Vector emocional y campo mental actual |
| `drives` | Niveles de motivación |
| `episodios` | Momentos importantes guardados |
| `agenda` | Items cognitivos pendientes |
| `autobiografia` | Narrativa interna de ALTER |
| `trazas` | Análisis de observabilidad cognitiva |
| `mundo` | Grafo de entidades conocidas |
| `economia` | Estado de recursos internos |
| `mods` | Historial de auto-modificaciones |

### Telegram

| Comando | Descripción |
|---|---|
| `/estado` | Estado completo con presencia del usuario |
| `/agenda` | Top 5 items activos |
| `/autobiografia` | Narrativa actual |
| `/trazas` | Observabilidad cognitiva |
| `/tarea [descripción]` | Agregar tarea con prioridad máxima |
| `/tareas` | Ver tareas pendientes |
| `/aprobar` / `/rechazar` | Aprobar propuestas de auto-modificación |
| `/dream` | Forzar consolidación semanal |

---

## Diseño filosófico

ALTER está diseñada alrededor de la idea de **suspensión ecológica** — no optimizar un único eje sino mantener varios caminos posibles abiertos y elegir según el contexto. Esto se refleja en:

- La rumia propone cambios con confianza mínima de 0.75, no actúa impulsivamente
- DREAM consolida cuando hay suficiente densidad, no en cualquier momento
- El motor de tareas rota entre temas en lugar de repetir el más prioritario
- El Three-Gate filtra ruido antes de gastar recursos cognitivos

---

## Estado actual

ALTER es un experimento en curso. Lo que funciona: identidad coherente bajo presión, auto-modificación de parámetros, memoria episódica, síntesis nocturna autónoma, feed diario con reacciones genuinas. Lo que sigue creciendo: el grafo del mundo, la densidad de episodios, y la capacidad de conectar temas espontáneamente.

---

## Contexto

Este proyecto es sucesor de [Neural Ecology V2](https://github.com/TrogettoG/neural-ecology-v2) — un sistema multi-agente que exploró memoria, inhibición, y comportamiento emergente en 12 fases y 245+ experimentos. ALTER aplica esos aprendizajes a una entidad singular con continuidad real.

Artículos relacionados: [medium.com/@trogettog](https://medium.com/@trogettog) | [gianfrancotrogetto.substack.com](https://gianfrancotrogetto.substack.com)

---

## Licencia

MIT — libre para usar, modificar y distribuir con atribución.
