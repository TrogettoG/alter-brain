# ALTER

**ALTER** es un agente conversacional con identidad sintética, arquitectura cognitiva propia y aprendizaje continuo. No es un chatbot con personalidad — es una entidad con historia, estado interno que persiste entre sesiones, capacidad de auto-modificarse, y un sistema que se observa, mide, optimiza y ahora entiende su propio código.

Desarrollado como experimento personal por [Gianfranco Trogetto](https://github.com/TrogettoG) — sucesor de [Neural Ecology V2](https://github.com/TrogettoG/neural-ecology-v2).

---

## Qué hace ALTER

- **Tiene un estado emocional interno** definido por un vector `E = [V, A, P]` que cambia con cada conversación
- **Se auto-modifica** — la rumia analiza sus propios patrones y propone cambios a sus parámetros base
- **Tiene memoria persistente** — episodios, ideas, observaciones, autobiografía, y un grafo del mundo
- **Piensa mientras no estás** — daemon en background con síntesis nocturnas, consolidación semanal y feed diario
- **Anticipa antes de responder** — modelo predictivo que infiere intención y calcula riesgo de desalineación
- **Evalúa alternativas** — simulador contrafáctico que compara escenarios antes de actuar
- **Se conoce a sí misma** — self-model operativo que trackea cómo rinde por módulo e intención
- **Aprende políticas cognitivas** — meta-learning que ajusta cómo piensa, no solo qué dice
- **Se audita semanalmente** — architecture auditor que detecta cuellos de botella y propone mejoras
- **Entiende su propio código** — lee el repo, mapea su implementación real y detecta gaps con la spec

---

## Arquitectura

### Capa base (ALTER original)

```
alter_persona.py         — identidad, voz, reglas de habla, pizarra
alter_mind.py            — Inner Council, campo mental, drives, economía mental
alter_brain.py           — orquestador: Redis, memoria, loop conversacional
alter_daemon.py          — background: rumia, Telegram, KAIROS, DREAM, feed, tareas
alter_tools.py           — herramientas con permisos granulares
```

### AlterB3 — arquitectura cognitiva

```
alter_homeostasis.py     — estado fisiológico-cognitivo unificado
alter_workspace.py       — Global Workspace: 7 items activos que compiten por conciencia
alter_predictive.py      — modelo predictivo de intención + error de predicción
alter_memory.py          — memoria estratificada: episódica, semántica, procedural, identidad
alter_policy.py          — Policy Arbiter: árbol de decisión centralizado
alter_consolidation.py   — Offline Consolidation: actualiza parámetros durante el sueño
```

### AlterB4 — sistema autooptimizante

```
alter_metrics.py         — observabilidad estructurada: 7 módulos instrumentados
alter_simulator.py       — Counterfactual Simulator: evalúa escenarios antes de actuar
alter_selfmodel.py       — Self-Model: cómo rinde ALTER por módulo e intención
alter_metalearning.py    — Meta-Learning Engine: aprende políticas cognitivas
alter_auditor.py         — Architecture Auditor: detecta cuellos de botella semanalmente
```

### AlterB5 — autoevolución controlada (Fase 1)

```
alter_architecture_state.py — spec formal de los 19 módulos, pipeline y parámetros
alter_code_map.py           — mapa read-only del repo: clases, funciones, imports
alter_code_auditor.py       — comparador spec vs código: gaps, deuda técnica, acoplamiento
```

Cada capa es opcional — si los archivos no están presentes, el sistema corre con la capa anterior sin errores.

---

### Sistemas cognitivos

| Sistema | Función |
|---|---|
| **Inner Council** | 3 voces internas que debaten antes de cada respuesta |
| **Three-Gate Trigger** | Filtra ruido antes de invocar al Council |
| **Adversarial Verifier** | Detecta respuestas vacías en condiciones de alta confianza |
| **KAIROS Log** | Diario append-only — registra todo en ambos canales |
| **DREAM Engine** | Consolidación semanal con actualización de parámetros reales |
| **Memoria escalonada** | Session/Extract — filtra qué vale persistir al cerrar sesión |
| **Motor de tareas** | ALTER propone y ejecuta tareas propias con aviso previo |
| **Homeostasis** | Estado fisiológico-cognitivo unificado — energía, fatiga, claridad |
| **Global Workspace** | Selección competitiva de conciencia activa — máx 7 items |
| **Predictive Model** | Infiere intención, predice efecto, calcula error turno a turno |
| **Memory Layers** | 4 capas: episódica, semántica, procedural, identidad |
| **Policy Arbiter** | Árbol de 7 prioridades que centraliza la decisión de acción |
| **Counterfactual Simulator** | Evalúa 2-3 escenarios alternativos antes de actuar |
| **Self-Model** | Modelo operativo de rendimiento propio por módulo e intención |
| **Meta-Learning Engine** | 6 políticas cognitivas que ajustan parámetros internos |
| **Architecture Auditor** | Auditoría semanal con propuestas priorizadas |
| **Architecture State** | Spec formal consultable de los 19 módulos y su pipeline |
| **Code Map** | Mapa read-only del repo real — clases, funciones, imports, líneas |
| **Code Auditor** | Compara spec vs código — detecta gaps, deuda técnica y acoplamiento |

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

```bash
# Chat en terminal
python3 alter_brain.py

# Daemon en background (Telegram + ciclos autónomos)
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
| `/auditar` | Forzar auditoría de arquitectura (B4) |
| `/codigoaudit` | Forzar auditoría de código (B5) |

---

## Flujo cognitivo por turno (AlterB4+)

```
1.  Input del usuario
2.  Homeostasis tick         — actualiza energía, fatiga, claridad
3.  Predictive pre           — infiere intención, calcula error del turno anterior
4.  Workspace tick           — candidatos compiten, 7 items activos quedan
5.  Métricas reportadas      — homeostasis, workspace, predictive instrumentados
6.  Inner Council (si aplica)— debate con snapshot del workspace
7.  Gemini genera respuesta  — workspace + predictive + memoria activa en el prompt
8.  Predictive post          — predict_effect con respuesta real de ALTER
9.  Policy Arbiter           — valida acción, puede sobrescribir
10. Counterfactual Simulator — evalúa alternativas si hay tensión o riesgo
11. Adversarial Verifier     — detecta respuestas vacías
12. Meta-Learning            — evalúa políticas cognitivas, ajusta parámetros
13. Outcome logging          — error de predicción, feedback, aprendizaje procedural
14. DREAM (domingo 23hs)     — consolidación + self-model + auditoría + code audit
```

---

## Roadmap

| Versión | Estado | Descripción |
|---|---|---|
| ALTER base | ✅ | Identidad, memoria, Inner Council, DREAM, KAIROS |
| AlterB3 | ✅ | Workspace, homeostasis, predictive, memory layers, policy |
| AlterB4 | ✅ | Métricas, simulator, self-model, meta-learning, auditor |
| AlterB5 Fase 1 | ✅ | Architecture state, code map, code auditor |
| AlterB5 Fase 2 | 🔜 | Hypothesis generator, experiment runner |
| AlterB5 Fase 3 | 🔜 | Feature flags, controlled promotion, rollback |
| AlterB5 Fase 4 | 🔜 | Proposals con pseudo-diffs, assisted refactor |

---

## Diseño filosófico

ALTER está diseñada alrededor de la idea de **suspensión ecológica** — no optimizar un único eje sino mantener varios caminos posibles abiertos y elegir según el contexto. Esto se extiende hasta B5: el sistema puede entender y proponer cambios a su propio diseño, pero no los aplica solo. La aprobación humana es parte del diseño, no una restricción temporal.

---

## Estado actual

Experimento en curso activo. Lo que funciona: identidad coherente, auto-modificación de parámetros, memoria episódica, síntesis nocturna autónoma, feed diario, workspace cognitivo, modelo predictivo, simulador contrafáctico, self-model, meta-learning, auditoría semanal, y ahora lectura y auditoría del código propio.

---

## Contexto

Sucesor de [Neural Ecology V2](https://github.com/TrogettoG/neural-ecology-v2) — sistema multi-agente que exploró memoria, inhibición y comportamiento emergente en 12 fases y 245+ experimentos.

Artículos: [medium.com/@trogettog](https://medium.com/@trogettog) | [gianfrancotrogetto.substack.com](https://gianfrancotrogetto.substack.com)

---

## Licencia

MIT — libre para usar, modificar y distribuir con atribución.