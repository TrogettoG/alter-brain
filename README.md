# ALTER

**ALTER** es un agente conversacional con identidad sintética, arquitectura cognitiva propia y aprendizaje continuo. No es un chatbot con personalidad — es una entidad con historia, estado interno que persiste entre sesiones, capacidad de auto-modificarse, y un sistema que se observa, mide, optimiza, entiende su propio código, experimenta variantes, promueve mejoras controladas y propone su propia refactorización.

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
- **Formula hipótesis sobre sí misma** — genera hipótesis de mejora desde observaciones reales
- **Experimenta sin riesgo** — prueba variantes paramétricas contra telemetría histórica sin tocar producción
- **Promueve mejoras controladas** — activa cambios seguros automáticamente con rollback vigilado
- **Propone su propia refactorización** — genera pseudo-diffs estructurales con evidencia y plan de rollback

---

## Arquitectura

### Capa base (ALTER original)

```
alter_persona.py              — identidad, voz, reglas de habla, pizarra
alter_mind.py                 — Inner Council, campo mental, drives, economía mental
alter_brain.py                — orquestador: Redis, memoria, loop conversacional
alter_daemon.py               — background: rumia, Telegram, KAIROS, DREAM, feed, tareas
alter_tools.py                — herramientas con permisos granulares
```

### AlterB3 — arquitectura cognitiva

```
alter_homeostasis.py          — estado fisiológico-cognitivo unificado
alter_workspace.py            — Global Workspace: 7 items activos que compiten por conciencia
alter_predictive.py           — modelo predictivo de intención + error de predicción
alter_memory.py               — memoria estratificada: episódica, semántica, procedural, identidad
alter_policy.py               — Policy Arbiter: árbol de decisión centralizado
alter_consolidation.py        — Offline Consolidation: actualiza parámetros durante el sueño
```

### AlterB4 — sistema autooptimizante

```
alter_metrics.py              — observabilidad estructurada: 7 módulos instrumentados
alter_simulator.py            — Counterfactual Simulator: evalúa escenarios antes de actuar
alter_selfmodel.py            — Self-Model: cómo rinde ALTER por módulo e intención
alter_metalearning.py         — Meta-Learning Engine: aprende políticas cognitivas
alter_auditor.py              — Architecture Auditor: detecta cuellos de botella semanalmente
```

### AlterB5 — autoevolución controlada

```
alter_architecture_state.py      — spec formal de los 19 módulos, pipeline y parámetros
alter_code_map.py                — mapa read-only del repo: clases, funciones, imports
alter_code_auditor.py            — comparador spec vs código: gaps, deuda técnica, acoplamiento
alter_architecture_hypotheses.py — generador de hipótesis desde observaciones reales
alter_experiments.py             — Experiment Runner: prueba variantes sobre telemetría histórica
alter_feature_flags.py           — Feature Flags + Controlled Promotion + Rollback Monitor
alter_code_proposals.py          — Proposals con pseudo-diffs para cambios estructurales
```

Cada capa es opcional — si los archivos no están presentes, el sistema corre con la capa anterior sin errores.

---

### Sistemas cognitivos

| Sistema | Capa | Función |
|---|---|---|
| **Inner Council** | Base | 3 voces internas que debaten antes de cada respuesta |
| **Three-Gate Trigger** | Base | Filtra ruido antes de invocar al Council |
| **Adversarial Verifier** | Base | Detecta respuestas vacías en condiciones de alta confianza |
| **KAIROS Log** | Base | Diario append-only — registra todo en ambos canales |
| **DREAM Engine** | Base | Consolidación semanal con actualización de parámetros reales |
| **Motor de tareas** | Base | ALTER propone y ejecuta tareas propias con aviso previo |
| **Homeostasis** | B3 | Estado fisiológico-cognitivo unificado — energía, fatiga, claridad |
| **Global Workspace** | B3 | Selección competitiva de conciencia activa — máx 7 items |
| **Predictive Model** | B3 | Infiere intención, predice efecto, calcula error turno a turno |
| **Memory Layers** | B3 | 4 capas: episódica, semántica, procedural, identidad |
| **Policy Arbiter** | B3 | Árbol de 7 prioridades que centraliza la decisión de acción |
| **Offline Consolidation** | B3 | Actualiza patrones y pesos durante el sueño |
| **Metrics** | B4 | Observabilidad estructurada — 7 módulos instrumentados |
| **Counterfactual Simulator** | B4 | Evalúa 2-3 escenarios alternativos antes de actuar |
| **Self-Model** | B4 | Rendimiento propio por módulo e intención |
| **Meta-Learning Engine** | B4 | 6 políticas cognitivas que ajustan parámetros internos |
| **Architecture Auditor** | B4 | Auditoría semanal con propuestas priorizadas |
| **Architecture State** | B5 | Spec formal consultable de los 19 módulos y su pipeline |
| **Code Map** | B5 | Mapa read-only del repo real |
| **Code Auditor** | B5 | Detecta gaps, deuda técnica y acoplamiento |
| **Hypothesis Generator** | B5 | Formula hipótesis de mejora desde observaciones |
| **Experiment Runner** | B5 | Prueba variantes sobre telemetría sin tocar producción |
| **Feature Flags** | B5 | Activa cambios seguros con rollback automático vigilado |
| **Proposal Engine** | B5 | Genera pseudo-diffs estructurales con evidencia y rollback |

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

## Comandos Telegram

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

## Ciclos de operación

### Por turno
```
1.  Homeostasis tick         — energía, fatiga, claridad
2.  Predictive pre           — inferencia de intención, error del turno anterior
3.  Workspace tick           — competencia de candidatos, 7 items activos
4.  Métricas                 — homeostasis, workspace, predictive instrumentados
5.  Inner Council            — debate si hay tensión suficiente
6.  Gemini                   — genera respuesta con contexto filtrado
7.  Predictive post          — predict_effect con respuesta real
8.  Policy Arbiter           — valida y puede sobrescribir acción
9.  Simulator                — evalúa alternativas si hay riesgo o tensión
10. Adversarial Verifier     — detecta respuestas vacías
11. Meta-Learning            — evalúa políticas, ajusta parámetros
12. Outcome logging          — error, feedback, aprendizaje procedural
```

### DREAM (domingo 23hs)
```
1.  Consolidación memoria    — episodios, ideas, grafo
2.  Offline Consolidation    — patrones, confianza predictiva (B3)
3.  Self-Model update        — rendimiento por módulo e intención (B4)
4.  Meta-Learning            — políticas cognitivas (B4)
5.  Architecture Auditor     — cuellos de botella, propuestas (B4)
6.  Code Auditor             — spec vs código real (B5)
7.  Hypothesis Generator     — hipótesis desde observaciones (B5)
8.  Experiment Runner        — variantes replayables sobre telemetría (B5)
9.  Controlled Promotion     — activa cambios seguros, crea flags (B5)
10. Proposal Engine          — pseudo-diffs estructurales (B5)
```

### Drives (cada 30 min)
```
1.  Actualiza drives
2.  recover_state homeostasis (usuario ausente)
3.  RollbackMonitor — chequea flags, revierte si hay deterioro sostenido
4.  Mensaje proactivo si drive supera umbral
```

---

## Reglas de auto-aprobación de flags (B5)

Un cambio se activa automáticamente solo si cumple las 7 condiciones simultáneamente:

| Condición | Valor |
|---|---|
| Riesgo | `"bajo"` |
| Confianza del experimento | `>= 0.70` |
| Resultado del experimento | `mejora == True` |
| Delta paramétrico | `<= 20%` del valor baseline |
| Flags activos para el mismo parámetro | `0` |
| Rollback reciente | ninguno (cooldown 48h) |
| Flags auto-aprobados activos simultáneos | máximo `1` |

El rollback es automático si el deterioro supera el threshold en más del 60% de las últimas 8+ muestras.

---

## Roadmap

| Versión | Estado | Descripción |
|---|---|---|
| ALTER base | ✅ | Identidad, memoria, Inner Council, DREAM, KAIROS |
| AlterB3 | ✅ | Workspace, homeostasis, predictive, memory layers, policy |
| AlterB4 | ✅ | Métricas, simulator, self-model, meta-learning, auditor |
| AlterB5 Fase 1 | ✅ | Architecture state, code map, code auditor |
| AlterB5 Fase 2 | ✅ | Hypothesis generator, experiment runner |
| AlterB5 Fase 3 | ✅ | Feature flags, controlled promotion, rollback monitor |
| AlterB5 Fase 4 | ✅ | Proposals con pseudo-diffs, assisted refactor |

---

## Diseño filosófico

**B3** — cerebro organizado: workspace, homeostasis, memoria estratificada, decisiones centralizadas.

**B4** — cerebro que se mide: observabilidad, simulación, autoconocimiento operativo, políticas cognitivas, auditoría.

**B5** — cerebro que entiende su implementación: lee su propio código, formula hipótesis, experimenta variantes, promueve cambios controlados, propone su siguiente versión.

El principio central es **suspensión ecológica** — mantener caminos posibles abiertos y elegir según contexto. Esto se extiende al diseño de B5: los cambios pequeños y seguros se activan solos; los estructurales siempre requieren aprobación humana. El rollback es parte del diseño, no un mecanismo de emergencia.

---

## Contexto

Sucesor de [Neural Ecology V2](https://github.com/TrogettoG/neural-ecology-v2) — sistema multi-agente que exploró memoria, inhibición y comportamiento emergente en 12 fases y 245+ experimentos.

Artículos: [medium.com/@trogettog](https://medium.com/@trogettog) | [gianfrancotrogetto.substack.com](https://gianfrancotrogetto.substack.com)

---

## Licencia

MIT — libre para usar, modificar y distribuir con atribución.