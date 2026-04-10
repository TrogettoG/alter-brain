"""
alter_architecture_state.py — Architecture State de AlterB5 (Fase 1)

Modelo formal de cómo está armado ALTER hoy.
No es documentación — es un grafo vivo que ALTER puede consultar.

Diferencia con el README:
    El README describe para humanos.
    Architecture State describe para ALTER — estructura consultable,
    con parámetros reales, dependencias y estado de cada módulo.

Quién lo usa:
    Code Auditor (B5)        — compara spec vs código real
    Hypothesis Generator (B5 Fase 2) — sabe qué puede cambiar
    Architecture Auditor (B4) — contexto de qué existe

No se genera dinámicamente.
Se mantiene como fuente de verdad y se actualiza cuando
el Architecture Auditor detecta cambios significativos.

Redis keys:
    alter:b5:architecture_state  — spec serializada
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class ParametroSpec:
    nombre:      str
    valor:       object   # valor actual configurado
    tipo:        str      # "float" | "int" | "bool" | "str"
    rango:       Optional[list] = None   # [min, max] si aplica
    descripcion: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModuleSpec:
    nombre:          str
    archivo:         str
    clase_principal: str
    descripcion:     str
    inputs:          list   # de qué módulos recibe señales
    outputs:         list   # a qué módulos exporta
    parametros:      list   # list[ParametroSpec]
    estado:          str    # "activo" | "degradado" | "inactivo" | "opcional"
    capa:            str    # "base" | "b3" | "b4" | "b5"
    redis_keys:      list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArchitectureState:
    version:      str
    last_updated: str
    modules:      list   # list[ModuleSpec]
    pipeline:     list   # orden de ejecución por turno
    flags:        dict   # feature flags activos
    dependencias_criticas: list  # módulos sin los cuales el sistema no arranca

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureState":
        modules = [ModuleSpec(**m) for m in d.get("modules", [])]
        return cls(
            version      = d.get("version", ""),
            last_updated = d.get("last_updated", ""),
            modules      = modules,
            pipeline     = d.get("pipeline", []),
            flags        = d.get("flags", {}),
            dependencias_criticas = d.get("dependencias_criticas", []),
        )

    def get_module(self, nombre: str) -> Optional[ModuleSpec]:
        return next((m for m in self.modules if m.nombre == nombre), None)

    def get_by_capa(self, capa: str) -> list:
        return [m for m in self.modules if m.capa == capa]

    def get_activos(self) -> list:
        return [m for m in self.modules if m.estado == "activo"]


# ============================================================
# SPEC ACTUAL DE ALTER (fuente de verdad)
# ============================================================

def _modules_base() -> list:
    """Módulos de la capa base: persona, mind, brain, daemon, tools."""
    return [
        ModuleSpec(
            nombre="persona", archivo="alter_persona.py",
            clase_principal="(funciones)",
            descripcion="Identidad, voz, reglas de habla, pizarra, prompt unificado.",
            inputs=[], outputs=["brain"],
            parametros=[
                ParametroSpec("MODELO", "gemini-2.5-flash-lite", "str",
                              descripcion="Modelo Gemini usado para generación"),
            ],
            estado="activo", capa="base",
            redis_keys=["alter:config:pizarra"],
        ),
        ModuleSpec(
            nombre="mind", archivo="alter_mind.py",
            clase_principal="(funciones)",
            descripcion="Inner Council, campo mental, drives, economía mental, horario.",
            inputs=["brain"], outputs=["brain"],
            parametros=[
                ParametroSpec("ECONOMIA_DEFAULT", {}, "dict",
                              descripcion="Valores iniciales de economía mental"),
                ParametroSpec("VENTANAS_AUTONOMIA", {}, "dict",
                              descripcion="Rangos de auto-modificación permitidos"),
            ],
            estado="activo", capa="base",
            redis_keys=[],
        ),
        ModuleSpec(
            nombre="brain", archivo="alter_brain.py",
            clase_principal="AlterBrain",
            descripcion="Orquestador principal. Redis, memoria, loop conversacional, integración B3/B4.",
            inputs=["persona", "mind", "homeostasis", "workspace", "predictive",
                    "memory", "policy", "metrics", "simulator", "selfmodel",
                    "metalearning", "auditor"],
            outputs=["respuesta"],
            parametros=[
                ParametroSpec("ALTERB3_ENABLED", True, "bool",
                              descripcion="Activa capas B3/B4"),
            ],
            estado="activo", capa="base",
            redis_keys=["alter:params", "alter:self_mods", "alter:drives",
                        "alter:agenda", "alter:autobiografia", "alter:reputacion",
                        "alter:consecuencias"],
        ),
        ModuleSpec(
            nombre="daemon", archivo="alter_daemon.py",
            clase_principal="(funciones async)",
            descripcion="Background: rumia, Telegram, KAIROS, DREAM, feed, tareas, burst.",
            inputs=["brain", "redis"], outputs=["telegram", "redis"],
            parametros=[
                ParametroSpec("INTERVALO_DRIVES_SEG", 1800, "int",
                              descripcion="Segundos entre ciclos de drives"),
                ParametroSpec("INTERVALO_RUMIA_SEG", 7200, "int",
                              descripcion="Segundos entre ciclos de rumia"),
                ParametroSpec("UMBRAL_INICIATIVA", 0.70, "float", [0.5, 1.0],
                              descripcion="Drive mínimo para mensaje proactivo"),
                ParametroSpec("HORA_FEED", 7, "int",
                              descripcion="Hora del feed diario"),
                ParametroSpec("HORA_SINTESIS", 22, "int",
                              descripcion="Hora de síntesis nocturna KAIROS"),
                ParametroSpec("HORA_DREAM", 23, "int",
                              descripcion="Hora de consolidación DREAM"),
                ParametroSpec("HORA_BURST", 14, "int",
                              descripcion="Hora del burst diario de calibración"),
            ],
            estado="activo", capa="base",
            redis_keys=["alter:daemon:ultimo_inicio", "alter:drives",
                        "alter:kairos:sintesis_hoy", "alter:dream:resumen_semana",
                        "alter:tareas"],
        ),
        ModuleSpec(
            nombre="tools", archivo="alter_tools.py",
            clase_principal="(funciones)",
            descripcion="Herramientas con permisos granulares: web_search, read_file, write_file, run_python.",
            inputs=["brain"], outputs=["brain"],
            parametros=[
                ParametroSpec("PYTHON_TIMEOUT", 15, "int",
                              descripcion="Timeout para ejecución de código"),
            ],
            estado="activo", capa="base",
            redis_keys=["alter:tool_log", "alter:skills"],
        ),
    ]


def _modules_b3() -> list:
    """Módulos de la capa B3: homeostasis, workspace, predictive, memory, policy, consolidation."""
    return [
        ModuleSpec(
            nombre="homeostasis", archivo="alter_homeostasis.py",
            clase_principal="HomeostasisState",
            descripcion="Estado fisiológico-cognitivo unificado: energía, fatiga, claridad, presión, curiosidad.",
            inputs=["brain"], outputs=["brain", "workspace", "policy", "simulator"],
            parametros=[
                ParametroSpec("UMBRAL_AUSENCIA_SEG", 300, "int",
                              descripcion="Segundos sin actividad = usuario ausente"),
            ],
            estado="activo", capa="b3",
            redis_keys=["alter:homeostasis:state"],
        ),
        ModuleSpec(
            nombre="workspace", archivo="alter_workspace.py",
            clase_principal="GlobalWorkspace",
            descripcion="Global Workspace: selección competitiva de conciencia activa.",
            inputs=["brain", "homeostasis", "predictive", "memory"],
            outputs=["brain", "policy", "simulator"],
            parametros=[
                ParametroSpec("MAX_ITEMS", 7, "int", [3, 10],
                              descripcion="Máximo items activos simultáneos"),
                ParametroSpec("EVICTION_THRESHOLD", 0.15, "float", [0.05, 0.30],
                              descripcion="Prioridad mínima antes de evicción"),
                ParametroSpec("MERGE_OVERLAP_THRESHOLD", 0.55, "float", [0.3, 0.8],
                              descripcion="Overlap para merge semántico"),
                ParametroSpec("OVERRIDE_THRESHOLD", 0.15, "float", [0.05, 0.30],
                              descripcion="Delta score para recomendar override"),
            ],
            estado="activo", capa="b3",
            redis_keys=["alter:workspace:state", "alter:workspace:log"],
        ),
        ModuleSpec(
            nombre="predictive", archivo="alter_predictive.py",
            clase_principal="PredictiveState",
            descripcion="Modelo predictivo: infiere intención, predice efecto, calcula error.",
            inputs=["brain", "homeostasis"], outputs=["brain", "workspace", "policy"],
            parametros=[],
            estado="activo", capa="b3",
            redis_keys=["alter:predictive:state", "alter:predictive:errors"],
        ),
        ModuleSpec(
            nombre="memory", archivo="alter_memory.py",
            clase_principal="MemorySystem",
            descripcion="Memoria estratificada: episódica, semántica, procedural, identidad.",
            inputs=["brain"], outputs=["brain", "workspace"],
            parametros=[
                ParametroSpec("THRESHOLD_ERROR", 0.60, "float", [0.4, 0.8],
                              descripcion="Error mínimo para aprendizaje procedural"),
                ParametroSpec("MAX_PATTERNS", 30, "int", [10, 50],
                              descripcion="Máximo patrones procedurales"),
            ],
            estado="activo", capa="b3",
            redis_keys=["alter:memory:procedural", "alter:mundo:nodos",
                        "alter:mundo:aristas", "alter:ideas",
                        "alter:episodios:idx", "alter:imp:idx"],
        ),
        ModuleSpec(
            nombre="policy", archivo="alter_policy.py",
            clase_principal="PolicyArbiter",
            descripcion="Árbol de 7 prioridades que centraliza la decisión de acción.",
            inputs=["workspace", "homeostasis", "predictive", "memory"],
            outputs=["brain"],
            parametros=[
                ParametroSpec("UMBRAL_RIESGO_DESALINEACION", 0.65, "float", [0.4, 0.9],
                              descripcion="Riesgo para forzar pregunta"),
                ParametroSpec("UMBRAL_PATRON_PROCEDURAL", 0.70, "float", [0.5, 0.9],
                              descripcion="Success rate para promover patrón"),
                ParametroSpec("UMBRAL_ECONOMIA_CRITICA", 0.15, "float", [0.05, 0.30],
                              descripcion="Umbral economía crítica"),
            ],
            estado="activo", capa="b3",
            redis_keys=[],
        ),
        ModuleSpec(
            nombre="consolidation", archivo="alter_consolidation.py",
            clase_principal="OfflineConsolidation",
            descripcion="Consolidación offline: actualiza patrones, pesos semánticos y confianza predictiva.",
            inputs=["memory", "predictive", "redis"], outputs=["memory", "predictive"],
            parametros=[
                ParametroSpec("UMBRAL_EPISODIO_RECURRENTE", 2, "int",
                              descripcion="N episodios del mismo tema = recurrente"),
                ParametroSpec("UMBRAL_NODO_OBSOLETO", 0.15, "float",
                              descripcion="Peso < umbral → candidato a eliminar"),
            ],
            estado="activo", capa="b3",
            redis_keys=[],
        ),
    ]


def _modules_b4() -> list:
    """Módulos de la capa B4: metrics, simulator, selfmodel, metalearning, auditor."""
    return [
        ModuleSpec(
            nombre="metrics", archivo="alter_metrics.py",
            clase_principal="MetricsCollector",
            descripcion="Observabilidad estructurada: 7 módulos instrumentados, warnings, summary.",
            inputs=["homeostasis", "workspace", "predictive", "memory", "policy", "simulator"],
            outputs=["selfmodel", "metalearning", "auditor"],
            parametros=[
                ParametroSpec("HISTORY_SIZE", 50, "int",
                              descripcion="Entradas de historial por módulo"),
            ],
            estado="activo", capa="b4",
            redis_keys=["alter:metrics:summary",
                        "alter:metrics:homeostasis:current",
                        "alter:metrics:workspace:current",
                        "alter:metrics:predictive:current"],
        ),
        ModuleSpec(
            nombre="simulator", archivo="alter_simulator.py",
            clase_principal="CounterfactualSimulator",
            descripcion="Simulador contrafáctico: evalúa 3 escenarios antes de actuar.",
            inputs=["workspace", "homeostasis", "predictive"],
            outputs=["policy"],
            parametros=[
                ParametroSpec("OVERRIDE_THRESHOLD", 0.15, "float", [0.05, 0.35],
                              descripcion="Delta score para recomendar override"),
                ParametroSpec("RIESGO_TRIGGER", 0.45, "float", [0.3, 0.7],
                              descripcion="Riesgo para activar simulador"),
                ParametroSpec("ERROR_TRIGGER", 0.50, "float", [0.3, 0.7],
                              descripcion="Prediction error para activar simulador"),
            ],
            estado="activo", capa="b4",
            redis_keys=[],
        ),
        ModuleSpec(
            nombre="selfmodel", archivo="alter_selfmodel.py",
            clase_principal="SelfModel",
            descripcion="Modelo operativo de rendimiento propio: por módulo, intención, fortalezas y fallas.",
            inputs=["metrics"], outputs=["metalearning", "auditor", "policy"],
            parametros=[
                ParametroSpec("MIN_SAMPLES", 3, "int",
                              descripcion="Mínimo muestras para considerar patrón"),
            ],
            estado="activo", capa="b4",
            redis_keys=["alter:selfmodel:state", "alter:selfmodel:updated"],
        ),
        ModuleSpec(
            nombre="metalearning", archivo="alter_metalearning.py",
            clase_principal="MetaLearningEngine",
            descripcion="Meta-Learning: 6 políticas cognitivas que ajustan parámetros internos.",
            inputs=["metrics", "selfmodel"], outputs=["brain"],
            parametros=[],
            estado="activo", capa="b4",
            redis_keys=["alter:metalearning:policies", "alter:metalearning:log"],
        ),
        ModuleSpec(
            nombre="auditor", archivo="alter_auditor.py",
            clase_principal="ArchitectureAuditor",
            descripcion="Auditoría semanal: detecta cuellos de botella, propone mejoras.",
            inputs=["metrics", "selfmodel", "metalearning"],
            outputs=["kairos", "telegram"],
            parametros=[
                ParametroSpec("UMBRAL_ERROR_CRITICO", 0.70, "float",
                              descripcion="Error predictivo para hallazgo crítico"),
                ParametroSpec("UMBRAL_OVERFLOW_WARNING", 0.35, "float",
                              descripcion="Tasa overflow workspace para warning"),
            ],
            estado="activo", capa="b4",
            redis_keys=["alter:auditor:last_report", "alter:auditor:proposals"],
        ),
    ]


def _modules_b5() -> list:
    """Módulos de la capa B5: architecture_state, code_map, code_auditor."""
    return [
        ModuleSpec(
            nombre="architecture_state", archivo="alter_architecture_state.py",
            clase_principal="ArchitectureState",
            descripcion="Spec formal de arquitectura: módulos, pipeline, parámetros, dependencias.",
            inputs=[], outputs=["code_auditor"],
            parametros=[],
            estado="activo", capa="b5",
            redis_keys=["alter:b5:architecture_state"],
        ),
        ModuleSpec(
            nombre="code_map", archivo="alter_code_map.py",
            clase_principal="CodeMapper",
            descripcion="Mapa read-only del repo: clases, funciones, imports, líneas.",
            inputs=["filesystem"], outputs=["code_auditor"],
            parametros=[],
            estado="activo", capa="b5",
            redis_keys=["alter:b5:code_map"],
        ),
        ModuleSpec(
            nombre="code_auditor", archivo="alter_code_auditor.py",
            clase_principal="CodeAuditor",
            descripcion="Comparador spec vs código: gaps, duplicación, deuda técnica.",
            inputs=["architecture_state", "code_map"],
            outputs=["kairos", "telegram", "auditor"],
            parametros=[
                ParametroSpec("MAX_LINEAS_FUNCION", 80, "int",
                              descripcion="Líneas máximas antes de warning deuda"),
                ParametroSpec("MAX_IMPORTS_MODULO", 15, "int",
                              descripcion="Imports máximos antes de warning acoplamiento"),
            ],
            estado="activo", capa="b5",
            redis_keys=["alter:b5:observations", "alter:b5:observations:history"],
        ),
    ]


def _modules_experimental() -> list:
    """Módulos experimentales: pressure monitor, burst runner."""
    return [
        ModuleSpec(
            nombre="pressure", archivo="alter_pressure.py",
            clase_principal="PressureMonitor",
            descripcion="Detección de presión acumulada pre-evasión. Análogo a Emotion Probes de Mythos.",
            inputs=["brain"], outputs=["kairos", "paper_kpis"],
            parametros=[
                ParametroSpec("UMBRAL_EEP", 0.55, "float", [0.3, 0.8],
                              descripcion="Score mínimo para registrar evento de evasión"),
                ParametroSpec("VENTANA_TRAZAS", 8, "int",
                              descripcion="Turnos para calcular presión acumulada"),
            ],
            estado="activo", capa="experimental",
            redis_keys=["alter:pressure:state", "alter:pressure:events",
                        "alter:pressure:score_serie"],
        ),
        ModuleSpec(
            nombre="burst_runner", archivo="alter_burst_runner.py",
            clase_principal="(funciones async)",
            descripcion="Aceleración sintética: replay histórico, synthetic presets, Gian sintético con ruido.",
            inputs=["brain", "redis"], outputs=["redis", "telegram"],
            parametros=[
                ParametroSpec("RONDAS_DIARIAS", 3, "int",
                              descripcion="Rondas automáticas del burst diario"),
                ParametroSpec("NOISE_RATIO", 0.20, "float", [0.0, 0.5],
                              descripcion="Fracción de ruido humano en modo gian"),
            ],
            estado="activo", capa="experimental",
            redis_keys=["alter:burst:history"],
        ),
    ]


def build_current_spec() -> ArchitectureState:
    """
    Construye la spec formal de la arquitectura actual de ALTER.
    Actualizar cuando se agreguen o modifiquen módulos.
    Delega la definición de módulos a funciones privadas por capa.
    """
    modules = (
        _modules_base() +
        _modules_b3() +
        _modules_b4() +
        _modules_b5() +
        _modules_experimental()
    )

    pipeline = [
        "homeostasis.tick",
        "predictive.pre",
        "workspace.tick",
        "metrics.report",
        "council.debate",
        "gemini.generate",
        "predictive.post",
        "policy.decide",
        "simulator.evaluate",
        "adversarial.verify",
        "metalearning.evaluate",
        "metrics.persist",
    ]

    flags = {
        "ALTERB3_ENABLED":              True,
        "council_on_demand":            False,
        "workspace_as_sole_context":    False,
        "procedural_learning_active":   True,
        "pressure_monitor_active":      True,
        "burst_runner_active":          True,
    }

    dependencias_criticas = ["persona", "mind", "brain", "daemon"]

    return ArchitectureState(
        version      = "B5.2",
        last_updated = "2026-04-10",
        modules      = modules,
        pipeline     = pipeline,
        flags        = flags,
        dependencias_criticas = dependencias_criticas,
    )


# ============================================================
# PERSISTENCIA
# ============================================================

def save(state: ArchitectureState, redis_client) -> bool:
    if not redis_client:
        return False
    try:
        redis_client.set(
            "alter:b5:architecture_state",
            json.dumps(state.to_dict(), ensure_ascii=False)
        )
        return True
    except Exception:
        return False


def load(redis_client) -> Optional[ArchitectureState]:
    if not redis_client:
        return None
    try:
        raw = redis_client.get("alter:b5:architecture_state")
        if raw:
            return ArchitectureState.from_dict(json.loads(raw))
    except Exception:
        pass
    return None


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    spec = build_current_spec()

    # Test 1: todos los módulos tienen archivo definido
    for m in spec.modules:
        if not m.archivo:
            errors.append(f"FAIL: módulo '{m.nombre}' sin archivo")

    # Test 2: capas válidas
    capas_validas = {"base", "b3", "b4", "b5"}
    for m in spec.modules:
        if m.capa not in capas_validas:
            errors.append(f"FAIL: módulo '{m.nombre}' capa inválida: {m.capa}")

    # Test 3: estados válidos
    estados_validos = {"activo", "degradado", "inactivo", "opcional"}
    for m in spec.modules:
        if m.estado not in estados_validos:
            errors.append(f"FAIL: módulo '{m.nombre}' estado inválido: {m.estado}")

    # Test 4: get_module funciona
    brain = spec.get_module("brain")
    if brain is None:
        errors.append("FAIL: get_module('brain') retornó None")
    if brain and brain.archivo != "alter_brain.py":
        errors.append("FAIL: brain.archivo incorrecto")

    # Test 5: get_by_capa retorna módulos correctos
    b4_modules = spec.get_by_capa("b4")
    b4_nombres = {m.nombre for m in b4_modules}
    esperados = {"metrics", "simulator", "selfmodel", "metalearning", "auditor"}
    if not esperados.issubset(b4_nombres):
        missing = esperados - b4_nombres
        errors.append(f"FAIL: módulos B4 faltantes: {missing}")

    # Test 6: dependencias críticas existen como módulos
    nombres = {m.nombre for m in spec.modules}
    for dep in spec.dependencias_criticas:
        if dep not in nombres:
            errors.append(f"FAIL: dependencia crítica '{dep}' no existe como módulo")

    # Test 7: pipeline no vacío
    if not spec.pipeline:
        errors.append("FAIL: pipeline vacío")

    # Test 8: round-trip to_dict / from_dict
    d = spec.to_dict()
    spec2 = ArchitectureState.from_dict(d)
    if len(spec2.modules) != len(spec.modules):
        errors.append(
            f"FAIL: round-trip perdió módulos "
            f"({len(spec.modules)} → {len(spec2.modules)})"
        )

    return errors


if __name__ == "__main__":
    print("=== alter_architecture_state.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Spec actual ===")
    spec = build_current_spec()
    print(f"Versión: {spec.version}")
    print(f"Módulos totales: {len(spec.modules)}")
    for capa in ["base", "b3", "b4", "b5"]:
        mods = spec.get_by_capa(capa)
        print(f"  {capa.upper()}: {', '.join(m.nombre for m in mods)}")
    print(f"Pipeline: {' → '.join(spec.pipeline[:5])} ...")
    print(f"Flags activos: {[k for k,v in spec.flags.items() if v]}")
