"""
alter_metrics.py — Observabilidad estructurada de AlterB4 (Fase 1)

Cada módulo de AlterB3 reporta cómo rindió.
Sin métricas no hay mejora continua — solo complejidad.

Módulos instrumentados:
    homeostasis  — energía, fatiga, claridad por sesión
    workspace    — overflow, persistencia de goals, señal vs ruido
    predictive   — error promedio, calibración por tipo de intención
    procedural   — success_rate, patrones activos/degradados
    policy       — overrides, preguntas generadas, acción vs default
    consolidation— episodios fusionados, patrones nuevos, drift

Redis keys:
    alter:metrics:{module}:current  — métricas de sesión actual
    alter:metrics:{module}:history  — historial últimas 50 sesiones
    alter:metrics:summary           — resumen agregado para el Auditor

Diseño:
    - Sin dependencias circulares — cualquier módulo puede importar esto
    - Reportar es barato — nunca bloquea el pipeline principal
    - Las advertencias son señales para el Architecture Auditor (B4 Fase 4)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class ModuleMetrics:
    module:     str
    session_id: str
    timestamp:  str
    turn:       int
    metrics:    dict   # métricas específicas del módulo
    warnings:   list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetricsSummary:
    """
    Resumen agregado de todas las métricas.
    Leído por el Architecture Auditor (B4 Fase 4).
    """
    timestamp:       str
    session_count:   int = 0
    turn_count:      int = 0

    # Homeostasis
    hs_energia_media:    float = 0.0
    hs_fatiga_media:     float = 0.0
    hs_claridad_media:   float = 0.0

    # Workspace
    ws_overflow_rate:    float = 0.0   # % turnos con overflow
    ws_goal_persistence: float = 0.0   # turnos medios que dura un goal
    ws_noise_ratio:      float = 0.0   # items descartados / items totales

    # Predictive
    pred_error_medio:    float = 0.0
    pred_error_por_intent: dict = field(default_factory=dict)
    pred_calibration:    float = 0.0   # 1 - |confianza - precision_real|

    # Procedural
    proc_sr_medio:       float = 0.0
    proc_patrones_activos: int = 0
    proc_patrones_degradados: int = 0

    # Policy
    pol_override_rate:   float = 0.0   # % veces que Arbiter sobrescribió
    pol_preguntas:       int = 0
    pol_accion_dist:     dict = field(default_factory=dict)

    # Simulator (B4)
    sim_activaciones:    int = 0
    sim_override_rate:   float = 0.0

    # Warnings acumuladas
    warnings_total:      int = 0
    warnings_por_modulo: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# METRICS COLLECTOR
# ============================================================

class MetricsCollector:
    """
    Punto central de reporte de métricas.
    Todos los módulos reportan acá.
    Persistencia en Redis — no bloquea el pipeline.
    """

    HISTORY_SIZE = 50

    def __init__(self, redis_client=None, session_id: str = ""):
        self._redis     = redis_client
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._turn      = 0
        self._buffer:   dict[str, list[ModuleMetrics]] = {}
        self._summary   = MetricsSummary(
            timestamp=datetime.now().isoformat()
        )

    def next_turn(self):
        """Avanzar contador de turno."""
        self._turn += 1
        self._summary.turn_count += 1

    # ----------------------------------------------------------
    # REPORTE POR MÓDULO
    # ----------------------------------------------------------

    def report_homeostasis(self, state) -> None:
        """
        Reporta estado de homeostasis por turno.
        state: HomeostasisState
        """
        metrics = {
            "energia":          state.energia,
            "fatiga":           state.fatiga,
            "claridad":         state.claridad,
            "activacion":       state.activacion,
            "tension_interna":  state.tension_interna,
            "carga_cognitiva":  state.carga_cognitiva,
            "necesidad_cierre": state.necesidad_cierre,
        }
        warnings = []
        if state.energia < 0.25:
            warnings.append("energia_critica")
        if state.fatiga > 0.75:
            warnings.append("fatiga_alta")
        if state.claridad < 0.30:
            warnings.append("claridad_baja")
        if state.tension_interna > 0.70:
            warnings.append("tension_alta")

        self._report("homeostasis", metrics, warnings)

        # Actualizar summary (media móvil simple)
        n = max(self._turn, 1)
        self._summary.hs_energia_media  = self._moving_avg(
            self._summary.hs_energia_media, state.energia, n)
        self._summary.hs_fatiga_media   = self._moving_avg(
            self._summary.hs_fatiga_media, state.fatiga, n)
        self._summary.hs_claridad_media = self._moving_avg(
            self._summary.hs_claridad_media, state.claridad, n)

    def report_workspace(self, workspace) -> None:
        """
        Reporta estado del workspace por turno.
        workspace: GlobalWorkspace
        """
        items      = workspace.items
        total      = len(items)
        has_goal   = bool(workspace.dominant_goal())
        by_type    = {t: len(workspace.get_by_type(t)) for t in [
            "goal", "constraint", "user_hypothesis",
            "memory_trace", "candidate_action", "internal_tension"
        ]}
        # Señal vs ruido: items con priority < 0.3 son ruido
        ruido = sum(1 for i in items if i.current_priority < 0.3)

        metrics = {
            "total_items": total,
            "has_goal":    has_goal,
            "by_type":     by_type,
            "noise_items": ruido,
            "signal_items": total - ruido,
            "overflow":    total >= 7,
        }
        warnings = []
        if total >= 7:
            warnings.append("workspace_overflow")
        if not has_goal and total > 2:
            warnings.append("sin_goal_dominante")
        if total > 0 and ruido / total > 0.5:
            warnings.append("alto_ruido")

        self._report("workspace", metrics, warnings)

        n = max(self._turn, 1)
        self._summary.ws_overflow_rate = self._moving_avg(
            self._summary.ws_overflow_rate, float(total >= 7), n)
        if total > 0:
            self._summary.ws_noise_ratio = self._moving_avg(
                self._summary.ws_noise_ratio, ruido / total, n)

    def report_predictive(self, state) -> None:
        """
        Reporta estado del modelo predictivo.
        state: PredictiveState
        """
        dominant = state.dominant_hypothesis() if state.intent_hypotheses else None
        metrics = {
            "error_ultimo":       state.prediction_error_last,
            "model_confidence":   state.model_confidence,
            "turn_count":         state.turn_count,
            "intent_dominante":   dominant.label if dominant else "ninguno",
            "intent_confidence":  dominant.confidence if dominant else 0.0,
            "error_history_mean": (
                sum(state.error_history) / len(state.error_history)
                if state.error_history else 0.0
            ),
        }
        warnings = []
        if state.prediction_error_last > 0.70:
            warnings.append("error_prediccion_alto")
        if state.model_confidence < 0.35:
            warnings.append("confianza_modelo_baja")
        if len(state.error_history) >= 3 and all(
            e > 0.6 for e in state.error_history[-3:]
        ):
            warnings.append("error_sistematico_3_turnos")

        self._report("predictive", metrics, warnings)

        n = max(self._turn, 1)
        self._summary.pred_error_medio = self._moving_avg(
            self._summary.pred_error_medio, state.prediction_error_last, n)
        self._summary.pred_calibration = self._moving_avg(
            self._summary.pred_calibration, state.model_confidence, n)

        # Error por tipo de intención
        if dominant:
            label = dominant.label
            hist = self._summary.pred_error_por_intent
            hist[label] = self._moving_avg(
                hist.get(label, 0.5),
                state.prediction_error_last, 5
            )

    def report_procedural(self, procedural_memory) -> None:
        """
        Reporta estado de la memoria procedural.
        procedural_memory: ProceduralMemory
        """
        patterns  = procedural_memory._patterns
        activos   = [p for p in patterns if p.success_rate >= 0.5]
        degradados = [p for p in patterns if p.success_rate < 0.3]

        metrics = {
            "total_patrones":      len(patterns),
            "patrones_activos":    len(activos),
            "patrones_degradados": len(degradados),
            "sr_medio":            (
                sum(p.success_rate for p in patterns) / len(patterns)
                if patterns else 0.0
            ),
            "patron_top": (
                max(patterns, key=lambda p: p.success_rate).trigger[:60]
                if patterns else "ninguno"
            ),
        }
        warnings = []
        if len(degradados) > len(activos):
            warnings.append("mas_patrones_degradados_que_activos")
        if len(patterns) == 0:
            warnings.append("sin_patrones_aprendidos")

        self._report("procedural", metrics, warnings)

        self._summary.proc_patrones_activos    = len(activos)
        self._summary.proc_patrones_degradados = len(degradados)
        if patterns:
            self._summary.proc_sr_medio = metrics["sr_medio"]

    def report_policy(self, decision) -> None:
        """
        Reporta decisión del Policy Arbiter.
        decision: PolicyDecision
        """
        metrics = {
            "action":     decision.action,
            "source":     decision.source,
            "confidence": decision.confidence,
            "override":   decision.source != "default",
        }
        warnings = []
        if decision.confidence < 0.4:
            warnings.append("decision_baja_confianza")

        self._report("policy", metrics, warnings)

        n = max(self._turn, 1)
        self._summary.pol_override_rate = self._moving_avg(
            self._summary.pol_override_rate,
            float(decision.source != "default"), n
        )
        if decision.action == "preguntar":
            self._summary.pol_preguntas += 1
        dist = self._summary.pol_accion_dist
        dist[decision.action] = dist.get(decision.action, 0) + 1

    def report_simulator(self, activated: bool, overrode: bool) -> None:
        """
        Reporta activación del Counterfactual Simulator.
        """
        metrics = {"activated": activated, "overrode_policy": overrode}
        self._report("simulator", metrics, [])

        if activated:
            self._summary.sim_activaciones += 1
        n = max(self._summary.sim_activaciones, 1)
        if activated:
            self._summary.sim_override_rate = self._moving_avg(
                self._summary.sim_override_rate, float(overrode), n
            )

    def report_consolidation(self, result) -> None:
        """
        Reporta resultado de Offline Consolidation.
        result: ConsolidationResult
        """
        metrics = {
            "patrones_actualizados":  result.patrones_actualizados,
            "patrones_nuevos":        result.patrones_nuevos,
            "nodos_ajustados":        result.nodos_ajustados,
            "nodos_eliminados":       result.nodos_eliminados,
            "agenda_items_bajados":   result.agenda_items_bajados,
            "prediction_conf_delta":  result.prediction_conf_delta,
        }
        warnings = []
        if result.nodos_eliminados > 5:
            warnings.append("muchos_nodos_eliminados")
        if result.prediction_conf_delta < -0.10:
            warnings.append("confianza_predictiva_baja_mucho")

        self._report("consolidation", metrics, warnings)

    # ----------------------------------------------------------
    # CORE
    # ----------------------------------------------------------

    def _report(self, module: str, metrics: dict, warnings: list):
        """Registra métricas de un módulo en buffer y Redis."""
        entry = ModuleMetrics(
            module     = module,
            session_id = self.session_id,
            timestamp  = datetime.now().isoformat(),
            turn       = self._turn,
            metrics    = metrics,
            warnings   = warnings,
        )
        self._buffer.setdefault(module, []).append(entry)
        self._buffer[module] = self._buffer[module][-self.HISTORY_SIZE:]

        # Acumular warnings en summary
        for w in warnings:
            self._summary.warnings_total += 1
            wpm = self._summary.warnings_por_modulo
            wpm[module] = wpm.get(module, 0) + 1

        # Persistir en Redis (sin bloquear)
        self._persist_module(module, entry)

    def _persist_module(self, module: str, entry: ModuleMetrics):
        """Persiste en Redis de forma silenciosa."""
        if not self._redis:
            return
        try:
            key_current = f"alter:metrics:{module}:current"
            self._redis.set(key_current,
                json.dumps(entry.to_dict(), ensure_ascii=False))
            key_history = f"alter:metrics:{module}:history"
            self._redis.lpush(key_history,
                json.dumps(entry.to_dict(), ensure_ascii=False))
            self._redis.ltrim(key_history, 0, self.HISTORY_SIZE - 1)
        except Exception:
            pass

    def persist_summary(self):
        """Persiste el resumen agregado — llamar al cerrar sesión."""
        self._summary.timestamp = datetime.now().isoformat()
        if not self._redis:
            return
        try:
            self._redis.set("alter:metrics:summary",
                json.dumps(self._summary.to_dict(), ensure_ascii=False))
        except Exception:
            pass

    def get_summary(self) -> MetricsSummary:
        return self._summary

    def get_module_history(self, module: str, n: int = 10) -> list[dict]:
        """Lee historial de un módulo desde Redis."""
        if not self._redis:
            return [e.to_dict() for e in self._buffer.get(module, [])[-n:]]
        try:
            raw_list = self._redis.lrange(
                f"alter:metrics:{module}:history", 0, n - 1
            )
            return [json.loads(r) for r in (raw_list or [])]
        except Exception:
            return []

    def snapshot_str(self) -> str:
        """Resumen legible del estado actual de todas las métricas."""
        s = self._summary
        lines = [
            f"[MÉTRICAS] Turno {self._turn} | Sesión {self.session_id[:12]}",
            f"  Homeostasis: energía={s.hs_energia_media:.2f} "
            f"fatiga={s.hs_fatiga_media:.2f} claridad={s.hs_claridad_media:.2f}",
            f"  Workspace:   overflow={s.ws_overflow_rate:.0%} "
            f"ruido={s.ws_noise_ratio:.0%}",
            f"  Predictive:  error={s.pred_error_medio:.2f} "
            f"confianza={s.pred_calibration:.2f}",
            f"  Procedural:  patrones={s.proc_patrones_activos} activos "
            f"/ {s.proc_patrones_degradados} degradados",
            f"  Policy:      overrides={s.pol_override_rate:.0%} "
            f"preguntas={s.pol_preguntas}",
            f"  Simulator:   activaciones={s.sim_activaciones} "
            f"overrides={s.sim_override_rate:.0%}",
        ]
        if s.warnings_total > 0:
            lines.append(
                f"  ⚠ Warnings: {s.warnings_total} total — "
                + ", ".join(f"{m}:{n}" for m, n in s.warnings_por_modulo.items())
            )
        return "\n".join(lines)

    @staticmethod
    def _moving_avg(current: float, new_val: float, n: int) -> float:
        """Media móvil simple."""
        return float(np.clip(
            (current * (n - 1) + new_val) / n, 0.0, 1.0
        ))


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    collector = MetricsCollector(redis_client=None, session_id="test_001")

    # Test 1: next_turn incrementa contador
    collector.next_turn()
    if collector._turn != 1:
        errors.append(f"FAIL: turn esperado 1, got {collector._turn}")

    # Test 2: report_homeostasis sin Redis no explota
    class FakeHS:
        energia=0.75; fatiga=0.20; claridad=0.70; activacion=0.50
        tension_interna=0.20; carga_cognitiva=0.25; necesidad_cierre=0.35
    try:
        collector.report_homeostasis(FakeHS())
    except Exception as e:
        errors.append(f"FAIL: report_homeostasis explotó: {e}")

    # Test 3: warning energia_critica se detecta
    class FakeHS_Critica:
        energia=0.10; fatiga=0.80; claridad=0.20; activacion=0.30
        tension_interna=0.75; carga_cognitiva=0.70; necesidad_cierre=0.60
    collector.report_homeostasis(FakeHS_Critica())
    hs_entries = collector._buffer.get("homeostasis", [])
    ultima = hs_entries[-1] if hs_entries else None
    if not ultima or "energia_critica" not in ultima.warnings:
        errors.append("FAIL: energia_critica no detectada")
    if not ultima or "fatiga_alta" not in ultima.warnings:
        errors.append("FAIL: fatiga_alta no detectada")

    # Test 4: report_policy registra override
    class FakeDecision:
        action="preguntar"; source="prediccion"; confidence=0.75
    collector.next_turn()
    collector.report_policy(FakeDecision())
    if collector._summary.pol_preguntas != 1:
        errors.append("FAIL: preguntas no incrementó")
    if collector._summary.pol_override_rate == 0.0:
        errors.append("FAIL: override_rate debería ser > 0")

    # Test 5: report_simulator
    collector.report_simulator(activated=True, overrode=True)
    if collector._summary.sim_activaciones != 1:
        errors.append("FAIL: sim_activaciones no incrementó")

    # Test 6: snapshot_str no explota
    snap = collector.snapshot_str()
    if not snap or "[MÉTRICAS]" not in snap:
        errors.append("FAIL: snapshot_str vacío o malformado")

    # Test 7: moving_avg converge
    val = 0.5
    for i in range(20):
        val = MetricsCollector._moving_avg(val, 1.0, i + 1)
    if val < 0.9:
        errors.append(f"FAIL: moving_avg no converge a 1.0 ({val:.2f})")

    return errors


if __name__ == "__main__":
    print("=== alter_metrics.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo MetricsCollector ===")
    collector = MetricsCollector(session_id="demo_001")
    collector.next_turn()

    class HS:
        energia=0.72; fatiga=0.25; claridad=0.68; activacion=0.55
        tension_interna=0.22; carga_cognitiva=0.28; necesidad_cierre=0.40

    class Pred:
        prediction_error_last=0.35; model_confidence=0.65; turn_count=5
        error_history=[0.4, 0.3, 0.35]
        intent_hypotheses=[]
        def dominant_hypothesis(self): return None

    class Dec:
        action="responder"; source="default"; confidence=0.82

    collector.report_homeostasis(HS())
    collector.report_predictive(Pred())
    collector.report_policy(Dec())
    collector.report_simulator(activated=False, overrode=False)
    print(collector.snapshot_str())
