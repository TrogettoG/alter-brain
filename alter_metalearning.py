"""
alter_metalearning.py — Meta-Learning Engine de AlterB4 (Fase 3)

Aprende políticas cognitivas, no solo patrones de respuesta.

Diferencia con ProceduralMemory:
    ProceduralMemory: "cuando X pasa, hacer Y en la respuesta"
    Meta-Learning:    "cuando el sistema está en estado Z,
                       cambiar cómo piensa (parámetros internos)"

Ejemplos de políticas:
    "si prediction_error > 0.6 por 3 turnos → bajar confianza base 0.1"
    "si workspace_overflow_rate > 40% → reducir candidatos por turno"
    "si claridad < 0.35 → activar modo conservación"
    "si sr_medio_procedural < 0.4 → pausar aprendizaje procedural"

Qué ajusta:
    Parámetros cognitivos de los módulos B3/B4.
    NO toca identidad, pizarra ni autobiografía.

Redis keys:
    alter:metalearning:policies  — políticas activas
    alter:metalearning:log       — historial de ajustes aplicados
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class CognitivePolicy:
    id:          str
    nombre:      str
    condicion:   dict   # qué métricas/estado la disparan
    ajuste:      dict   # qué parámetro cambia y cuánto
    activa:      bool = True
    evidencia:   int  = 0   # cuántas veces se confirmó útil
    activaciones: int = 0   # cuántas veces se activó
    last_activated: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def matches(self, state: dict) -> bool:
        """
        Evalúa si la condición de esta política se cumple.
        state: dict con valores actuales de métricas y módulos.
        """
        cond = self.condicion
        for key, spec in cond.items():
            val = state.get(key)
            if val is None:
                return False
            if isinstance(spec, dict):
                op  = spec.get("op", "gt")
                thr = spec.get("val", 0.5)
                n   = spec.get("n", 1)   # para condiciones de N turnos
                if op == "gt"  and not val > thr:
                    return False
                if op == "lt"  and not val < thr:
                    return False
                if op == "gte" and not val >= thr:
                    return False
                if op == "lte" and not val <= thr:
                    return False
                if op == "consecutive_gt":
                    # val debería ser una lista de últimos N valores
                    if not isinstance(val, list) or len(val) < n:
                        return False
                    if not all(v > thr for v in val[-n:]):
                        return False
            elif isinstance(spec, bool):
                if val != spec:
                    return False
        return True


@dataclass
class PolicyApplication:
    """Registro de una aplicación de política."""
    policy_id:   str
    policy_name: str
    timestamp:   str
    ajuste:      dict
    modulo:      str
    parametro:   str
    valor_antes: float
    valor_despues: float

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# POLÍTICAS PREDEFINIDAS
# ============================================================

DEFAULT_POLICIES = [
    CognitivePolicy(
        id     = "mp001",
        nombre = "Bajar confianza tras error sistémico",
        condicion = {
            "pred_error_history": {"op": "consecutive_gt", "val": 0.60, "n": 3}
        },
        ajuste = {
            "modulo":    "predictive",
            "parametro": "model_confidence",
            "delta":     -0.10,
            "min":        0.25,
        }
    ),
    CognitivePolicy(
        id     = "mp002",
        nombre = "Reducir candidatos workspace si overflow recurrente",
        condicion = {
            "ws_overflow_rate": {"op": "gt", "val": 0.40}
        },
        ajuste = {
            "modulo":    "workspace",
            "parametro": "max_candidates_per_turn",
            "delta":     -1,
            "min":        3,
        }
    ),
    CognitivePolicy(
        id     = "mp003",
        nombre = "Modo conservación si claridad colapsa",
        condicion = {
            "hs_claridad_media": {"op": "lt", "val": 0.35}
        },
        ajuste = {
            "modulo":    "homeostasis",
            "parametro": "modo_forzado",
            "valor":     "conservacion",
        }
    ),
    CognitivePolicy(
        id     = "mp004",
        nombre = "Pausar aprendizaje procedural si sr_medio bajo",
        condicion = {
            "proc_sr_medio": {"op": "lt", "val": 0.35}
        },
        ajuste = {
            "modulo":    "procedural",
            "parametro": "aprendizaje_activo",
            "valor":     False,
        }
    ),
    CognitivePolicy(
        id     = "mp005",
        nombre = "Aumentar threshold simulator si override_rate alto",
        condicion = {
            "sim_override_rate": {"op": "gt", "val": 0.50}
        },
        ajuste = {
            "modulo":    "simulator",
            "parametro": "override_threshold",
            "delta":     +0.05,
            "max":        0.35,
        }
    ),
    CognitivePolicy(
        id     = "mp006",
        nombre = "Activar Council más frecuente si policy override_rate bajo",
        condicion = {
            "pol_override_rate":  {"op": "lt", "val": 0.05},
            "pred_error_medio":   {"op": "gt", "val": 0.55},
        },
        ajuste = {
            "modulo":    "council",
            "parametro": "tension_minima_trigger",
            "valor":     "baja",
        }
    ),
]


# ============================================================
# META-LEARNING ENGINE
# ============================================================

class MetaLearningEngine:
    """
    Evalúa políticas cognitivas y aplica ajustes a parámetros internos.

    Ciclo:
        1. Leer estado actual (métricas + self-model)
        2. Evaluar qué políticas se disparan
        3. Aplicar ajustes
        4. Registrar aplicaciones
        5. Actualizar evidencia de las políticas

    No toca identidad ni pizarra.
    Reporta cada ajuste para trazabilidad.
    """

    def __init__(self, redis_client=None):
        self._redis   = redis_client
        self._policies: list[CognitivePolicy] = []
        self._load_policies()

    def _load_policies(self):
        """Carga políticas desde Redis o usa las predefinidas."""
        if self._redis:
            try:
                raw = self._redis.get("alter:metalearning:policies")
                if raw:
                    pols_raw = json.loads(raw)
                    self._policies = [CognitivePolicy(**p) for p in pols_raw]
                    return
            except Exception:
                pass
        # Usar políticas predefinidas
        self._policies = [CognitivePolicy(**p.to_dict()) for p in DEFAULT_POLICIES]

    def _save_policies(self):
        if not self._redis:
            return
        try:
            self._redis.set(
                "alter:metalearning:policies",
                json.dumps([p.to_dict() for p in self._policies],
                           ensure_ascii=False)
            )
        except Exception:
            pass

    def evaluate(
        self,
        metrics_summary,   # MetricsSummary de alter_metrics
        self_model,        # SelfModel de alter_selfmodel
    ) -> list[PolicyApplication]:
        """
        Evalúa todas las políticas contra el estado actual.
        Retorna lista de aplicaciones — qué se ajustó y por qué.
        """
        # Construir estado para matching
        state = self._build_state(metrics_summary, self_model)
        applications = []

        for policy in self._policies:
            if not policy.activa:
                continue
            if policy.matches(state):
                app = self._apply_policy(policy, state)
                if app:
                    applications.append(app)
                    policy.activaciones += 1
                    policy.last_activated = datetime.now().isoformat()

        self._save_policies()
        self._log_applications(applications)
        return applications

    def _build_state(self, metrics_summary, self_model) -> dict:
        """Construye el dict de estado para el matching de políticas."""
        s = metrics_summary
        state = {
            # Homeostasis
            "hs_energia_media":    s.hs_energia_media,
            "hs_fatiga_media":     s.hs_fatiga_media,
            "hs_claridad_media":   s.hs_claridad_media,
            # Workspace
            "ws_overflow_rate":    s.ws_overflow_rate,
            "ws_noise_ratio":      s.ws_noise_ratio,
            # Predictive
            "pred_error_medio":    s.pred_error_medio,
            "pred_calibration":    s.pred_calibration,
            # Procedural
            "proc_sr_medio":       s.proc_sr_medio,
            # Policy
            "pol_override_rate":   s.pol_override_rate,
            # Simulator
            "sim_override_rate":   s.sim_override_rate,
        }
        # Para condición consecutive_gt necesitamos lista de errores recientes
        if hasattr(self_model, "intent_performance"):
            state["pred_error_history"] = [
                1.0 - ip.success_rate
                for ip in self_model.intent_performance
            ]
        return state

    def _apply_policy(
        self,
        policy: CognitivePolicy,
        state: dict,
    ) -> Optional[PolicyApplication]:
        """
        Registra la aplicación de una política.
        Los ajustes reales los aplica alter_brain.py al leer las aplicaciones.
        """
        ajuste   = policy.ajuste
        modulo   = ajuste.get("modulo", "")
        parametro= ajuste.get("parametro", "")

        # Calcular valor_antes (desde state si disponible)
        mapping_state = {
            "model_confidence":           "pred_calibration",
            "ws_overflow_rate":           "ws_overflow_rate",
            "hs_claridad_media":          "hs_claridad_media",
            "proc_sr_medio":              "proc_sr_medio",
            "sim_override_rate":          "sim_override_rate",
            "pol_override_rate":          "pol_override_rate",
        }
        state_key  = mapping_state.get(parametro, "")
        valor_antes = float(state.get(state_key, 0.5))

        # Calcular valor_después
        if "delta" in ajuste:
            delta        = ajuste["delta"]
            valor_despues = valor_antes + delta
            if "min" in ajuste:
                valor_despues = max(ajuste["min"], valor_despues)
            if "max" in ajuste:
                valor_despues = min(ajuste["max"], valor_despues)
        elif "valor" in ajuste:
            valor_despues = ajuste["valor"] if isinstance(ajuste["valor"], float) else 0.0
        else:
            return None

        return PolicyApplication(
            policy_id    = policy.id,
            policy_name  = policy.nombre,
            timestamp    = datetime.now().isoformat(),
            ajuste       = ajuste,
            modulo       = modulo,
            parametro    = parametro,
            valor_antes  = valor_antes,
            valor_despues= float(valor_despues) if isinstance(valor_despues, float) else 0.0,
        )

    def _log_applications(self, applications: list[PolicyApplication]):
        """Persiste aplicaciones en Redis para trazabilidad."""
        if not self._redis or not applications:
            return
        try:
            for app in applications:
                self._redis.lpush(
                    "alter:metalearning:log",
                    json.dumps(app.to_dict(), ensure_ascii=False)
                )
            self._redis.ltrim("alter:metalearning:log", 0, 99)
        except Exception:
            pass

    def reinforce_policy(self, policy_id: str, useful: bool):
        """Sube o baja evidencia de una política."""
        for p in self._policies:
            if p.id == policy_id:
                p.evidencia += 1 if useful else -1
                if p.evidencia < -5:
                    p.activa = False  # demasiada evidencia negativa → desactivar
                break
        self._save_policies()

    def get_active_policies(self) -> list[CognitivePolicy]:
        return [p for p in self._policies if p.activa]

    def snapshot_str(self) -> str:
        activas = self.get_active_policies()
        lines = [f"[META-LEARNING] {len(activas)} políticas activas"]
        for p in activas:
            lines.append(
                f"  [{p.id}] {p.nombre[:50]} "
                f"(activaciones:{p.activaciones} evidencia:{p.evidencia})"
            )
        return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    # Test 1: CognitivePolicy.matches — condición gt
    from alter_selfmodel import SelfModel, IntentPerformance
    from alter_metrics import MetricsSummary

    p = CognitivePolicy(
        id="t1", nombre="test",
        condicion={"pred_error_medio": {"op": "gt", "val": 0.5}},
        ajuste={"modulo": "predictive", "parametro": "model_confidence",
                "delta": -0.1, "min": 0.2}
    )
    state_match    = {"pred_error_medio": 0.7}
    state_no_match = {"pred_error_medio": 0.3}

    if not p.matches(state_match):
        errors.append("FAIL: policy gt no matcheó cuando debería")
    if p.matches(state_no_match):
        errors.append("FAIL: policy gt matcheó cuando no debería")

    # Test 2: CognitivePolicy.matches — consecutive_gt
    p2 = CognitivePolicy(
        id="t2", nombre="test2",
        condicion={"pred_error_history": {"op": "consecutive_gt", "val": 0.6, "n": 3}},
        ajuste={"modulo": "predictive", "parametro": "model_confidence",
                "delta": -0.1, "min": 0.2}
    )
    state_consec = {"pred_error_history": [0.7, 0.8, 0.75, 0.9]}
    state_no_consec = {"pred_error_history": [0.7, 0.3, 0.75]}

    if not p2.matches(state_consec):
        errors.append("FAIL: consecutive_gt no matcheó")
    if p2.matches(state_no_consec):
        errors.append("FAIL: consecutive_gt matcheó cuando no debería")

    # Test 3: MetaLearningEngine sin Redis no explota
    engine = MetaLearningEngine(redis_client=None)
    if not engine._policies:
        errors.append("FAIL: no cargó políticas predefinidas")

    # Test 4: evaluate con estado vacío no explota
    ms = MetricsSummary(timestamp=datetime.now().isoformat())
    sm = SelfModel()
    try:
        applications = engine.evaluate(ms, sm)
        if not isinstance(applications, list):
            errors.append("FAIL: evaluate no retornó lista")
    except Exception as e:
        errors.append(f"FAIL: evaluate explotó: {e}")

    # Test 5: política se activa con estado correcto
    ms2 = MetricsSummary(
        timestamp=datetime.now().isoformat(),
        pred_error_medio=0.75,  # > 0.55
        pol_override_rate=0.02, # < 0.05
    )
    apps = engine.evaluate(ms2, sm)
    # mp006 debería activarse (pred_error alto + pol_override bajo)
    triggered = [a.policy_id for a in apps]
    if "mp006" not in triggered:
        errors.append(f"FAIL: mp006 no se activó con pred_error=0.75, pol_override=0.02 — triggered: {triggered}")

    # Test 6: reinforce_policy desactiva con evidencia muy negativa
    engine2 = MetaLearningEngine(redis_client=None)
    pol = engine2._policies[0]
    pol_id = pol.id
    for _ in range(6):
        engine2.reinforce_policy(pol_id, useful=False)
    pol_updated = next(p for p in engine2._policies if p.id == pol_id)
    if pol_updated.activa:
        errors.append("FAIL: política debería desactivarse con evidencia < -5")

    # Test 7: snapshot_str no explota
    snap = engine.snapshot_str()
    if "[META-LEARNING]" not in snap:
        errors.append("FAIL: snapshot_str malformado")

    return errors


if __name__ == "__main__":
    print("=== alter_metalearning.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo MetaLearningEngine ===")
    from alter_selfmodel import SelfModel, IntentPerformance
    from alter_metrics import MetricsSummary

    engine = MetaLearningEngine(redis_client=None)
    print(engine.snapshot_str())

    # Simular estado con varias condiciones activas
    ms = MetricsSummary(
        timestamp=datetime.now().isoformat(),
        ws_overflow_rate=0.55,    # activa mp002
        hs_claridad_media=0.28,   # activa mp003
        pred_error_medio=0.72,    # activa mp006 (junto con pol_override bajo)
        pol_override_rate=0.03,   # activa mp006
        proc_sr_medio=0.30,       # activa mp004
    )
    sm = SelfModel()
    sm.intent_performance = [
        IntentPerformance("quiere_accion", 0.4, 5, 0.6),
        IntentPerformance("quiere_analisis", 0.35, 5, 0.65),
        IntentPerformance("quiere_cierre", 0.38, 5, 0.62),
    ]
    apps = engine.evaluate(ms, sm)
    print(f"\nPolíticas activadas: {len(apps)}")
    for app in apps:
        print(f"  [{app.policy_id}] {app.policy_name[:50]}")
        print(f"    {app.modulo}.{app.parametro}: "
              f"{app.valor_antes:.2f} → {app.valor_despues:.2f}")
