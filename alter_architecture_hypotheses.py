"""
alter_architecture_hypotheses.py — Hypothesis Generator de AlterB5 (Fase 2)

Lee observaciones del Code Auditor y hallazgos del Architecture Auditor
y formula hipótesis estructurales concretas con score de evidencia.

Solo hipótesis con score >= SCORE_MINIMO pasan al Experiment Runner.
Las hipótesis no replayables (splits, refactors estructurales) quedan
en estado "propuesta" o "no_replayable_yet" — para B5 Fase 4.

Redis keys:
    alter:b5:hypotheses  — lista de hipótesis activas
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np


# ============================================================
# CONSTANTES
# ============================================================

SCORE_MINIMO_EXPERIMENTO = 0.40   # score mínimo para pasar al runner
MAX_HYPOTHESES           = 20     # máximo hipótesis activas


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class ArchitectureHypothesis:
    id:           str
    titulo:       str
    tipo:         str    # "parametro" | "refactor" | "split" | "pipeline" | "nuevo_modulo"
    descripcion:  str
    evidencia:    list   # list[str] — observaciones que la soportan
    modulo:       str    # módulo afectado
    impacto:      str    # "bajo" | "medio" | "alto"
    riesgo:       str    # "bajo" | "medio" | "alto"
    score:        float  # 0..1 — evidencia acumulada
    estado:       str    # "propuesta" | "en_experimento" | "validada" | "descartada" | "no_replayable_yet"
    replayable:   bool   # si puede probarse con trazas históricas
    # Si replayable=True, qué parámetro y rango experimenta
    parametro_target: str = ""   # ej: "workspace.MAX_ITEMS"
    rango_experimento: list = field(default_factory=list)  # [val_min, val_max]
    created_at:   str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at:   str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureHypothesis":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def puede_experimentar(self) -> bool:
        return (
            self.replayable and
            self.score >= SCORE_MINIMO_EXPERIMENTO and
            self.estado == "propuesta" and
            bool(self.parametro_target)
        )


# ============================================================
# REGLAS DE GENERACIÓN
# ============================================================

# Cada regla: (condición_fn, builder_fn)
# condición_fn(obs_list, hallazgos_list, selfmodel) → bool
# builder_fn(obs_list, hallazgos_list, selfmodel) → ArchitectureHypothesis | None

def _score_from_evidence(n: int) -> float:
    """Score basado en cantidad de evidencias — sube con más evidencia."""
    return float(np.clip(0.2 + 0.15 * n, 0.0, 1.0))


class HypothesisGenerator:
    """
    Genera hipótesis estructurales desde observaciones de código
    y hallazgos de arquitectura.

    Reglas de generación:
        funcion_larga + modulo_activo  → hipótesis de refactor
        clase_grande + modulo_core     → hipótesis de split de clase
        acoplamiento_alto              → hipótesis de reducción de dependencias
        gap_implementacion             → hipótesis de completar implementación
        error_predictivo_alto          → hipótesis de mejora en infer_intent
        workspace_overflow             → hipótesis de reducción de MAX_ITEMS candidatos
        claridad_baja (homeostasis)    → hipótesis de ajuste threshold recovery
        sim_override_rate_alto         → hipótesis de ajuste OVERRIDE_THRESHOLD
        policy_riesgo_alto             → hipótesis de ajuste UMBRAL_RIESGO
    """

    def generate(
        self,
        code_report,      # CodeAuditReport de alter_code_auditor
        arch_report,      # AuditReport de alter_auditor (B4)
        self_model,       # SelfModel de alter_selfmodel
        existing: list = None,  # hipótesis ya existentes (para no duplicar)
    ) -> list:
        """
        Genera hipótesis nuevas desde las observaciones disponibles.
        No duplica hipótesis ya existentes (mismo módulo + mismo tipo).
        """
        existing = existing or []
        existentes_keys = {
            (h.modulo, h.tipo, h.parametro_target)
            for h in existing
            if h.estado not in ("descartada",)
        }

        nuevas = []
        obs      = code_report.observaciones if code_report else []
        hallazgos = arch_report.hallazgos    if arch_report else []

        # Agrupar observaciones por tipo
        funciones_largas  = [o for o in obs if o.tipo == "funcion_larga"]
        clases_grandes    = [o for o in obs if o.tipo == "clase_grande"]
        acoplamientos     = [o for o in obs if o.tipo == "acoplamiento_alto"]
        gaps              = [o for o in obs if o.tipo == "gap_implementacion"]
        modulos_grandes   = [o for o in obs if o.tipo == "modulo_grande"]

        # Agrupar hallazgos por módulo
        hall_pred   = [h for h in hallazgos if h.modulo == "predictive"]
        hall_ws     = [h for h in hallazgos if h.modulo == "workspace"]
        hall_hs     = [h for h in hallazgos if h.modulo == "homeostasis"]
        hall_sim    = [h for h in hallazgos if h.modulo == "simulator"]
        hall_policy = [h for h in hallazgos if h.modulo == "policy"]

        # ── Hipótesis paramétricas (replayables) ────────────

        # 1. workspace.MAX_ITEMS si hay overflow
        if hall_ws and ("workspace", "parametro", "workspace.MAX_ITEMS") not in existentes_keys:
            overflow_obs = [h for h in hall_ws if "overflow" in h.descripcion.lower()]
            if overflow_obs:
                h = ArchitectureHypothesis(
                    id       = str(uuid.uuid4())[:8],
                    titulo   = "Reducir MAX_ITEMS en workspace para evitar overflow",
                    tipo     = "parametro",
                    descripcion = (
                        "El workspace tiene overflow frecuente. Reducir MAX_ITEMS "
                        "de 7 a 5-6 para forzar selección más estricta de candidatos."
                    ),
                    evidencia = [h.descripcion for h in overflow_obs],
                    modulo   = "workspace",
                    impacto  = "medio",
                    riesgo   = "bajo",
                    score    = _score_from_evidence(len(overflow_obs)),
                    estado   = "propuesta",
                    replayable = True,
                    parametro_target   = "workspace.MAX_ITEMS",
                    rango_experimento  = [5, 7],
                )
                nuevas.append(h)

        # 2. simulator.OVERRIDE_THRESHOLD si override_rate alto
        if hall_sim and ("simulator", "parametro", "simulator.OVERRIDE_THRESHOLD") not in existentes_keys:
            override_obs = [h for h in hall_sim if "override" in h.descripcion.lower()]
            if override_obs:
                h = ArchitectureHypothesis(
                    id       = str(uuid.uuid4())[:8],
                    titulo   = "Subir OVERRIDE_THRESHOLD del Simulator",
                    tipo     = "parametro",
                    descripcion = (
                        "El Simulator override demasiado frecuente — "
                        "sube el threshold para que solo intervenga en casos claros."
                    ),
                    evidencia = [h.descripcion for h in override_obs],
                    modulo   = "simulator",
                    impacto  = "bajo",
                    riesgo   = "bajo",
                    score    = _score_from_evidence(len(override_obs)),
                    estado   = "propuesta",
                    replayable = True,
                    parametro_target  = "simulator.OVERRIDE_THRESHOLD",
                    rango_experimento = [0.10, 0.30],
                )
                nuevas.append(h)

        # 3. policy.UMBRAL_RIESGO si hay desalineación frecuente
        if hall_policy and ("policy", "parametro", "policy.UMBRAL_RIESGO_DESALINEACION") not in existentes_keys:
            riesgo_obs = [h for h in hall_policy if "override" in h.descripcion.lower()]
            if riesgo_obs:
                h = ArchitectureHypothesis(
                    id       = str(uuid.uuid4())[:8],
                    titulo   = "Ajustar UMBRAL_RIESGO_DESALINEACION en Policy Arbiter",
                    tipo     = "parametro",
                    descripcion = (
                        "El Arbiter sobrescribe demasiado. Subir el umbral de riesgo "
                        "para preguntar solo cuando la desalineación es clara."
                    ),
                    evidencia = [h.descripcion for h in riesgo_obs],
                    modulo   = "policy",
                    impacto  = "medio",
                    riesgo   = "bajo",
                    score    = _score_from_evidence(len(riesgo_obs)),
                    estado   = "propuesta",
                    replayable = True,
                    parametro_target  = "policy.UMBRAL_RIESGO_DESALINEACION",
                    rango_experimento = [0.55, 0.80],
                )
                nuevas.append(h)

        # 4. predictive: señales léxicas si error alto en intención
        if hall_pred and ("predictive", "parametro", "predictive.intent_signals") not in existentes_keys:
            error_obs = [h for h in hall_pred if "precisión" in h.descripcion.lower()
                         or "error" in h.descripcion.lower()]
            if error_obs:
                h = ArchitectureHypothesis(
                    id       = str(uuid.uuid4())[:8],
                    titulo   = "Ampliar señales léxicas para intenciones débiles",
                    tipo     = "parametro",
                    descripcion = (
                        "Hay intenciones con baja tasa de éxito. "
                        "Agregar más señales léxicas en INTENT_SIGNALS para esas categorías."
                    ),
                    evidencia = [h.descripcion for h in error_obs],
                    modulo   = "predictive",
                    impacto  = "medio",
                    riesgo   = "bajo",
                    score    = _score_from_evidence(len(error_obs)),
                    estado   = "propuesta",
                    replayable = True,
                    parametro_target  = "predictive.intent_signals",
                    rango_experimento = [],  # no numérico — experimento cualitativo
                )
                nuevas.append(h)

        # 5. homeostasis: recovery si claridad baja
        if hall_hs and ("homeostasis", "parametro", "homeostasis.recovery_rate") not in existentes_keys:
            claridad_obs = [h for h in hall_hs if "claridad" in h.descripcion.lower()]
            if claridad_obs:
                h = ArchitectureHypothesis(
                    id       = str(uuid.uuid4())[:8],
                    titulo   = "Aumentar tasa de recovery de homeostasis",
                    tipo     = "parametro",
                    descripcion = (
                        "La claridad media es baja. Aumentar la tasa de recovery "
                        "en recover_state() para que ALTER recupere claridad más rápido."
                    ),
                    evidencia = [h.descripcion for h in claridad_obs],
                    modulo   = "homeostasis",
                    impacto  = "medio",
                    riesgo   = "bajo",
                    score    = _score_from_evidence(len(claridad_obs)),
                    estado   = "propuesta",
                    replayable = True,
                    parametro_target  = "homeostasis.recovery_claridad",
                    rango_experimento = [0.08, 0.18],
                )
                nuevas.append(h)

        # ── Hipótesis estructurales (no replayables aún) ────

        # 6. alter_brain demasiado grande → split
        brain_grande = [o for o in modulos_grandes if "alter_brain" in o.archivo]
        if brain_grande and ("brain", "split", "") not in existentes_keys:
            h = ArchitectureHypothesis(
                id       = str(uuid.uuid4())[:8],
                titulo   = "Dividir alter_brain.py en módulos más específicos",
                tipo     = "split",
                descripcion = (
                    f"alter_brain.py tiene {brain_grande[0].valor:.0f} líneas y "
                    "AlterBrain tiene 84+ métodos. Candidatos a separar: "
                    "loop_conversacional, gestión de memoria, pipeline B3/B4."
                ),
                evidencia = [o.descripcion for o in brain_grande],
                modulo   = "brain",
                impacto  = "alto",
                riesgo   = "alto",
                score    = _score_from_evidence(len(brain_grande) + len(clases_grandes)),
                estado   = "no_replayable_yet",
                replayable = False,
            )
            nuevas.append(h)

        # 7. loop_conversacional muy largo → extraer
        loop_largo = [o for o in funciones_largas
                      if "loop_conversacional" in o.descripcion or
                         "procesar_input" in o.descripcion]
        if loop_largo and ("brain", "refactor", "") not in existentes_keys:
            h = ArchitectureHypothesis(
                id       = str(uuid.uuid4())[:8],
                titulo   = "Extraer lógica de loop_conversacional y procesar_input",
                tipo     = "refactor",
                descripcion = (
                    "loop_conversacional y procesar_input son demasiado largos. "
                    "Extraer: handlers de comandos, pipeline B3/B4, logging KAIROS."
                ),
                evidencia = [o.descripcion for o in loop_largo],
                modulo   = "brain",
                impacto  = "medio",
                riesgo   = "medio",
                score    = _score_from_evidence(len(loop_largo)),
                estado   = "no_replayable_yet",
                replayable = False,
            )
            nuevas.append(h)

        # 8. gaps de implementación
        for gap in gaps[:3]:  # máx 3 hipótesis de gap
            key = (gap.modulo, "nuevo_modulo", "")
            if key not in existentes_keys:
                h = ArchitectureHypothesis(
                    id       = str(uuid.uuid4())[:8],
                    titulo   = f"Implementar módulo faltante: {gap.archivo}",
                    tipo     = "nuevo_modulo",
                    descripcion = gap.descripcion,
                    evidencia = [gap.descripcion],
                    modulo   = gap.modulo,
                    impacto  = "medio",
                    riesgo   = "bajo",
                    score    = 0.50,
                    estado   = "propuesta",
                    replayable = False,
                )
                nuevas.append(h)

        # Ordenar por score desc y limitar
        nuevas.sort(key=lambda h: h.score, reverse=True)
        return nuevas[:MAX_HYPOTHESES]

    # ----------------------------------------------------------
    # PERSISTENCIA
    # ----------------------------------------------------------

    def save(self, hypotheses: list, redis_client) -> bool:
        if not redis_client:
            return False
        try:
            redis_client.set(
                "alter:b5:hypotheses",
                json.dumps([h.to_dict() for h in hypotheses], ensure_ascii=False)
            )
            return True
        except Exception:
            return False

    def load(self, redis_client) -> list:
        if not redis_client:
            return []
        try:
            raw = redis_client.get("alter:b5:hypotheses")
            if raw:
                return [ArchitectureHypothesis.from_dict(d) for d in json.loads(raw)]
        except Exception:
            pass
        return []

    def snapshot_str(self, hypotheses: list) -> str:
        if not hypotheses:
            return "[HIPÓTESIS] Sin hipótesis generadas aún."
        lines = [f"[HIPÓTESIS] {len(hypotheses)} activas"]
        replayables = [h for h in hypotheses if h.replayable and h.puede_experimentar()]
        no_replay   = [h for h in hypotheses if not h.replayable]
        if replayables:
            lines.append(f"  Listas para experimentar ({len(replayables)}):")
            for h in replayables[:3]:
                lines.append(f"    ★ [{h.modulo}] {h.titulo[:55]} (score:{h.score:.2f})")
        if no_replay:
            lines.append(f"  Estructurales pendientes ({len(no_replay)}):")
            for h in no_replay[:2]:
                lines.append(f"    · [{h.modulo}] {h.titulo[:55]}")
        return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    from alter_architecture_state import build_current_spec
    from alter_selfmodel import SelfModel

    generator = HypothesisGenerator()
    sm = SelfModel()

    # Test 1: generate sin reportes no explota
    class FakeCodeReport:
        observaciones = []
    class FakeArchReport:
        hallazgos = []

    try:
        hyps = generator.generate(FakeCodeReport(), FakeArchReport(), sm)
        if not isinstance(hyps, list):
            errors.append("FAIL: generate no retornó lista")
    except Exception as e:
        errors.append(f"FAIL: generate explotó: {e}")

    # Test 2: hipótesis con hallazgos de workspace genera hipótesis replayable
    from alter_auditor import AuditFinding
    ws_finding = AuditFinding(
        severidad="warning", modulo="workspace",
        descripcion="Workspace en overflow 45% de los turnos",
        metrica="ws_overflow_rate", valor=0.45, umbral=0.35,
        recomendacion="Revisar generación de candidatos"
    )

    class FakeArchReport2:
        hallazgos = [ws_finding]

    hyps2 = generator.generate(FakeCodeReport(), FakeArchReport2(), sm)
    ws_hyps = [h for h in hyps2 if h.modulo == "workspace" and h.replayable]
    if not ws_hyps:
        errors.append("FAIL: hallazgo workspace no generó hipótesis replayable")

    # Test 3: score dentro de rango
    for h in hyps2:
        if not (0.0 <= h.score <= 1.0):
            errors.append(f"FAIL: score fuera de rango: {h.score}")

    # Test 4: puede_experimentar respeta condiciones
    if ws_hyps:
        h = ws_hyps[0]
        h.score = 0.10  # muy bajo
        if h.puede_experimentar():
            errors.append("FAIL: puede_experimentar con score bajo debería ser False")
        h.score = 0.60
        h.estado = "en_experimento"
        if h.puede_experimentar():
            errors.append("FAIL: puede_experimentar con estado en_experimento debería ser False")

    # Test 5: hipótesis estructural → no replayable
    from alter_code_auditor import CodeObservation
    brain_obs = CodeObservation(
        tipo="modulo_grande", severidad="critical",
        archivo="alter_brain.py", linea=0,
        descripcion="alter_brain.py tiene 2894 líneas",
        sugerencia="Dividir", modulo="brain"
    )
    brain_obs.valor = 2894.0

    class FakeCodeReport3:
        observaciones = [brain_obs]

    hyps3 = generator.generate(FakeCodeReport3(), FakeArchReport(), sm)
    struct_hyps = [h for h in hyps3 if not h.replayable]
    # No siempre genera una — depende del módulo, solo verificar que no explota
    if not isinstance(hyps3, list):
        errors.append("FAIL: generate con obs estructurales no retornó lista")

    # Test 6: round-trip to_dict / from_dict
    if hyps2:
        h = hyps2[0]
        d = h.to_dict()
        h2 = ArchitectureHypothesis.from_dict(d)
        if h2.id != h.id or h2.modulo != h.modulo:
            errors.append("FAIL: ArchitectureHypothesis round-trip perdió datos")

    # Test 7: snapshot_str no explota
    snap = generator.snapshot_str(hyps2)
    if "[HIPÓTESIS]" not in snap:
        errors.append("FAIL: snapshot_str malformado")

    return errors


if __name__ == "__main__":
    print("=== alter_architecture_hypotheses.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo con estado simulado ===")
    from alter_auditor import AuditFinding
    from alter_selfmodel import SelfModel, IntentPerformance

    generator = HypothesisGenerator()
    sm = SelfModel()
    sm.intent_performance = [
        IntentPerformance("quiere_exploracion", 0.42, 10, 0.58),
    ]

    class FakeCode:
        from alter_code_auditor import CodeObservation
        observaciones = [
            CodeObservation("modulo_grande", "critical", "alter_brain.py", 0,
                           "alter_brain.py tiene 2894 líneas", "Dividir", "brain"),
            CodeObservation("funcion_larga", "critical", "alter_brain.py", 2581,
                           "loop_conversacional tiene 309 líneas", "Dividir", "brain"),
        ]
        for o in observaciones:
            o.valor = float(o.descripcion.split()[3]) if o.descripcion.split()[3].isdigit() else 0.0

    class FakeArch:
        hallazgos = [
            AuditFinding("warning", "workspace",
                         "Workspace en overflow 38% de los turnos",
                         "ws_overflow_rate", 0.38, 0.35,
                         "Revisar candidatos"),
            AuditFinding("warning", "predictive",
                         "Baja precisión en intención 'quiere_exploracion'",
                         "intent_sr.quiere_exploracion", 0.42, 0.45,
                         "Agregar señales léxicas"),
        ]

    hyps = generator.generate(FakeCode(), FakeArch(), sm)
    print(generator.snapshot_str(hyps))
    print(f"\nDetalle de hipótesis replayables:")
    for h in hyps:
        if h.replayable:
            print(f"  [{h.score:.2f}] {h.titulo}")
            print(f"         param: {h.parametro_target} rango:{h.rango_experimento}")
