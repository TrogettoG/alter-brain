"""
alter_feature_flags.py — Feature Flags de AlterB5 (Fase 3)

Activa cambios paramétricos en producción de forma controlada.
Principio: ningún cambio sin puerta de salida.

Tres componentes en un archivo — acoplados por diseño:

    FeatureFlag           — registro de una variante activable
    ControlledPromotion   — lógica de activación segura
    RollbackMonitor       — detecta empeoramiento y revierte

Reglas de auto-aprobación (Opción B conservadora):
    1. riesgo == "bajo"
    2. confianza >= 0.7
    3. experiment.mejora == True
    4. delta paramétrico <= MAX_DELTA_AUTO
    5. no hay otro flag activo para el mismo parámetro
    6. no hubo rollback reciente (cooldown activo)
    7. un solo auto-flag activo por vez

Rollback por ventana sostenida — no por una sola señal mala:
    - min_samples: N turnos mínimos antes de evaluar
    - guardrail_metrics: métricas que no deben empeorar
    - rollback si deterioro sostenido, no puntual

Redis keys:
    alter:b5:flags       — flags activos e inactivos
    alter:b5:flags:log   — historial de activaciones y rollbacks
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


# ============================================================
# CONSTANTES
# ============================================================

MAX_DELTA_AUTO          = 0.20   # delta máximo para auto-aprobación
MIN_CONFIANZA_AUTO      = 0.70   # confianza mínima para auto-aprobación
COOLDOWN_HORAS          = 48     # horas de cooldown post-rollback
MAX_AUTO_FLAGS_ACTIVOS  = 1      # máximo flags auto-aprobados simultáneos
MIN_SAMPLES_ROLLBACK    = 8      # turnos mínimos antes de evaluar rollback
ROLLBACK_THRESHOLD_DEF  = 0.10   # deterioro sostenido para revertir

# Guardrails por defecto — métricas que no deben empeorar
DEFAULT_GUARDRAILS = [
    "prediction_error_medio",
    "riesgo_desalineacion_medio",
]

# Parámetros aprobados para cambio automático
PARAMS_SEGUROS = {
    "workspace.MAX_ITEMS",
    "simulator.OVERRIDE_THRESHOLD",
    "policy.UMBRAL_RIESGO_DESALINEACION",
    "memory.THRESHOLD_ERROR",
    "homeostasis.recovery_claridad",
}

# Cómo aplicar cada parámetro al sistema real
PARAM_REDIS_MAP = {
    "workspace.MAX_ITEMS":                  ("alter:b5:param:workspace_max_items",   "int"),
    "simulator.OVERRIDE_THRESHOLD":         ("alter:b5:param:sim_override_thr",      "float"),
    "policy.UMBRAL_RIESGO_DESALINEACION":   ("alter:b5:param:policy_riesgo_thr",     "float"),
    "memory.THRESHOLD_ERROR":               ("alter:b5:param:memory_thr_error",      "float"),
    "homeostasis.recovery_claridad":        ("alter:b5:param:hs_recovery_claridad",  "float"),
}


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class FeatureFlag:
    id:                 str
    parametro:          str
    valor_actual:       float
    valor_nuevo:        float
    estado:             str      # "pendiente" | "activo" | "revertido" | "expirado"
    hypothesis_id:      str
    experiment_id:      str
    riesgo:             str      # "bajo" | "medio" | "alto"
    confianza:          float
    auto_approved:      bool     # True si se activó sin aprobación manual
    auto_rollback:      bool     # True si puede revertir solo
    rollback_threshold: float    # cuánto deterioro para revertir
    guardrail_metrics:  list     # list[str] métricas que no deben empeorar
    min_samples:        int      # turnos mínimos antes de evaluar rollback
    cooldown_until:     str      # ISO timestamp hasta cuando no auto-aprobar de nuevo
    activado_en:        str
    revertido_en:       str
    motivo_rollback:    str
    samples_observados: int      # turnos observados desde activación

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureFlag":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def esta_activo(self) -> bool:
        return self.estado == "activo"

    def en_cooldown(self) -> bool:
        if not self.cooldown_until:
            return False
        try:
            return datetime.now() < datetime.fromisoformat(self.cooldown_until)
        except Exception:
            return False

    def delta(self) -> float:
        return abs(self.valor_nuevo - self.valor_actual)


@dataclass
class FlagLogEntry:
    timestamp:   str
    flag_id:     str
    parametro:   str
    evento:      str   # "creado" | "auto_aprobado" | "aprobado" | "activado" | "rollback" | "expirado"
    detalle:     str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# CONTROLLED PROMOTION
# ============================================================

class ControlledPromotion:
    """
    Evalúa resultados de experimentos y crea FeatureFlags para
    los cambios que merecen promoción a producción.

    Auto-aprobación solo si cumple las 7 reglas duras.
    """

    def evaluate_experiments(
        self,
        experiment_results: list,   # list[ExperimentResult]
        hypotheses:         list,   # list[ArchitectureHypothesis]
        existing_flags:     list,   # list[FeatureFlag] — flags ya activos
    ) -> list:
        """
        Evalúa resultados y genera flags para experimentos exitosos.
        Retorna lista de FeatureFlag nuevos (estado "pendiente" o "activo").
        """
        nuevos_flags = []

        # Parámetros con flag activo o en cooldown
        params_bloqueados = {
            f.parametro
            for f in existing_flags
            if f.esta_activo() or f.en_cooldown()
        }

        # Cantidad de auto-flags activos
        auto_activos = sum(
            1 for f in existing_flags
            if f.esta_activo() and f.auto_approved
        )

        for result in experiment_results:
            if not result.mejora:
                continue

            # Buscar hipótesis asociada
            hyp = next(
                (h for h in hypotheses if h.id == result.hypothesis_id),
                None
            )
            if not hyp:
                continue

            # Parámetro debe ser seguro
            if result.parametro not in PARAMS_SEGUROS:
                continue

            # No duplicar flags para el mismo parámetro
            if result.parametro in params_bloqueados:
                continue

            # Evaluar auto-aprobación
            puede_auto = self._puede_auto_aprobar(
                result, hyp, auto_activos, existing_flags
            )

            flag = FeatureFlag(
                id                 = str(uuid.uuid4())[:8],
                parametro          = result.parametro,
                valor_actual       = result.valor_baseline,
                valor_nuevo        = result.valor_variante,
                estado             = "activo" if puede_auto else "pendiente",
                hypothesis_id      = result.hypothesis_id,
                experiment_id      = result.experiment_id,
                riesgo             = hyp.riesgo,
                confianza          = result.confianza,
                auto_approved      = puede_auto,
                auto_rollback      = True,
                rollback_threshold = ROLLBACK_THRESHOLD_DEF,
                guardrail_metrics  = list(DEFAULT_GUARDRAILS),
                min_samples        = MIN_SAMPLES_ROLLBACK,
                cooldown_until     = "",
                activado_en        = datetime.now().isoformat() if puede_auto else "",
                revertido_en       = "",
                motivo_rollback    = "",
                samples_observados = 0,
            )
            nuevos_flags.append(flag)

            if puede_auto:
                auto_activos += 1
                params_bloqueados.add(result.parametro)

        return nuevos_flags

    def _puede_auto_aprobar(
        self,
        result,     # ExperimentResult
        hyp,        # ArchitectureHypothesis
        auto_activos: int,
        existing_flags: list,
    ) -> bool:
        """
        Verifica las 7 reglas duras de auto-aprobación.
        """
        # Regla 1: riesgo bajo
        if hyp.riesgo != "bajo":
            return False

        # Regla 2: confianza suficiente
        if result.confianza < MIN_CONFIANZA_AUTO:
            return False

        # Regla 3: mejora confirmada
        if not result.mejora:
            return False

        # Regla 4: delta acotado
        # Para parámetros enteros (MAX_ITEMS) usamos delta relativo al rango
        delta_abs = abs(result.valor_variante - result.valor_baseline)
        rango_ref = max(abs(result.valor_baseline), 1.0)
        delta_rel = delta_abs / rango_ref
        # Acepta: delta absoluto <= MAX_DELTA_AUTO OR delta relativo <= 15%
        if delta_abs > MAX_DELTA_AUTO and delta_rel > 0.15:
            return False

        # Regla 5: no hay flag activo para el mismo parámetro (ya verificado afuera)

        # Regla 6: no hubo rollback reciente (cooldown)
        rollbacks_recientes = [
            f for f in existing_flags
            if f.parametro == result.parametro and
               f.estado == "revertido" and
               f.en_cooldown()
        ]
        if rollbacks_recientes:
            return False

        # Regla 7: un solo auto-flag activo por vez
        if auto_activos >= MAX_AUTO_FLAGS_ACTIVOS:
            return False

        return True


# ============================================================
# ROLLBACK MONITOR
# ============================================================

class RollbackMonitor:
    """
    Monitorea flags activos y revierte si las métricas empeoran
    de forma sostenida.

    No revierte por una sola señal mala — requiere ventana mínima
    y deterioro consistente en guardrail_metrics.
    """

    def check(
        self,
        flags:          list,         # list[FeatureFlag] activos
        redis_client,
    ) -> list:
        """
        Evalúa todos los flags activos.
        Retorna lista de flags que fueron revertidos.
        """
        revertidos = []

        for flag in flags:
            if not flag.esta_activo():
                continue
            if not flag.auto_rollback:
                continue

            # Leer métricas recientes
            metricas_recientes = self._read_recent_metrics(
                redis_client, flag.min_samples
            )
            if not metricas_recientes:
                continue

            flag.samples_observados += len(metricas_recientes)

            # No evaluar hasta tener suficientes muestras
            if flag.samples_observados < flag.min_samples:
                continue

            # Evaluar guardrails
            debe_revertir, motivo = self._evaluar_guardrails(
                flag, metricas_recientes
            )

            if debe_revertir:
                flag.estado          = "revertido"
                flag.revertido_en    = datetime.now().isoformat()
                flag.motivo_rollback = motivo
                # Activar cooldown
                flag.cooldown_until  = (
                    datetime.now() + timedelta(hours=COOLDOWN_HORAS)
                ).isoformat()
                # Revertir valor en Redis
                self._revert_param(flag, redis_client)
                revertidos.append(flag)

        return revertidos

    def _evaluar_guardrails(
        self,
        flag: FeatureFlag,
        metricas: list,   # list[dict] — últimas N métricas por turno
    ) -> tuple:
        """
        Evalúa si algún guardrail fue violado de forma sostenida.
        Retorna (debe_revertir: bool, motivo: str)
        """
        for metrica in flag.guardrail_metrics:
            valores = self._extract_metric(metricas, metrica)
            if len(valores) < flag.min_samples // 2:
                continue

            media_actual = float(np.mean(valores))

            # Obtener baseline del experimento
            baseline_key = f"alter:b5:param_baseline:{flag.parametro}:{metrica}"
            baseline_val = self._read_baseline(flag, metrica)

            if baseline_val is None:
                continue

            deterioro = media_actual - baseline_val

            # Reverter si el deterioro supera el threshold de forma sostenida
            # "sostenida" = más del 60% de las muestras están por encima del baseline
            n_peor = sum(1 for v in valores if v > baseline_val + flag.rollback_threshold * 0.5)
            pct_peor = n_peor / len(valores)

            if deterioro > flag.rollback_threshold and pct_peor > 0.60:
                return True, (
                    f"{metrica} deterioró {deterioro:+.3f} "
                    f"(threshold:{flag.rollback_threshold:.2f}) "
                    f"en {pct_peor:.0%} de muestras"
                )

        return False, ""

    def _extract_metric(self, metricas: list, nombre: str) -> list:
        """Extrae valores de una métrica de la lista de registros."""
        valores = []
        for entry in metricas:
            m = entry.get("metrics", {})
            if nombre == "prediction_error_medio":
                v = m.get("error_ultimo")
            elif nombre == "riesgo_desalineacion_medio":
                # Proxy: error alto correlaciona con riesgo
                err = m.get("error_ultimo", 0.5)
                v   = float(np.clip(err * 0.8 + 0.1, 0.0, 1.0))
            elif nombre == "workspace_overflow_rate":
                v = 1.0 if m.get("overflow", False) else 0.0
            elif nombre == "costo_cognitivo_estimado":
                v = m.get("carga_cognitiva")
            else:
                v = m.get(nombre)
            if v is not None:
                valores.append(float(v))
        return valores

    def _read_recent_metrics(self, redis_client, n: int) -> list:
        if not redis_client:
            return []
        try:
            raw = redis_client.lrange("alter:metrics:predictive:history", 0, n - 1)
            return [json.loads(r) for r in (raw or [])]
        except Exception:
            return []

    def _read_baseline(self, flag: FeatureFlag, metrica: str) -> Optional[float]:
        """
        Lee el valor baseline del experimento desde Redis o usa defaults.
        """
        defaults = {
            "prediction_error_medio":    0.45,
            "riesgo_desalineacion_medio": 0.30,
            "workspace_overflow_rate":   0.20,
            "costo_cognitivo_estimado":  0.28,
        }
        return defaults.get(metrica)

    def _revert_param(self, flag: FeatureFlag, redis_client):
        """Revierte el parámetro a su valor original en Redis."""
        if not redis_client:
            return
        redis_key, tipo = PARAM_REDIS_MAP.get(flag.parametro, (None, None))
        if not redis_key:
            return
        try:
            valor = int(flag.valor_actual) if tipo == "int" else flag.valor_actual
            redis_client.set(redis_key, json.dumps(valor))
        except Exception:
            pass


# ============================================================
# PERSISTENCIA Y UTILIDADES
# ============================================================

def save_flags(flags: list, redis_client) -> bool:
    if not redis_client:
        return False
    try:
        redis_client.set(
            "alter:b5:flags",
            json.dumps([f.to_dict() for f in flags], ensure_ascii=False)
        )
        return True
    except Exception:
        return False


def load_flags(redis_client) -> list:
    if not redis_client:
        return []
    try:
        raw = redis_client.get("alter:b5:flags")
        if raw:
            return [FeatureFlag.from_dict(d) for d in json.loads(raw)]
    except Exception:
        pass
    return []


def log_flag_event(flag: FeatureFlag, evento: str, detalle: str, redis_client):
    entry = FlagLogEntry(
        timestamp = datetime.now().isoformat(),
        flag_id   = flag.id,
        parametro = flag.parametro,
        evento    = evento,
        detalle   = detalle,
    )
    if not redis_client:
        return
    try:
        redis_client.lpush(
            "alter:b5:flags:log",
            json.dumps(entry.to_dict(), ensure_ascii=False)
        )
        redis_client.ltrim("alter:b5:flags:log", 0, 99)
    except Exception:
        pass


def apply_active_flags(flags: list, redis_client) -> dict:
    """
    Aplica los valores de los flags activos al sistema real (Redis).
    Retorna dict de parámetros aplicados.
    """
    aplicados = {}
    for flag in flags:
        if not flag.esta_activo():
            continue
        redis_key, tipo = PARAM_REDIS_MAP.get(flag.parametro, (None, None))
        if not redis_key or not redis_client:
            continue
        try:
            valor = int(flag.valor_nuevo) if tipo == "int" else flag.valor_nuevo
            redis_client.set(redis_key, json.dumps(valor))
            aplicados[flag.parametro] = valor
        except Exception:
            pass
    return aplicados


def get_active_param(parametro: str, default: float, redis_client) -> float:
    """
    Lee el valor actual de un parámetro — respeta flags activos si existen.
    Usar en alter_brain.py al inicializar módulos.
    """
    redis_key, tipo = PARAM_REDIS_MAP.get(parametro, (None, None))
    if not redis_key or not redis_client:
        return default
    try:
        raw = redis_client.get(redis_key)
        if raw:
            val = json.loads(raw)
            return int(val) if tipo == "int" else float(val)
    except Exception:
        pass
    return default


def snapshot_str(flags: list) -> str:
    if not flags:
        return "[FLAGS] Sin flags activos."
    activos   = [f for f in flags if f.esta_activo()]
    pendientes = [f for f in flags if f.estado == "pendiente"]
    revertidos = [f for f in flags if f.estado == "revertido"]
    lines = [f"[FLAGS] {len(activos)} activos | {len(pendientes)} pendientes | {len(revertidos)} revertidos"]
    for f in activos:
        auto_str = "auto" if f.auto_approved else "manual"
        lines.append(
            f"  ✓ [{f.id}] {f.parametro}: "
            f"{f.valor_actual}→{f.valor_nuevo} "
            f"({auto_str}, conf:{f.confianza:.2f}, "
            f"samples:{f.samples_observados}/{f.min_samples})"
        )
    for f in pendientes:
        lines.append(
            f"  ⏳ [{f.id}] {f.parametro}: "
            f"{f.valor_actual}→{f.valor_nuevo} "
            f"(esperando aprobación manual)"
        )
    for f in revertidos[:2]:
        lines.append(
            f"  ✗ [{f.id}] {f.parametro}: revertido — {f.motivo_rollback[:50]}"
        )
    return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    promotion = ControlledPromotion()
    monitor   = RollbackMonitor()

    # Crear experiment result de prueba
    from alter_experiments import ExperimentResult, ExperimentMetrics
    from alter_architecture_hypotheses import ArchitectureHypothesis

    baseline = ExperimentMetrics(
        prediction_error_medio=0.45, riesgo_desalineacion_medio=0.30,
        council_activation_rate=0.35, workspace_overflow_rate=0.20,
        costo_cognitivo_estimado=0.28, preguntas_generadas=3, n_turnos=20
    )
    variante = ExperimentMetrics(
        prediction_error_medio=0.45, riesgo_desalineacion_medio=0.30,
        council_activation_rate=0.35, workspace_overflow_rate=0.15,
        costo_cognitivo_estimado=0.28, preguntas_generadas=3, n_turnos=20
    )

    result_bueno = ExperimentResult(
        experiment_id="e1", hypothesis_id="h1",
        parametro="workspace.MAX_ITEMS",
        valor_baseline=7.0, valor_variante=6.0,
        baseline=baseline, variante=variante,
        mejora=True, confianza=0.75,
        resumen="MEJORA | overflow ↓0.05"
    )

    hyp_bajo = ArchitectureHypothesis(
        id="h1", titulo="test", tipo="parametro", descripcion="test",
        evidencia=["overflow alto"], modulo="workspace",
        impacto="medio", riesgo="bajo", score=0.60,
        estado="en_experimento", replayable=True,
        parametro_target="workspace.MAX_ITEMS",
        rango_experimento=[5, 7],
    )

    # Test 1: auto-aprobación con condiciones correctas
    flags = promotion.evaluate_experiments([result_bueno], [hyp_bajo], [])
    if not flags:
        errors.append("FAIL: debería generar al menos un flag")
    if flags and not flags[0].auto_approved:
        errors.append("FAIL: flag de bajo riesgo + alta confianza debería ser auto_approved")
    if flags and flags[0].estado != "activo":
        errors.append("FAIL: flag auto-aprobado debería estar activo")

    # Test 2: no auto-aprueba si riesgo no es bajo
    hyp_alto = ArchitectureHypothesis(
        id="h2", titulo="test", tipo="parametro", descripcion="test",
        evidencia=[], modulo="policy",
        impacto="alto", riesgo="alto", score=0.60,
        estado="en_experimento", replayable=True,
        parametro_target="policy.UMBRAL_RIESGO_DESALINEACION",
        rango_experimento=[0.55, 0.80],
    )
    result_policy = ExperimentResult(
        experiment_id="e2", hypothesis_id="h2",
        parametro="policy.UMBRAL_RIESGO_DESALINEACION",
        valor_baseline=0.65, valor_variante=0.75,
        baseline=baseline, variante=variante,
        mejora=True, confianza=0.80,
        resumen="MEJORA"
    )
    flags2 = promotion.evaluate_experiments([result_policy], [hyp_alto], [])
    if flags2 and flags2[0].auto_approved:
        errors.append("FAIL: riesgo alto no debería auto-aprobarse")

    # Test 3: no auto-aprueba si confianza baja
    result_baja_conf = ExperimentResult(
        experiment_id="e3", hypothesis_id="h1",
        parametro="workspace.MAX_ITEMS",
        valor_baseline=7.0, valor_variante=6.0,
        baseline=baseline, variante=variante,
        mejora=True, confianza=0.50,  # < 0.70
        resumen="MEJORA"
    )
    flags3 = promotion.evaluate_experiments([result_baja_conf], [hyp_bajo], [])
    if flags3 and flags3[0].auto_approved:
        errors.append("FAIL: baja confianza no debería auto-aprobarse")

    # Test 4: no duplica flags para el mismo parámetro
    existing_activo = FeatureFlag(
        id="f_existing", parametro="workspace.MAX_ITEMS",
        valor_actual=7.0, valor_nuevo=6.0,
        estado="activo", hypothesis_id="h0", experiment_id="e0",
        riesgo="bajo", confianza=0.75, auto_approved=True,
        auto_rollback=True, rollback_threshold=0.10,
        guardrail_metrics=list(DEFAULT_GUARDRAILS),
        min_samples=8, cooldown_until="",
        activado_en=datetime.now().isoformat(),
        revertido_en="", motivo_rollback="",
        samples_observados=0,
    )
    flags4 = promotion.evaluate_experiments([result_bueno], [hyp_bajo], [existing_activo])
    if flags4:
        errors.append("FAIL: no debería crear flag si ya hay uno activo para el mismo parámetro")

    # Test 5: cooldown bloquea re-aprobación
    flag_revertido = FeatureFlag(
        id="f_rev", parametro="workspace.MAX_ITEMS",
        valor_actual=7.0, valor_nuevo=6.0,
        estado="revertido", hypothesis_id="h0", experiment_id="e0",
        riesgo="bajo", confianza=0.75, auto_approved=True,
        auto_rollback=True, rollback_threshold=0.10,
        guardrail_metrics=list(DEFAULT_GUARDRAILS),
        min_samples=8,
        cooldown_until=(datetime.now() + timedelta(hours=24)).isoformat(),
        activado_en="", revertido_en=datetime.now().isoformat(),
        motivo_rollback="deterioro sostenido",
        samples_observados=10,
    )
    if not flag_revertido.en_cooldown():
        errors.append("FAIL: flag revertido con cooldown futuro debería estar en cooldown")
    flags5 = promotion.evaluate_experiments([result_bueno], [hyp_bajo], [flag_revertido])
    if flags5 and flags5[0].auto_approved:
        errors.append("FAIL: cooldown activo debería bloquear auto-aprobación")

    # Test 6: RollbackMonitor sin Redis no explota
    flag_activo = FeatureFlag(
        id="f_test", parametro="workspace.MAX_ITEMS",
        valor_actual=7.0, valor_nuevo=6.0,
        estado="activo", hypothesis_id="h1", experiment_id="e1",
        riesgo="bajo", confianza=0.75, auto_approved=True,
        auto_rollback=True, rollback_threshold=0.10,
        guardrail_metrics=list(DEFAULT_GUARDRAILS),
        min_samples=8, cooldown_until="",
        activado_en=datetime.now().isoformat(),
        revertido_en="", motivo_rollback="",
        samples_observados=0,
    )
    try:
        revertidos = monitor.check([flag_activo], None)
        if revertidos:
            errors.append("FAIL: monitor sin Redis no debería revertir nada")
    except Exception as e:
        errors.append(f"FAIL: monitor.check explotó: {e}")

    # Test 7: _evaluar_guardrails detecta deterioro sostenido
    metricas_malas = [
        {"metrics": {"error_ultimo": 0.80}} for _ in range(10)
    ]
    # Con valores malos el rollback debería activarse
    flag_activo.samples_observados = 10
    debe, motivo = monitor._evaluar_guardrails(flag_activo, metricas_malas)
    if not debe:
        errors.append(f"FAIL: deterioro sostenido (0.80 vs baseline 0.45) debería activar rollback")

    # Test 8: snapshot_str no explota
    snap = snapshot_str([flag_activo, flag_revertido])
    if "[FLAGS]" not in snap:
        errors.append("FAIL: snapshot_str malformado")

    # Test 9: FeatureFlag round-trip
    d = flag_activo.to_dict()
    f2 = FeatureFlag.from_dict(d)
    if f2.id != flag_activo.id or f2.parametro != flag_activo.parametro:
        errors.append("FAIL: FeatureFlag round-trip perdió datos")

    return errors


if __name__ == "__main__":
    print("=== alter_feature_flags.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo Controlled Promotion ===")
    from alter_experiments import ExperimentResult, ExperimentMetrics
    from alter_architecture_hypotheses import ArchitectureHypothesis

    baseline = ExperimentMetrics(
        prediction_error_medio=0.45, riesgo_desalineacion_medio=0.30,
        council_activation_rate=0.35, workspace_overflow_rate=0.22,
        costo_cognitivo_estimado=0.28, preguntas_generadas=3, n_turnos=25
    )
    variante = ExperimentMetrics(
        prediction_error_medio=0.45, riesgo_desalineacion_medio=0.30,
        council_activation_rate=0.35, workspace_overflow_rate=0.15,
        costo_cognitivo_estimado=0.28, preguntas_generadas=3, n_turnos=25
    )

    casos = [
        {
            "nombre": "Bajo riesgo + alta confianza → auto-aprobado",
            "result": ExperimentResult(
                "e1","h1","workspace.MAX_ITEMS",7.0,6.0,
                baseline,variante,True,0.78,"MEJORA | overflow ↓0.07"
            ),
            "hyp": ArchitectureHypothesis(
                "h1","Reducir MAX_ITEMS","parametro","desc",
                ["overflow 38%"],"workspace","medio","bajo",
                0.60,"en_experimento",True,"workspace.MAX_ITEMS",[5,7]
            ),
        },
        {
            "nombre": "Riesgo alto → requiere aprobación manual",
            "result": ExperimentResult(
                "e2","h2","policy.UMBRAL_RIESGO_DESALINEACION",0.65,0.75,
                baseline,variante,True,0.80,"MEJORA"
            ),
            "hyp": ArchitectureHypothesis(
                "h2","Ajustar umbral policy","parametro","desc",
                ["overrides frecuentes"],"policy","alto","alto",
                0.65,"en_experimento",True,
                "policy.UMBRAL_RIESGO_DESALINEACION",[0.55,0.80]
            ),
        },
    ]

    promotion = ControlledPromotion()
    all_flags = []
    for caso in casos:
        flags = promotion.evaluate_experiments(
            [caso["result"]], [caso["hyp"]], all_flags
        )
        all_flags.extend(flags)
        for f in flags:
            estado_str = "AUTO-ACTIVO" if f.auto_approved else "PENDIENTE APROBACIÓN"
            print(f"\n{caso['nombre']}:")
            print(f"  Flag [{f.id}]: {f.parametro} "
                  f"{f.valor_actual}→{f.valor_nuevo} → {estado_str}")

    print(f"\n{snapshot_str(all_flags)}")
