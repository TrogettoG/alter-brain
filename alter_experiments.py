"""
alter_experiments.py — Experiment Runner de AlterB5 (Fase 2)

Prueba hipótesis paramétricas replayables contra trazas históricas.
Sin llamadas a Gemini. Sin tocar producción.

Baseline: trazas por turno ya registradas por B4 (alter:metrics:*:history)
Variante: mismo parámetro con valor alternativo

Qué puede variar:
    workspace.MAX_ITEMS               — afecta overflow rate
    simulator.OVERRIDE_THRESHOLD     — afecta frecuencia de overrides
    policy.UMBRAL_RIESGO_DESALINEACION— afecta frecuencia de preguntas
    memory.THRESHOLD_ERROR            — afecta aprendizaje procedural
    homeostasis.recovery_claridad     — afecta recuperación de claridad

Métricas comparadas (Replay Evaluator):
    prediction_error_medio
    riesgo_desalineacion_medio
    council_activation_rate
    workspace_overflow_rate
    costo_cognitivo_estimado
    preguntas_generadas

Redis keys:
    alter:b5:experiments         — lista de experimentos
    alter:b5:experiments:results — resultados de comparaciones
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
class ExperimentMetrics:
    """Métricas medidas o estimadas para un escenario."""
    prediction_error_medio:    float
    riesgo_desalineacion_medio: float
    council_activation_rate:   float
    workspace_overflow_rate:   float
    costo_cognitivo_estimado:  float
    preguntas_generadas:       int
    n_turnos:                  int   # turnos analizados

    def to_dict(self) -> dict:
        return asdict(self)

    def delta(self, other: "ExperimentMetrics") -> dict:
        """Diferencia self - other para cada métrica."""
        return {
            "prediction_error_medio":     self.prediction_error_medio - other.prediction_error_medio,
            "riesgo_desalineacion_medio":  self.riesgo_desalineacion_medio - other.riesgo_desalineacion_medio,
            "council_activation_rate":    self.council_activation_rate - other.council_activation_rate,
            "workspace_overflow_rate":    self.workspace_overflow_rate - other.workspace_overflow_rate,
            "costo_cognitivo_estimado":   self.costo_cognitivo_estimado - other.costo_cognitivo_estimado,
            "preguntas_generadas":        self.preguntas_generadas - other.preguntas_generadas,
        }


@dataclass
class ExperimentResult:
    """Resultado completo de un experimento."""
    experiment_id:   str
    hypothesis_id:   str
    parametro:       str
    valor_baseline:  float
    valor_variante:  float
    baseline:        ExperimentMetrics
    variante:        ExperimentMetrics
    mejora:          bool    # True si la variante es mejor
    confianza:       float   # 0..1 — qué tan confiable es la comparación
    resumen:         str
    timestamp:       str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Experiment:
    id:              str
    hypothesis_id:   str
    parametro:       str
    valor_baseline:  float
    valor_variante:  float
    baseline_type:   str   # "trace_replay" | "summary_only"
    modo:            str   # "replay" — shadow es Fase 3
    estado:          str   # "pendiente" | "corriendo" | "completado" | "fallido"
    confidence:      float = 0.0
    resultado:       Optional[dict] = None
    created_at:      str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at:    str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================
# REPLAY EVALUATOR
# ============================================================

class ReplayEvaluator:
    """
    Evalúa el impacto de un cambio paramétrico sobre trazas históricas.

    Baseline: métricas reales del sistema de las trazas almacenadas.
    Variante: estimación de cómo cambiarían esas métricas con el nuevo valor.

    Sin Gemini. Sin re-ejecución generativa.
    Solo análisis de telemetría por turno.
    """

    MIN_TURNOS = 5   # mínimo de turnos para confiar en el resultado

    def compute_baseline(self, redis_client) -> Optional[ExperimentMetrics]:
        """
        Lee trazas históricas de B4 y computa métricas baseline reales.
        Fuente: alter:metrics:*:history
        """
        if not redis_client:
            return None

        pred_turns   = self._load_history(redis_client, "predictive", 50)
        ws_turns     = self._load_history(redis_client, "workspace",  50)
        policy_turns = self._load_history(redis_client, "policy",     50)
        hs_turns     = self._load_history(redis_client, "homeostasis", 50)

        n = max(len(pred_turns), len(ws_turns), 1)
        if n < self.MIN_TURNOS:
            return None

        # prediction_error_medio — desde trazas predictive
        errors = [t.get("metrics", {}).get("error_ultimo", 0.5)
                  for t in pred_turns if "metrics" in t]
        pred_error = float(np.mean(errors)) if errors else 0.5

        # riesgo_desalineacion — desde trazas predictive (expected_effect no siempre)
        # Proxy: error alto correlaciona con riesgo
        riesgo = float(np.clip(pred_error * 0.8 + 0.1, 0.0, 1.0))

        # council_activation_rate — desde trazas predictive (council_tension)
        council_acts = [
            1 if t.get("metrics", {}).get("error_ultimo", 0) > 0.5 else 0
            for t in pred_turns
        ]
        council_rate = float(np.mean(council_acts)) if council_acts else 0.3

        # workspace_overflow_rate — desde trazas workspace
        overflows = [
            1 if t.get("metrics", {}).get("overflow", False) else 0
            for t in ws_turns
        ]
        overflow_rate = float(np.mean(overflows)) if overflows else 0.2

        # costo_cognitivo — desde trazas homeostasis (fatiga + carga)
        costos = [
            t.get("metrics", {}).get("carga_cognitiva", 0.3)
            for t in hs_turns if "metrics" in t
        ]
        costo = float(np.mean(costos)) if costos else 0.3

        # preguntas — desde trazas policy
        preguntas = sum(
            1 for t in policy_turns
            if t.get("metrics", {}).get("action") == "preguntar"
        )

        return ExperimentMetrics(
            prediction_error_medio    = pred_error,
            riesgo_desalineacion_medio = riesgo,
            council_activation_rate   = council_rate,
            workspace_overflow_rate   = overflow_rate,
            costo_cognitivo_estimado  = costo,
            preguntas_generadas       = preguntas,
            n_turnos                  = n,
        )

    def estimate_variante(
        self,
        baseline: ExperimentMetrics,
        parametro: str,
        valor_baseline: float,
        valor_variante: float,
    ) -> ExperimentMetrics:
        """
        Estima las métricas de la variante aplicando el efecto esperado
        del cambio paramétrico sobre las métricas baseline.

        Lógica heurística por parámetro — sin Gemini.
        """
        ratio = valor_variante / max(valor_baseline, 0.001)

        pred_error   = baseline.prediction_error_medio
        riesgo       = baseline.riesgo_desalineacion_medio
        council_rate = baseline.council_activation_rate
        overflow     = baseline.workspace_overflow_rate
        costo        = baseline.costo_cognitivo_estimado
        preguntas    = baseline.preguntas_generadas

        if parametro == "workspace.MAX_ITEMS":
            # Menos items → menos overflow, más selección estricta
            # Más items → más overflow, más contexto disponible
            delta_overflow = (valor_variante - valor_baseline) * 0.05
            overflow       = float(np.clip(overflow + delta_overflow, 0.0, 1.0))
            # Menos items → ligeramente más costo (más evicción)
            costo          = float(np.clip(costo + abs(delta_overflow) * 0.3, 0.0, 1.0))

        elif parametro == "simulator.OVERRIDE_THRESHOLD":
            # Threshold más alto → menos overrides → más preguntas evitadas
            # Threshold más bajo → más overrides → más preguntas
            delta_override = (valor_variante - valor_baseline) * 2.0
            preguntas_delta = int(-delta_override * baseline.n_turnos * 0.1)
            preguntas       = max(0, preguntas + preguntas_delta)

        elif parametro == "policy.UMBRAL_RIESGO_DESALINEACION":
            # Umbral más alto → menos preguntas, más riesgo de desalineación
            # Umbral más bajo → más preguntas, menos riesgo
            delta = (valor_variante - valor_baseline)
            riesgo    = float(np.clip(riesgo + delta * 0.3, 0.0, 1.0))
            preguntas = max(0, preguntas + int(-delta * baseline.n_turnos * 0.15))

        elif parametro == "memory.THRESHOLD_ERROR":
            # Threshold más alto → menos aprendizaje procedural → más errores a largo plazo
            # Threshold más bajo → más aprendizaje → puede mejorar pred_error
            delta      = (valor_variante - valor_baseline)
            pred_error = float(np.clip(pred_error + delta * 0.1, 0.0, 1.0))

        elif parametro == "homeostasis.recovery_claridad":
            # Tasa más alta → más claridad → menos costo cognitivo
            delta_clarity = (valor_variante - valor_baseline) * 2.0
            costo         = float(np.clip(costo - delta_clarity * 0.2, 0.0, 1.0))
            pred_error    = float(np.clip(pred_error - delta_clarity * 0.05, 0.0, 1.0))

        return ExperimentMetrics(
            prediction_error_medio    = pred_error,
            riesgo_desalineacion_medio = riesgo,
            council_activation_rate   = council_rate,
            workspace_overflow_rate   = overflow,
            costo_cognitivo_estimado  = costo,
            preguntas_generadas       = preguntas,
            n_turnos                  = baseline.n_turnos,
        )

    def compare(
        self,
        baseline: ExperimentMetrics,
        variante: ExperimentMetrics,
        parametro: str,
    ) -> tuple:
        """
        Compara baseline vs variante.
        Retorna (mejora: bool, confianza: float, resumen: str)

        Métricas donde MENOS es mejor:
            prediction_error_medio, riesgo_desalineacion_medio,
            workspace_overflow_rate, costo_cognitivo_estimado

        Métricas donde depende:
            council_activation_rate, preguntas_generadas
        """
        deltas = variante.delta(baseline)

        # Score de mejora: positivo = peor, negativo = mejor
        score_mejora = (
            deltas["prediction_error_medio"]     * 2.0 +
            deltas["riesgo_desalineacion_medio"]  * 1.5 +
            deltas["workspace_overflow_rate"]    * 1.0 +
            deltas["costo_cognitivo_estimado"]   * 0.5
        )
        mejora = score_mejora < -0.02  # mejora real si el score baja > 0.02

        # Confianza: más turnos = más confianza
        confianza = float(np.clip(
            0.3 + (baseline.n_turnos / 50) * 0.5, 0.1, 0.85
        ))

        # Resumen
        cambios = []
        if abs(deltas["prediction_error_medio"]) > 0.02:
            dir_str = "↓" if deltas["prediction_error_medio"] < 0 else "↑"
            cambios.append(f"pred_error {dir_str}{abs(deltas['prediction_error_medio']):.2f}")
        if abs(deltas["workspace_overflow_rate"]) > 0.02:
            dir_str = "↓" if deltas["workspace_overflow_rate"] < 0 else "↑"
            cambios.append(f"overflow {dir_str}{abs(deltas['workspace_overflow_rate']):.2f}")
        if abs(deltas["riesgo_desalineacion_medio"]) > 0.02:
            dir_str = "↓" if deltas["riesgo_desalineacion_medio"] < 0 else "↑"
            cambios.append(f"riesgo {dir_str}{abs(deltas['riesgo_desalineacion_medio']):.2f}")
        if deltas["preguntas_generadas"] != 0:
            cambios.append(f"preguntas {deltas['preguntas_generadas']:+d}")

        resultado_str = "MEJORA" if mejora else "SIN MEJORA o EMPEORA"
        resumen = f"{resultado_str} | " + (", ".join(cambios) if cambios else "cambios mínimos")

        return mejora, confianza, resumen

    def _load_history(self, redis_client, module: str, n: int) -> list:
        try:
            raw_list = redis_client.lrange(
                f"alter:metrics:{module}:history", 0, n - 1
            )
            return [json.loads(r) for r in (raw_list or [])]
        except Exception:
            return []


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

class ExperimentRunner:
    """
    Orquesta experimentos sobre hipótesis replayables.

    Ciclo:
        1. Seleccionar hipótesis listas para experimentar
        2. Para cada hipótesis, crear un experimento por valor del rango
        3. Correr el ReplayEvaluator
        4. Guardar resultados
        5. Actualizar estado de hipótesis
    """

    def __init__(self, redis_client=None):
        self._redis   = redis_client
        self.evaluator = ReplayEvaluator()

    def run_pending(self, hypotheses: list) -> list:
        """
        Corre experimentos para todas las hipótesis replayables
        con score suficiente.
        Retorna lista de ExperimentResult.
        """
        resultados = []
        baseline   = self.evaluator.compute_baseline(self._redis)

        if baseline is None:
            # Sin trazas suficientes — usar baseline sintético mínimo
            baseline = ExperimentMetrics(
                prediction_error_medio    = 0.45,
                riesgo_desalineacion_medio = 0.30,
                council_activation_rate   = 0.35,
                workspace_overflow_rate   = 0.20,
                costo_cognitivo_estimado  = 0.28,
                preguntas_generadas       = 3,
                n_turnos                  = 10,
            )

        for hyp in hypotheses:
            if not hyp.puede_experimentar():
                continue
            if not hyp.rango_experimento or len(hyp.rango_experimento) < 2:
                continue

            result = self._run_one(hyp, baseline)
            if result:
                resultados.append(result)
                # Actualizar estado de hipótesis
                hyp.estado     = "en_experimento"
                hyp.updated_at = datetime.now().isoformat()

        self._save_results(resultados)
        return resultados

    def _run_one(
        self,
        hyp,              # ArchitectureHypothesis
        baseline: ExperimentMetrics,
    ) -> Optional[ExperimentResult]:
        """Corre un experimento para una hipótesis específica."""
        val_min, val_max = hyp.rango_experimento
        val_mid = (val_min + val_max) / 2

        # Intentar inferir valor baseline desde el nombre del parámetro
        val_baseline = self._infer_baseline_value(hyp.parametro_target, val_mid)

        # Estimar variante con valor medio del rango
        variante = self.evaluator.estimate_variante(
            baseline,
            hyp.parametro_target,
            val_baseline,
            val_mid,
        )

        mejora, confianza, resumen = self.evaluator.compare(
            baseline, variante, hyp.parametro_target
        )

        exp = Experiment(
            id             = str(uuid.uuid4())[:8],
            hypothesis_id  = hyp.id,
            parametro      = hyp.parametro_target,
            valor_baseline = val_baseline,
            valor_variante = val_mid,
            baseline_type  = "trace_replay" if baseline.n_turnos >= 5 else "summary_only",
            modo           = "replay",
            estado         = "completado",
            confidence     = confianza,
            completed_at   = datetime.now().isoformat(),
        )

        result = ExperimentResult(
            experiment_id  = exp.id,
            hypothesis_id  = hyp.id,
            parametro      = hyp.parametro_target,
            valor_baseline = val_baseline,
            valor_variante = val_mid,
            baseline       = baseline,
            variante       = variante,
            mejora         = mejora,
            confianza      = confianza,
            resumen        = resumen,
        )
        exp.resultado = result.to_dict()
        self._save_experiment(exp)
        return result

    def _infer_baseline_value(self, parametro: str, fallback: float) -> float:
        """Infiere el valor actual del parámetro desde la spec."""
        defaults = {
            "workspace.MAX_ITEMS":                    7.0,
            "simulator.OVERRIDE_THRESHOLD":           0.15,
            "policy.UMBRAL_RIESGO_DESALINEACION":     0.65,
            "memory.THRESHOLD_ERROR":                 0.60,
            "homeostasis.recovery_claridad":          0.10,
        }
        return defaults.get(parametro, fallback)

    def _save_experiment(self, exp: Experiment):
        if not self._redis:
            return
        try:
            self._redis.lpush(
                "alter:b5:experiments",
                json.dumps(exp.to_dict(), ensure_ascii=False)
            )
            self._redis.ltrim("alter:b5:experiments", 0, 49)
        except Exception:
            pass

    def _save_results(self, results: list):
        if not self._redis or not results:
            return
        try:
            for r in results:
                self._redis.lpush(
                    "alter:b5:experiments:results",
                    json.dumps(r.to_dict(), ensure_ascii=False)
                )
            self._redis.ltrim("alter:b5:experiments:results", 0, 49)
        except Exception:
            pass

    def snapshot_str(self, results: list) -> str:
        if not results:
            return "[EXPERIMENTOS] Sin resultados aún."
        mejoras = [r for r in results if r.mejora]
        lines   = [f"[EXPERIMENTOS] {len(results)} corridos | {len(mejoras)} mejoras detectadas"]
        for r in results[:5]:
            icono = "✓" if r.mejora else "✗"
            lines.append(
                f"  {icono} {r.parametro}: "
                f"{r.valor_baseline:.2f}→{r.valor_variante:.2f} "
                f"(conf:{r.confianza:.2f}) {r.resumen[:50]}"
            )
        return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    evaluator = ReplayEvaluator()
    runner    = ExperimentRunner(redis_client=None)

    # Test 1: compute_baseline sin Redis retorna None
    result = evaluator.compute_baseline(None)
    if result is not None:
        errors.append("FAIL: compute_baseline sin Redis debería retornar None")

    # Test 2: estimate_variante no explota
    baseline = ExperimentMetrics(
        prediction_error_medio=0.45, riesgo_desalineacion_medio=0.30,
        council_activation_rate=0.35, workspace_overflow_rate=0.20,
        costo_cognitivo_estimado=0.28, preguntas_generadas=3, n_turnos=20
    )
    variantes_params = [
        ("workspace.MAX_ITEMS", 7.0, 5.0),
        ("simulator.OVERRIDE_THRESHOLD", 0.15, 0.25),
        ("policy.UMBRAL_RIESGO_DESALINEACION", 0.65, 0.75),
        ("memory.THRESHOLD_ERROR", 0.60, 0.50),
        ("homeostasis.recovery_claridad", 0.10, 0.15),
    ]
    for param, val_b, val_v in variantes_params:
        try:
            v = evaluator.estimate_variante(baseline, param, val_b, val_v)
            if not (0.0 <= v.prediction_error_medio <= 1.0):
                errors.append(f"FAIL: {param} variante pred_error fuera de rango")
        except Exception as e:
            errors.append(f"FAIL: estimate_variante {param} explotó: {e}")

    # Test 3: MAX_ITEMS más pequeño reduce overflow
    v_menos = evaluator.estimate_variante(baseline, "workspace.MAX_ITEMS", 7.0, 5.0)
    v_mas   = evaluator.estimate_variante(baseline, "workspace.MAX_ITEMS", 7.0, 9.0)
    if v_menos.workspace_overflow_rate >= baseline.workspace_overflow_rate:
        errors.append("FAIL: menos MAX_ITEMS debería reducir overflow")
    if v_mas.workspace_overflow_rate <= baseline.workspace_overflow_rate:
        errors.append("FAIL: más MAX_ITEMS debería aumentar overflow")

    # Test 4: compare retorna tipos correctos
    mejora, conf, resumen = evaluator.compare(baseline, v_menos, "workspace.MAX_ITEMS")
    if not isinstance(mejora, bool):
        errors.append("FAIL: compare.mejora no es bool")
    if not (0.0 <= conf <= 1.0):
        errors.append(f"FAIL: compare.confianza fuera de rango: {conf}")
    if not isinstance(resumen, str):
        errors.append("FAIL: compare.resumen no es str")

    # Test 5: run_pending sin hipótesis replayables no explota
    from alter_architecture_hypotheses import ArchitectureHypothesis
    hyp_no_replay = ArchitectureHypothesis(
        id="t1", titulo="test", tipo="split", descripcion="test",
        evidencia=[], modulo="brain", impacto="alto", riesgo="alto",
        score=0.8, estado="propuesta", replayable=False,
    )
    try:
        results = runner.run_pending([hyp_no_replay])
        if results:
            errors.append("FAIL: run_pending con hipótesis no replayable no debería dar resultados")
    except Exception as e:
        errors.append(f"FAIL: run_pending explotó: {e}")

    # Test 6: run_pending con hipótesis replayable genera resultado
    hyp_replay = ArchitectureHypothesis(
        id="t2", titulo="test ws", tipo="parametro", descripcion="test",
        evidencia=["overflow alto"], modulo="workspace", impacto="medio", riesgo="bajo",
        score=0.60, estado="propuesta", replayable=True,
        parametro_target="workspace.MAX_ITEMS",
        rango_experimento=[5, 7],
    )
    results = runner.run_pending([hyp_replay])
    if not results:
        errors.append("FAIL: run_pending con hipótesis replayable no generó resultados")

    # Test 7: ExperimentResult round-trip
    if results:
        r = results[0]
        d = r.to_dict()
        if "experiment_id" not in d or "mejora" not in d:
            errors.append("FAIL: ExperimentResult.to_dict incompleto")

    # Test 8: snapshot_str no explota
    snap = runner.snapshot_str(results)
    if "[EXPERIMENTOS]" not in snap:
        errors.append("FAIL: snapshot_str malformado")

    return errors


if __name__ == "__main__":
    print("=== alter_experiments.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo Experiment Runner ===")
    from alter_architecture_hypotheses import ArchitectureHypothesis

    runner = ExperimentRunner(redis_client=None)

    hypotheses = [
        ArchitectureHypothesis(
            id="h1", titulo="Reducir MAX_ITEMS workspace",
            tipo="parametro", descripcion="Overflow frecuente",
            evidencia=["Workspace en overflow 38%"], modulo="workspace",
            impacto="medio", riesgo="bajo", score=0.55,
            estado="propuesta", replayable=True,
            parametro_target="workspace.MAX_ITEMS",
            rango_experimento=[5, 7],
        ),
        ArchitectureHypothesis(
            id="h2", titulo="Subir OVERRIDE_THRESHOLD simulator",
            tipo="parametro", descripcion="Override muy frecuente",
            evidencia=["Simulator override 65%"], modulo="simulator",
            impacto="bajo", riesgo="bajo", score=0.50,
            estado="propuesta", replayable=True,
            parametro_target="simulator.OVERRIDE_THRESHOLD",
            rango_experimento=[0.10, 0.30],
        ),
        ArchitectureHypothesis(
            id="h3", titulo="Dividir alter_brain.py",
            tipo="split", descripcion="2894 líneas",
            evidencia=["alter_brain.py tiene 2894 líneas"], modulo="brain",
            impacto="alto", riesgo="alto", score=0.70,
            estado="no_replayable_yet", replayable=False,
        ),
    ]

    results = runner.run_pending(hypotheses)
    print(runner.snapshot_str(results))
    print(f"\nDetalle de resultados:")
    for r in results:
        print(f"  {r.parametro}: {r.valor_baseline:.2f}→{r.valor_variante:.2f}")
        print(f"    Baseline: overflow={r.baseline.workspace_overflow_rate:.2f} "
              f"pred_err={r.baseline.prediction_error_medio:.2f}")
        print(f"    Variante: overflow={r.variante.workspace_overflow_rate:.2f} "
              f"pred_err={r.variante.prediction_error_medio:.2f}")
        print(f"    → {r.resumen}")
