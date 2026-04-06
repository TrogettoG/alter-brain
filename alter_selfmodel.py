"""
alter_selfmodel.py — Self-Model de AlterB4 (Fase 2)

ALTER tiene autobiografía (quién es) y ahora tiene un modelo
operativo de sí misma (cómo rinde).

Lee alter:metrics:*:history — los datos ya están desde Fase 1.
El Self-Model es una capa de análisis sobre las métricas acumuladas.

Qué trackea:
    module_performance   — score de rendimiento por módulo
    intent_performance   — éxito por tipo de intención del usuario
    failure_patterns     — condiciones donde ALTER falla recurrentemente
    strength_contexts    — condiciones donde ALTER rinde bien
    calibrated_confidence— confianza ajustada por dominio/contexto

Output clave:
    get_confidence_for(context) — confianza calibrada para el Arbiter
    snapshot_str()              — resumen para KAIROS y Architecture Auditor

Redis keys:
    alter:selfmodel:state    — estado serializado
    alter:selfmodel:updated  — timestamp de última actualización
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
class IntentPerformance:
    intent:      str
    success_rate: float  # 0..1
    n_samples:   int
    error_medio: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FailurePattern:
    modulo:     str
    condicion:  str   # descripción de la condición
    frecuencia: int   # cuántas veces se detectó
    ultimo:     str   # timestamp

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrengthContext:
    descripcion:  str
    score_medio:  float
    n_ocurrencias: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelfModel:
    # Rendimiento por módulo (0..1 — mayor es mejor)
    module_performance: dict = field(default_factory=lambda: {
        "homeostasis":   0.5,
        "workspace":     0.5,
        "predictive":    0.5,
        "procedural":    0.5,
        "policy":        0.5,
        "simulator":     0.5,
        "consolidation": 0.5,
    })

    # Rendimiento por tipo de intención
    intent_performance: list = field(default_factory=list)

    # Patrones de falla recurrentes
    failure_patterns: list = field(default_factory=list)

    # Contextos de fortaleza
    strength_contexts: list = field(default_factory=list)

    # Confianza calibrada por dominio
    calibrated_confidence: dict = field(default_factory=lambda: {
        "quiere_accion":      0.65,
        "quiere_analisis":    0.65,
        "quiere_validacion":  0.65,
        "quiere_exploracion": 0.65,
        "quiere_cierre":      0.65,
        "quiere_correccion":  0.65,
    })

    # Metadata
    sessions_analyzed: int = 0
    last_updated:      str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        intent_perf = [IntentPerformance(**i) for i in d.get("intent_performance", [])]
        failure_p   = [FailurePattern(**f)    for f in d.get("failure_patterns", [])]
        strength_c  = [StrengthContext(**s)   for s in d.get("strength_contexts", [])]
        return cls(
            module_performance    = d.get("module_performance", {}),
            intent_performance    = intent_perf,
            failure_patterns      = failure_p,
            strength_contexts     = strength_c,
            calibrated_confidence = d.get("calibrated_confidence", {}),
            sessions_analyzed     = d.get("sessions_analyzed", 0),
            last_updated          = d.get("last_updated", datetime.now().isoformat()),
        )

    def get_confidence_for(self, intent: str) -> float:
        """
        Retorna la confianza calibrada para un tipo de intención.
        Usada por el Arbiter para ajustar umbrales de decisión.
        """
        return float(self.calibrated_confidence.get(intent, 0.65))

    def get_module_score(self, module: str) -> float:
        return float(self.module_performance.get(module, 0.5))

    def weakest_module(self) -> str:
        if not self.module_performance:
            return "desconocido"
        return min(self.module_performance, key=self.module_performance.get)

    def strongest_module(self) -> str:
        if not self.module_performance:
            return "desconocido"
        return max(self.module_performance, key=self.module_performance.get)


# ============================================================
# SELF MODEL BUILDER
# ============================================================

class SelfModelBuilder:
    """
    Construye y actualiza el SelfModel desde las métricas acumuladas.
    Lee alter:metrics:*:history de Redis.
    """

    MIN_SAMPLES = 3   # mínimo de muestras para considerar un patrón

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def build(self, current_model: Optional[SelfModel] = None) -> SelfModel:
        """
        Construye un SelfModel actualizado desde las métricas disponibles.
        Si hay un modelo previo, hace merge en lugar de reemplazar.
        """
        model = current_model or SelfModel()

        # Leer métricas de todos los módulos
        raw_metrics = self._load_all_metrics()
        if not raw_metrics:
            return model

        # Actualizar rendimiento por módulo
        self._update_module_performance(model, raw_metrics)

        # Actualizar rendimiento por intención
        self._update_intent_performance(model, raw_metrics)

        # Detectar patrones de falla
        self._detect_failure_patterns(model, raw_metrics)

        # Detectar contextos de fortaleza
        self._detect_strength_contexts(model, raw_metrics)

        # Calibrar confianza por intención
        self._calibrate_confidence(model)

        model.sessions_analyzed += 1
        model.last_updated = datetime.now().isoformat()

        # Persistir
        self._save(model)

        return model

    # ----------------------------------------------------------
    # ANÁLISIS DE MÉTRICAS
    # ----------------------------------------------------------

    def _update_module_performance(self, model: SelfModel, raw: dict):
        """
        Calcula score de rendimiento por módulo basado en sus métricas.
        Score = 1 - tasa_de_warnings
        """
        for module, entries in raw.items():
            if not entries:
                continue
            total_warnings = sum(len(e.get("warnings", [])) for e in entries)
            total_turns    = len(entries)
            warning_rate   = total_warnings / max(total_turns, 1)
            # Score inverso a la tasa de warnings, suavizado
            score = float(np.clip(1.0 - warning_rate * 0.5, 0.2, 1.0))

            # Ajuste específico por módulo
            if module == "predictive":
                errors = [e["metrics"].get("error_ultimo", 0.5)
                          for e in entries if "metrics" in e]
                if errors:
                    error_medio = sum(errors) / len(errors)
                    score = float(np.clip(1.0 - error_medio, 0.2, 1.0))

            elif module == "workspace":
                overflows = [e["metrics"].get("overflow", False)
                             for e in entries if "metrics" in e]
                if overflows:
                    overflow_rate = sum(overflows) / len(overflows)
                    score = float(np.clip(1.0 - overflow_rate * 0.6, 0.2, 1.0))

            elif module == "procedural":
                sr_vals = [e["metrics"].get("sr_medio", 0.5)
                           for e in entries if "metrics" in e]
                if sr_vals:
                    score = float(np.clip(
                        sum(sr_vals) / len(sr_vals), 0.2, 1.0
                    ))

            # Media móvil con el score anterior
            prev = model.module_performance.get(module, 0.5)
            model.module_performance[module] = float(
                0.6 * prev + 0.4 * score
            )

    def _update_intent_performance(self, model: SelfModel, raw: dict):
        """
        Calcula éxito por tipo de intención desde métricas predictive.
        """
        pred_entries = raw.get("predictive", [])
        if not pred_entries:
            return

        intent_errors: dict[str, list[float]] = {}
        for entry in pred_entries:
            m = entry.get("metrics", {})
            intent = m.get("intent_dominante", "")
            error  = m.get("error_ultimo", 0.5)
            if intent and intent != "ninguno":
                intent_errors.setdefault(intent, []).append(error)

        new_perf = []
        for intent, errors in intent_errors.items():
            if len(errors) < self.MIN_SAMPLES:
                continue
            error_medio  = sum(errors) / len(errors)
            success_rate = float(np.clip(1.0 - error_medio, 0.0, 1.0))
            new_perf.append(IntentPerformance(
                intent       = intent,
                success_rate = success_rate,
                n_samples    = len(errors),
                error_medio  = error_medio,
            ))

        model.intent_performance = new_perf

    def _detect_failure_patterns(self, model: SelfModel, raw: dict):
        """
        Detecta condiciones donde ALTER falla recurrentemente.
        Un patrón = warning que aparece >= MIN_SAMPLES veces.
        """
        warning_count: dict[tuple, int] = {}
        for module, entries in raw.items():
            for entry in entries:
                for w in entry.get("warnings", []):
                    key = (module, w)
                    warning_count[key] = warning_count.get(key, 0) + 1

        failures = []
        for (modulo, condicion), count in warning_count.items():
            if count >= self.MIN_SAMPLES:
                failures.append(FailurePattern(
                    modulo    = modulo,
                    condicion = condicion,
                    frecuencia= count,
                    ultimo    = datetime.now().strftime("%Y-%m-%d %H:%M"),
                ))

        # Mantener solo los más frecuentes
        failures.sort(key=lambda f: f.frecuencia, reverse=True)
        model.failure_patterns = failures[:10]

    def _detect_strength_contexts(self, model: SelfModel, raw: dict):
        """
        Detecta condiciones donde ALTER rinde bien.
        """
        pred_entries = raw.get("predictive", [])
        strengths = []

        # Contextos de bajo error por intención
        intent_low_error: dict[str, list[float]] = {}
        for entry in pred_entries:
            m = entry.get("metrics", {})
            intent = m.get("intent_dominante", "")
            error  = m.get("error_ultimo", 0.5)
            if intent and error < 0.3:
                intent_low_error.setdefault(intent, []).append(error)

        for intent, errors in intent_low_error.items():
            if len(errors) >= self.MIN_SAMPLES:
                score = float(1.0 - sum(errors) / len(errors))
                strengths.append(StrengthContext(
                    descripcion  = f"Alta precisión en {intent}",
                    score_medio  = score,
                    n_ocurrencias= len(errors),
                ))

        strengths.sort(key=lambda s: s.score_medio, reverse=True)
        model.strength_contexts = strengths[:5]

    def _calibrate_confidence(self, model: SelfModel):
        """
        Ajusta la confianza calibrada por intención
        basándose en el rendimiento histórico.
        """
        for ip in model.intent_performance:
            # Confianza calibrada = success_rate ajustado hacia 0.65 base
            calibrated = 0.4 * 0.65 + 0.6 * ip.success_rate
            model.calibrated_confidence[ip.intent] = float(
                np.clip(calibrated, 0.30, 0.90)
            )

    # ----------------------------------------------------------
    # PERSISTENCIA
    # ----------------------------------------------------------

    def _load_all_metrics(self) -> dict:
        """Lee historial de métricas de todos los módulos desde Redis."""
        if not self._redis:
            return {}
        modules = [
            "homeostasis", "workspace", "predictive",
            "procedural", "policy", "simulator", "consolidation"
        ]
        result = {}
        for module in modules:
            try:
                raw_list = self._redis.lrange(
                    f"alter:metrics:{module}:history", 0, 49
                )
                if raw_list:
                    result[module] = [json.loads(r) for r in raw_list]
            except Exception:
                pass
        return result

    def _save(self, model: SelfModel):
        if not self._redis:
            return
        try:
            self._redis.set(
                "alter:selfmodel:state",
                json.dumps(model.to_dict(), ensure_ascii=False)
            )
            self._redis.set(
                "alter:selfmodel:updated",
                datetime.now().isoformat()
            )
        except Exception:
            pass

    def load(self) -> Optional[SelfModel]:
        """Carga el SelfModel desde Redis."""
        if not self._redis:
            return None
        try:
            raw = self._redis.get("alter:selfmodel:state")
            if raw:
                return SelfModel.from_dict(json.loads(raw))
        except Exception:
            pass
        return None


# ============================================================
# SNAPSHOT Y UTILIDADES
# ============================================================

def selfmodel_snapshot_str(model: SelfModel) -> str:
    """Resumen legible para KAIROS y el Architecture Auditor."""
    lines = ["[SELF-MODEL]"]

    # Rendimiento por módulo
    perf_sorted = sorted(
        model.module_performance.items(), key=lambda x: x[1]
    )
    lines.append("  Módulos (menor → mayor rendimiento):")
    for mod, score in perf_sorted:
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        lines.append(f"    {mod:<15} {bar} {score:.2f}")

    # Intención más difícil y más fácil
    if model.intent_performance:
        worst  = min(model.intent_performance, key=lambda i: i.success_rate)
        best   = max(model.intent_performance, key=lambda i: i.success_rate)
        lines.append(f"  Intención más difícil: {worst.intent} "
                     f"(sr:{worst.success_rate:.2f})")
        lines.append(f"  Intención más fácil:   {best.intent} "
                     f"(sr:{best.success_rate:.2f})")

    # Patrones de falla top
    if model.failure_patterns:
        top = model.failure_patterns[:3]
        lines.append("  Fallas recurrentes:")
        for f in top:
            lines.append(f"    [{f.modulo}] {f.condicion} x{f.frecuencia}")

    # Fortalezas
    if model.strength_contexts:
        lines.append("  Fortalezas:")
        for s in model.strength_contexts[:2]:
            lines.append(f"    {s.descripcion} (score:{s.score_medio:.2f})")

    return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    # Test 1: SelfModel inicializa con valores por defecto correctos
    model = SelfModel()
    if not model.module_performance:
        errors.append("FAIL: module_performance vacío en init")
    if model.get_confidence_for("quiere_accion") != 0.65:
        errors.append("FAIL: confianza base debería ser 0.65")

    # Test 2: from_dict round-trip
    model.failure_patterns = [
        FailurePattern("predictive", "error_alto", 5, "2026-04-01")
    ]
    model.strength_contexts = [
        StrengthContext("Alta precisión en cierre", 0.85, 8)
    ]
    d = model.to_dict()
    model2 = SelfModel.from_dict(d)
    if len(model2.failure_patterns) != 1:
        errors.append("FAIL: failure_patterns no sobrevivió round-trip")
    if len(model2.strength_contexts) != 1:
        errors.append("FAIL: strength_contexts no sobrevivió round-trip")

    # Test 3: get_confidence_for con intención desconocida retorna default
    conf = model.get_confidence_for("intención_rara")
    if conf != 0.65:
        errors.append(f"FAIL: confianza para intent desconocido debería ser 0.65, got {conf}")

    # Test 4: weakest/strongest módulo funciona
    model.module_performance = {
        "homeostasis": 0.9, "workspace": 0.3, "predictive": 0.6
    }
    if model.weakest_module() != "workspace":
        errors.append("FAIL: weakest_module debería ser workspace")
    if model.strongest_module() != "homeostasis":
        errors.append("FAIL: strongest_module debería ser homeostasis")

    # Test 5: SelfModelBuilder sin Redis no explota
    builder = SelfModelBuilder(redis_client=None)
    model3 = builder.build()
    if model3 is None:
        errors.append("FAIL: build() retornó None sin Redis")

    # Test 6: _calibrate_confidence ajusta correctamente
    model4 = SelfModel()
    model4.intent_performance = [
        IntentPerformance("quiere_accion", success_rate=0.9,
                          n_samples=10, error_medio=0.1)
    ]
    builder._calibrate_confidence(model4)
    conf_accion = model4.calibrated_confidence.get("quiere_accion", 0.65)
    if conf_accion <= 0.65:
        errors.append(
            f"FAIL: alta performance debería subir confianza, got {conf_accion:.2f}"
        )

    # Test 7: snapshot_str no explota
    snap = selfmodel_snapshot_str(model)
    if "[SELF-MODEL]" not in snap:
        errors.append("FAIL: snapshot_str malformado")

    return errors


if __name__ == "__main__":
    print("=== alter_selfmodel.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo SelfModel ===")
    model = SelfModel()
    model.module_performance = {
        "homeostasis": 0.82, "workspace": 0.61, "predictive": 0.55,
        "procedural": 0.70, "policy": 0.78, "simulator": 0.65,
        "consolidation": 0.80
    }
    model.intent_performance = [
        IntentPerformance("quiere_accion",    0.82, 15, 0.18),
        IntentPerformance("quiere_analisis",  0.70, 12, 0.30),
        IntentPerformance("quiere_validacion",0.88, 8,  0.12),
        IntentPerformance("quiere_cierre",    0.91, 6,  0.09),
        IntentPerformance("quiere_exploracion",0.58, 10, 0.42),
    ]
    model.failure_patterns = [
        FailurePattern("predictive", "error_prediccion_alto", 8, "2026-04-06"),
        FailurePattern("workspace",  "sin_goal_dominante",    5, "2026-04-05"),
    ]
    model.strength_contexts = [
        StrengthContext("Alta precisión en quiere_cierre", 0.91, 6),
        StrengthContext("Alta precisión en quiere_validacion", 0.88, 8),
    ]
    print(selfmodel_snapshot_str(model))
    print(f"\nConfianza para quiere_exploracion: "
          f"{model.get_confidence_for('quiere_exploracion'):.2f}")
