"""
alter_simulator.py — Counterfactual Simulator de AlterB4 (Fase 1)

Antes de que el Policy Arbiter decida, ALTER evalúa 2-3 escenarios
alternativos y elige el que maximiza el efecto esperado.

Opción A — estimación heurística pura:
    Sin llamadas a Gemini. Reglas basadas en intención, homeostasis
    y workspace. Rápido, predecible, trazable.

Cuándo corre:
    - council_tension >= "media"
    - riesgo_desalineacion > 0.45
    - prediction_error_last > 0.50

No corre en cada turno — solo cuando la decisión es no trivial.

Escenarios evaluados:
    responder_directo  — respuesta directa a lo que pidió
    preguntar          — pedir clarificación primero
    reformular         — cambiar el enfoque

Score final:
    score = 0.30 * claridad + 0.35 * satisfaccion
          - 0.25 * riesgo - 0.10 * costo_cognitivo

Integración:
    Simulator.evaluate() → list[ScenarioScore]
    Si el escenario ganador difiere de la acción de Gemini
    y la diferencia de score > OVERRIDE_THRESHOLD → informa al Arbiter.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ============================================================
# CONSTANTES
# ============================================================

OVERRIDE_THRESHOLD    = 0.15   # diferencia mínima para recomendar override
RIESGO_TRIGGER        = 0.45   # riesgo de desalineación para activar
ERROR_TRIGGER         = 0.50   # prediction error para activar
COUNCIL_TENSIONS_TRIGGER = {"media", "alta"}

ESCENARIOS = ["responder_directo", "preguntar", "reformular"]

SCORE_WEIGHTS = {
    "claridad":    0.30,
    "satisfaccion": 0.35,
    "riesgo":     -0.25,
    "costo":      -0.10,
}


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class ScenarioScore:
    accion:                str
    claridad_esperada:     float
    satisfaccion_esperada: float
    riesgo_desalineacion:  float
    costo_cognitivo:       float
    score_final:           float
    razon:                 str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulatorResult:
    activated:          bool
    escenarios:         list   # list[ScenarioScore]
    ganador:            Optional[ScenarioScore]
    accion_gemini:      str
    recomienda_override: bool
    delta_score:        float  # score_ganador - score_gemini
    razon:              str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ============================================================
# COUNTERFACTUAL SIMULATOR
# ============================================================

class CounterfactualSimulator:
    """
    Evalúa escenarios alternativos antes de que el Arbiter decida.
    Estimación heurística — sin llamadas a Gemini.
    """

    def should_activate(
        self,
        council_tension:    str,
        riesgo_desalineacion: float,
        prediction_error:   float,
    ) -> bool:
        """
        Determina si el simulador debe correr este turno.
        Solo se activa cuando la decisión es no trivial.
        """
        return (
            council_tension in COUNCIL_TENSIONS_TRIGGER or
            riesgo_desalineacion > RIESGO_TRIGGER or
            prediction_error > ERROR_TRIGGER
        )

    def evaluate(
        self,
        accion_gemini:      str,
        texto_input:        str,
        workspace_snap:     Optional[dict] = None,
        homeostasis_snap:   Optional[dict] = None,
        predictive_state=   None,
        council_tension:    str = "ninguna",
    ) -> SimulatorResult:
        """
        Evalúa todos los escenarios y retorna el resultado.
        Si el ganador difiere de Gemini con delta > OVERRIDE_THRESHOLD,
        recomienda override al Arbiter.
        """
        ws   = workspace_snap   or {}
        hs   = homeostasis_snap or {}

        # Extraer señales del estado actual
        signals = self._extract_signals(
            texto_input, ws, hs, predictive_state, council_tension
        )

        # Evaluar cada escenario
        escenarios = [
            self._score_scenario(accion, signals)
            for accion in ESCENARIOS
        ]
        escenarios.sort(key=lambda s: s.score_final, reverse=True)
        ganador = escenarios[0]

        # Score del escenario que Gemini eligió
        score_gemini = next(
            (s.score_final for s in escenarios
             if s.accion == self._normalize_accion(accion_gemini)),
            ganador.score_final
        )
        delta = ganador.score_final - score_gemini
        recomienda_override = (
            delta > OVERRIDE_THRESHOLD and
            ganador.accion != self._normalize_accion(accion_gemini)
        )

        razon = (
            f"Ganador: {ganador.accion} (score:{ganador.score_final:.2f}) "
            f"vs Gemini: {accion_gemini} (score:{score_gemini:.2f}) "
            f"delta:{delta:+.2f}"
        )

        return SimulatorResult(
            activated           = True,
            escenarios          = escenarios,
            ganador             = ganador,
            accion_gemini       = accion_gemini,
            recomienda_override = recomienda_override,
            delta_score         = delta,
            razon               = razon,
        )

    # ----------------------------------------------------------
    # EXTRACCIÓN DE SEÑALES
    # ----------------------------------------------------------

    def _extract_signals(
        self,
        texto_input:      str,
        workspace_snap:   dict,
        homeostasis_snap: dict,
        predictive_state,
        council_tension:  str,
    ) -> dict:
        """
        Extrae señales del estado actual para alimentar los scores.
        """
        texto_lower = texto_input.lower()

        # Intención dominante
        intent = "desconocido"
        intent_conf = 0.5
        if predictive_state and predictive_state.intent_hypotheses:
            dom = predictive_state.dominant_hypothesis()
            if dom:
                intent = dom.label
                intent_conf = dom.confidence

        # Riesgo de desalineación
        riesgo = 0.3
        if predictive_state:
            riesgo = predictive_state.expected_effect.get(
                "riesgo_desalineacion", 0.3
            )

        # Estado de homeostasis
        energia   = homeostasis_snap.get("energia",   0.7)
        fatiga    = homeostasis_snap.get("fatiga",     0.2)
        claridad  = homeostasis_snap.get("claridad",  0.7)
        presion   = homeostasis_snap.get("presion",   0.3)

        # Workspace
        tiene_goal = bool(workspace_snap.get("dominant_goal"))
        tensiones  = len(workspace_snap.get("active_tensions", []))

        # Ambigüedad del input
        es_pregunta    = texto_input.strip().endswith("?")
        es_corto       = len(texto_input.split()) < 8
        tiene_imperativos = any(
            p in texto_lower for p in
            ["hacé", "implementá", "codeá", "armá", "creá", "escribí"]
        )

        return {
            "intent":           intent,
            "intent_conf":      intent_conf,
            "riesgo_base":      riesgo,
            "energia":          energia,
            "fatiga":           fatiga,
            "claridad":         claridad,
            "presion":          presion,
            "tiene_goal":       tiene_goal,
            "tensiones":        tensiones,
            "es_pregunta":      es_pregunta,
            "es_corto":         es_corto,
            "tiene_imperativos": tiene_imperativos,
            "council_tension":  council_tension,
        }

    # ----------------------------------------------------------
    # SCORING DE ESCENARIOS
    # ----------------------------------------------------------

    def _score_scenario(self, accion: str, signals: dict) -> ScenarioScore:
        """
        Estima claridad, satisfacción, riesgo y costo para una acción.
        Lógica heurística — Opción A.
        """
        intent   = signals["intent"]
        riesgo_b = signals["riesgo_base"]
        energia  = signals["energia"]
        fatiga   = signals["fatiga"]
        claridad = signals["claridad"]
        council  = signals["council_tension"]

        if accion == "responder_directo":
            return self._score_responder(signals)
        elif accion == "preguntar":
            return self._score_preguntar(signals)
        elif accion == "reformular":
            return self._score_reformular(signals)
        else:
            return ScenarioScore(
                accion=accion,
                claridad_esperada=0.5,
                satisfaccion_esperada=0.5,
                riesgo_desalineacion=0.5,
                costo_cognitivo=0.3,
                score_final=0.3,
                razon="acción desconocida"
            )

    def _score_responder(self, s: dict) -> ScenarioScore:
        intent = s["intent"]

        # Claridad base según intención
        claridad = 0.65
        if intent == "quiere_accion":
            claridad = 0.75  # acción concreta = clara
        elif intent == "quiere_cierre":
            claridad = 0.80
        elif intent == "quiere_exploracion":
            claridad = 0.55  # exploración = menos concreto

        # Satisfacción base
        satisfaccion = 0.65
        if intent == "quiere_accion" and s["tiene_imperativos"]:
            satisfaccion = 0.82
        elif intent == "quiere_validacion":
            satisfaccion = 0.75
        elif intent == "quiere_cierre":
            satisfaccion = 0.80
        elif intent == "quiere_analisis":
            satisfaccion = 0.70

        # Riesgo: se amplia si hay alta tensión del Council
        riesgo = s["riesgo_base"]
        if s["council_tension"] == "alta":
            riesgo = min(1.0, riesgo + 0.15)
        if s["intent_conf"] < 0.45:
            riesgo = min(1.0, riesgo + 0.10)

        # Costo: energía invertida en respuesta directa
        costo = 0.20 + 0.10 * s["fatiga"]

        score = self._calc_score(claridad, satisfaccion, riesgo, costo)
        razon = f"respuesta directa para {intent} (conf:{s['intent_conf']:.2f})"
        return ScenarioScore("responder_directo", claridad, satisfaccion,
                             riesgo, costo, score, razon)

    def _score_preguntar(self, s: dict) -> ScenarioScore:
        intent = s["intent"]

        # Preguntar es bueno cuando hay ambigüedad
        claridad     = 0.70  # la pregunta aclara el contexto
        satisfaccion = 0.55  # el usuario quería una respuesta, no una pregunta

        # Excepción: si hay exploración activa o ambigüedad real
        if intent == "quiere_exploracion" or s["es_corto"]:
            satisfaccion = 0.68
        if s["riesgo_base"] > 0.55:
            satisfaccion = 0.72  # preguntar es mejor que desalinearse

        # Riesgo bajo — preguntar difícilmente desalinea
        riesgo = max(0.10, s["riesgo_base"] * 0.4)

        # Costo moderado — interrumpe el flujo
        costo = 0.25 + 0.05 * s["presion"]

        score = self._calc_score(claridad, satisfaccion, riesgo, costo)
        razon = f"pregunta para reducir riesgo {s['riesgo_base']:.2f}"
        return ScenarioScore("preguntar", claridad, satisfaccion,
                             riesgo, costo, score, razon)

    def _score_reformular(self, s: dict) -> ScenarioScore:
        intent = s["intent"]

        # Reformular es bueno cuando el enfoque actual tiene tensión
        claridad     = 0.68
        satisfaccion = 0.62

        if s["council_tension"] in ("media", "alta"):
            claridad     = 0.75
            satisfaccion = 0.70  # cambiar enfoque puede resolver la tensión

        if intent == "quiere_correccion":
            claridad     = 0.78
            satisfaccion = 0.75

        # Riesgo: reformular puede confundir si la intención era clara
        riesgo = s["riesgo_base"] * 0.7
        if s["intent_conf"] > 0.70:
            riesgo = min(riesgo + 0.10, 0.8)  # intención clara → reformular es más riesgoso

        # Costo mayor — requiere más procesamiento
        costo = 0.30 + 0.10 * (1.0 - s["claridad"])

        score = self._calc_score(claridad, satisfaccion, riesgo, costo)
        razon = f"reformular para tensión {s['council_tension']}"
        return ScenarioScore("reformular", claridad, satisfaccion,
                             riesgo, costo, score, razon)

    @staticmethod
    def _calc_score(claridad, satisfaccion, riesgo, costo) -> float:
        score = (
            SCORE_WEIGHTS["claridad"]    * claridad +
            SCORE_WEIGHTS["satisfaccion"] * satisfaccion +
            SCORE_WEIGHTS["riesgo"]      * riesgo +
            SCORE_WEIGHTS["costo"]       * costo
        )
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _normalize_accion(accion: str) -> str:
        """Mapea acciones de Gemini a nombres de escenarios."""
        mapping = {
            "responder":       "responder_directo",
            "registrar":       "responder_directo",
            "ignorar":         "responder_directo",
            "preguntar":       "preguntar",
            "reformular":      "reformular",
            "usar_herramienta": "responder_directo",
            "diferir":         "preguntar",
        }
        return mapping.get(accion, "responder_directo")


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []
    sim = CounterfactualSimulator()

    # Test 1: should_activate — condiciones correctas
    if not sim.should_activate("media", 0.3, 0.3):
        errors.append("FAIL: council tension media debería activar")
    if not sim.should_activate("ninguna", 0.60, 0.3):
        errors.append("FAIL: riesgo alto debería activar")
    if not sim.should_activate("ninguna", 0.3, 0.70):
        errors.append("FAIL: prediction error alto debería activar")
    if sim.should_activate("ninguna", 0.3, 0.3):
        errors.append("FAIL: condiciones bajas no deberían activar")

    # Test 2: evaluate retorna 3 escenarios
    result = sim.evaluate(
        accion_gemini="responder",
        texto_input="implementá el workspace",
        council_tension="media",
    )
    if not result.activated:
        errors.append("FAIL: debería estar activado")
    if len(result.escenarios) != 3:
        errors.append(f"FAIL: esperaba 3 escenarios, got {len(result.escenarios)}")

    # Test 3: scores dentro de rango
    for esc in result.escenarios:
        if not (0.0 <= esc.score_final <= 1.0):
            errors.append(f"FAIL: score fuera de rango: {esc.accion}={esc.score_final}")

    # Test 4: ganador tiene el score más alto
    scores = [e.score_final for e in result.escenarios]
    if result.ganador.score_final != max(scores):
        errors.append("FAIL: ganador no tiene el score más alto")

    # Test 5: riesgo alto favorece preguntar
    result_riesgo = sim.evaluate(
        accion_gemini="responder",
        texto_input="qué pensás",
        predictive_state=type("P", (), {
            "expected_effect": {"riesgo_desalineacion": 0.80},
            "intent_hypotheses": [],
            "prediction_error_last": 0.6,
            "dominant_hypothesis": lambda self: None
        })(),
        council_tension="alta",
    )
    preguntar_score = next(
        e.score_final for e in result_riesgo.escenarios
        if e.accion == "preguntar"
    )
    responder_score = next(
        e.score_final for e in result_riesgo.escenarios
        if e.accion == "responder_directo"
    )
    if preguntar_score <= responder_score:
        errors.append(
            f"FAIL: con riesgo alto, preguntar ({preguntar_score:.2f}) "
            f"debería superar responder ({responder_score:.2f})"
        )

    # Test 6: recomienda_override solo si delta > threshold
    result_sin_override = sim.evaluate(
        accion_gemini="responder",
        texto_input="dale",
        council_tension="ninguna",
    )
    # Con condiciones neutras el ganador debería ser responder y no haber override
    # (no siempre — depende de scores, solo verificamos que el flag es coherente)
    if result_sin_override.recomienda_override and result_sin_override.delta_score <= OVERRIDE_THRESHOLD:
        errors.append("FAIL: recomienda_override con delta <= threshold")

    # Test 7: SimulatorResult.to_dict no explota
    d = result.to_dict()
    if "ganador" not in d or "recomienda_override" not in d:
        errors.append("FAIL: SimulatorResult.to_dict incompleto")

    return errors


if __name__ == "__main__":
    print("=== alter_simulator.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo Simulator ===")
    sim = CounterfactualSimulator()

    casos = [
        {
            "nombre": "Acción concreta con bajo riesgo",
            "kwargs": {
                "accion_gemini": "responder",
                "texto_input": "implementá el workspace ahora",
                "council_tension": "media",
            }
        },
        {
            "nombre": "Alta ambigüedad + riesgo alto",
            "kwargs": {
                "accion_gemini": "responder",
                "texto_input": "qué pensás de todo esto",
                "homeostasis_snap": {"energia": 0.4, "fatiga": 0.7,
                                     "claridad": 0.4, "presion": 0.6},
                "predictive_state": type("P", (), {
                    "expected_effect": {"riesgo_desalineacion": 0.75},
                    "intent_hypotheses": [],
                    "prediction_error_last": 0.65,
                    "dominant_hypothesis": lambda self: None
                })(),
                "council_tension": "alta",
            }
        },
    ]

    for caso in casos:
        result = sim.evaluate(**caso["kwargs"])
        print(f"\n{caso['nombre']}:")
        for esc in result.escenarios:
            marca = "★" if esc == result.ganador else " "
            print(f"  {marca} {esc.accion:<20} score:{esc.score_final:.2f} "
                  f"| {esc.razon[:50]}")
        print(f"  Override: {result.recomienda_override} "
              f"(delta:{result.delta_score:+.2f})")
