"""
alter_predictive.py — Predictive Model de AlterB3 (Fase 2)

ALTER hoy procesa lo que entra. Este módulo hace que anticipe antes de responder.

Qué hace:
    - Infiere hipótesis sobre la intención real del usuario
    - Predice el efecto esperado de la respuesta de ALTER
    - Calcula el error de predicción comparando turno N vs turno N+1
    - El error de predicción es la señal de aprendizaje

Diferencia clave con la Fase 1:
    Homeostasis y Workspace son estado presente.
    El Predictive Model opera en el eje temporal —
    guarda lo que creyó y lo corrige cuando llega la evidencia.

Redis keys propias (no toca keys existentes):
    alter:predictive:state   — último PredictiveState serializado
    alter:predictive:errors  — historial de errores (últimos 20)

AlterB3 Fase 2 — integración con workspace:
    Exporta candidatos user_hypothesis, internal_tension, constraint
    según el estado predictivo actual.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class UserIntentHypothesis:
    label:      str    # descripción de la intención
    confidence: float  # 0..1
    source:     str    # cómo se infirió: "lexical" | "context" | "history" | "prior"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictiveState:
    # Hipótesis sobre qué quiere el usuario — orden por confianza desc
    intent_hypotheses: list[UserIntentHypothesis] = field(default_factory=list)

    # Efecto esperado de la respuesta de ALTER
    expected_effect: dict = field(default_factory=lambda: {
        "claridad":            0.5,
        "satisfaccion":        0.5,
        "riesgo_desalineacion": 0.3,
    })

    # Error del turno anterior (0 = perfecto, 1 = error total)
    prediction_error_last: float = 0.0

    # Historial de últimos 20 errores — detecta deriva
    error_history: list[float] = field(default_factory=list)

    # Confianza global del modelo (media móvil de 1 - error)
    model_confidence: float = 0.5

    # Predicción guardada del turno N — para comparar en turno N+1
    # Formato: {"dominant_label": str, "dominant_confidence": float, "turn": int}
    pending_prediction: Optional[dict] = None

    # Contador de turnos
    turn_count: int = 0

    timestamp: float = field(default_factory=time.time)

    def dominant_hypothesis(self) -> Optional[UserIntentHypothesis]:
        if not self.intent_hypotheses:
            return None
        return max(self.intent_hypotheses, key=lambda h: h.confidence)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PredictiveState":
        hyps = [UserIntentHypothesis(**h) for h in d.get("intent_hypotheses", [])]
        return cls(
            intent_hypotheses    = hyps,
            expected_effect      = d.get("expected_effect", {}),
            prediction_error_last= d.get("prediction_error_last", 0.0),
            error_history        = d.get("error_history", []),
            model_confidence     = d.get("model_confidence", 0.5),
            pending_prediction   = d.get("pending_prediction"),
            turn_count           = d.get("turn_count", 0),
            timestamp            = d.get("timestamp", time.time()),
        )


# ============================================================
# SEÑALES LÉXICAS POR INTENCIÓN
# ============================================================

INTENT_SIGNALS: dict[str, list[str]] = {
    "quiere_analisis": [
        "por qué", "cómo funciona", "cómo se", "explicá", "explicame",
        "qué es", "qué significa", "entiendo que", "no entiendo",
        "cuál es la diferencia", "qué tiene que ver",
    ],
    "quiere_accion": [
        "hacé", "implementá", "codeá", "armá", "creá", "escribí",
        "agregá", "modificá", "arreglá", "actualizá", "subí",
        "mandame", "generá", "ejecutá", "corrí",
    ],
    "quiere_validacion": [
        "qué te parece", "tiene sentido", "está bien", "es correcto",
        "lo ves bien", "te cierra", "te copa", "me confirmas",
        "verdad?", "no?", "cierto?", "opinión",
    ],
    "quiere_exploracion": [
        "qué pensás", "cómo lo ves", "qué harías", "si pudieras",
        "imaginate", "suponete", "qué pasaría", "y si",
        "hablemos de", "contame sobre", "qué sabés de",
    ],
    "quiere_cierre": [
        "entonces", "resumiendo", "en conclusión", "para cerrar",
        "queda claro", "listo", "perfecto", "dale", "bueno",
        "entendido", "ok", "listo",
    ],
    "quiere_correccion": [
        "no", "eso no es", "no es así", "me parece que", "creo que estás",
        "te equivocás", "no coincido", "no estoy de acuerdo",
        "pero", "aunque", "sin embargo",
    ],
    "quiere_conexion": [
        "cómo estás", "qué onda", "contame", "cómo vas",
        "te acordás", "hablamos de", "la otra vez",
    ],
}


# ============================================================
# INFERENCIA DE INTENCIÓN
# ============================================================

def infer_intent(
    texto: str,
    historial: list[tuple[str, str]],
    prior_state: Optional[PredictiveState] = None,
) -> list[UserIntentHypothesis]:
    """
    Infiere hipótesis de intención del usuario.
    Combina señales léxicas, contexto del historial y prior del turno anterior.

    Retorna lista de hipótesis ordenadas por confianza desc, normalizada.
    """
    texto_lower = texto.lower()
    scores: dict[str, float] = {label: 0.0 for label in INTENT_SIGNALS}

    # 1. Señales léxicas del texto actual
    for label, signals in INTENT_SIGNALS.items():
        hits = sum(1 for s in signals if s in texto_lower)
        if hits > 0:
            scores[label] += min(0.6, 0.2 * hits)

    # 2. Señales estructurales
    if texto.strip().endswith("?"):
        scores["quiere_analisis"]    += 0.15
        scores["quiere_validacion"]  += 0.10
    if len(texto.split()) <= 5:
        scores["quiere_cierre"]      += 0.10
        scores["quiere_conexion"]    += 0.08
    if len(texto.split()) > 30:
        scores["quiere_exploracion"] += 0.12
        scores["quiere_analisis"]    += 0.08

    # 3. Prior — continuidad del turno anterior
    if prior_state and prior_state.dominant_hypothesis():
        dominant = prior_state.dominant_hypothesis()
        # Si el prior tenía alta confianza, empuja levemente la misma intención
        if dominant.confidence > 0.6:
            scores[dominant.label] = scores.get(dominant.label, 0.0) + 0.10

    # 4. Contexto del historial reciente — último turno del usuario
    if historial and len(historial) >= 2:
        ultimo_user = next(
            (txt for rol, txt in reversed(historial) if rol != "ALTER"),
            ""
        )
        if ultimo_user:
            ul = ultimo_user.lower()
            for label, signals in INTENT_SIGNALS.items():
                hits = sum(1 for s in signals if s in ul)
                if hits > 0:
                    scores[label] += min(0.15, 0.05 * hits)

    # Filtrar scores > 0, normalizar, construir hipótesis
    activos = {k: v for k, v in scores.items() if v > 0.0}
    if not activos:
        # Sin señales — hipótesis uniforme
        return [
            UserIntentHypothesis("quiere_exploracion", 0.4, "prior"),
            UserIntentHypothesis("quiere_validacion",  0.3, "prior"),
            UserIntentHypothesis("quiere_analisis",    0.3, "prior"),
        ]

    total = sum(activos.values())
    hipotesis = [
        UserIntentHypothesis(
            label      = label,
            confidence = float(np.clip(score / total, 0.0, 1.0)),
            source     = "lexical+context"
        )
        for label, score in sorted(activos.items(), key=lambda x: x[1], reverse=True)
    ]

    return hipotesis[:5]  # máximo 5 hipótesis


# ============================================================
# PREDICCIÓN DE EFECTO
# ============================================================

def predict_effect(
    response_candidate: str,
    intent_hypotheses: list[UserIntentHypothesis],
    homeostasis_snap: Optional[dict] = None,
) -> dict:
    """
    Predice el efecto esperado de una respuesta candidata.

    Variables de salida:
        claridad            — ¿el usuario va a entender mejor?
        satisfaccion        — ¿va a sentir que fue útil?
        riesgo_desalineacion — ¿hay chance de que no conecte?

    Lógica:
        - Si la respuesta es larga y la intención es cierre → alto riesgo
        - Si la respuesta responde directamente a la hipótesis dominante → alta satisfacción
        - Si hay tensión interna alta en homeostasis → mayor riesgo de desalineación
    """
    dominant = max(intent_hypotheses, key=lambda h: h.confidence) if intent_hypotheses else None
    hs = homeostasis_snap or {}

    resp_words  = len(response_candidate.split())
    has_question = "?" in response_candidate

    # Claridad — respuestas cortas y directas son más claras
    if resp_words < 30:
        claridad = 0.75
    elif resp_words < 80:
        claridad = 0.65
    else:
        claridad = 0.50

    # Satisfacción — depende de alineación con intención dominante
    satisfaccion = 0.55  # base
    if dominant:
        if dominant.label == "quiere_accion" and resp_words > 20:
            satisfaccion = 0.80  # acción concreta ejecutada
        elif dominant.label == "quiere_analisis" and resp_words > 40:
            satisfaccion = 0.75  # análisis desarrollado
        elif dominant.label == "quiere_validacion" and resp_words < 50:
            satisfaccion = 0.78  # validación concisa
        elif dominant.label == "quiere_cierre" and resp_words < 30:
            satisfaccion = 0.82  # cierre limpio
        elif dominant.label == "quiere_exploracion":
            satisfaccion = 0.65 + (0.1 if has_question else 0.0)

    # Riesgo de desalineación
    riesgo = 0.20  # base baja
    if dominant and dominant.confidence < 0.45:
        riesgo += 0.15  # intención incierta
    if hs.get("tension_interna", 0) > 0.5:
        riesgo += 0.10  # estado interno tenso
    if dominant and dominant.label == "quiere_cierre" and resp_words > 60:
        riesgo += 0.20  # respuesta larga cuando quería cerrar
    if dominant and dominant.label == "quiere_accion" and resp_words < 15:
        riesgo += 0.15  # respuesta muy corta cuando quería acción

    return {
        "claridad":            float(np.clip(claridad,     0.0, 1.0)),
        "satisfaccion":        float(np.clip(satisfaccion, 0.0, 1.0)),
        "riesgo_desalineacion": float(np.clip(riesgo,      0.0, 1.0)),
    }


# ============================================================
# CÁLCULO DE ERROR DE PREDICCIÓN
# ============================================================

def compute_prediction_error(
    pending: dict,
    nuevo_texto: str,
    historial: list[tuple[str, str]],
) -> float:
    """
    Calcula el error de predicción comparando:
        - Lo que el modelo predijo en el turno N (pending_prediction)
        - Lo que el usuario hizo en el turno N+1 (nuevo_texto)

    Error = 1 - confianza_de_la_hipotesis_correcta

    Si predijo quiere_analisis con confianza 0.8 y el usuario
    efectivamente preguntó más → error = 1 - 0.8 = 0.20
    Si predijo quiere_cierre y el usuario abrió tema nuevo → error alto
    """
    if not pending:
        return 0.0

    predicted_label      = pending.get("dominant_label", "")
    predicted_confidence = float(pending.get("dominant_confidence", 0.5))

    # Inferir intención real del nuevo texto
    hipotesis_reales = infer_intent(nuevo_texto, historial)
    dominant_real = hipotesis_reales[0] if hipotesis_reales else None

    if not dominant_real:
        return 0.5

    if dominant_real.label == predicted_label:
        # Predijo bien — error proporcional a la incertidumbre
        error = 1.0 - predicted_confidence
    else:
        # Predijo mal — error proporcional a la confianza equivocada
        error = predicted_confidence * 0.8 + 0.2

    return float(np.clip(error, 0.0, 1.0))


# ============================================================
# UPDATE DIVIDIDO — PRE Y POST RESPUESTA
# ============================================================

def update_pre_response(
    state: PredictiveState,
    texto_input: str,
    historial: list[tuple[str, str]],
    homeostasis_snap: Optional[dict] = None,
) -> PredictiveState:
    """
    Primera mitad del ciclo — corre ANTES de que Gemini responda.

    Paso 1: Calcular error de predicción del turno anterior.
    Paso 2: Inferir nuevas hipótesis de intención.
    Paso 4: Guardar pending_prediction para el turno siguiente.

    NO calcula predict_effect — eso requiere la respuesta real.
    El expected_effect queda con valores neutros hasta update_post_response().
    """
    # Paso 1 — error del turno anterior
    error = 0.0
    if state.pending_prediction:
        error = compute_prediction_error(
            state.pending_prediction,
            texto_input,
            historial,
        )

    # Paso 2 — nuevas hipótesis
    hipotesis = infer_intent(texto_input, historial, prior_state=state)

    # Paso 4 — guardar pending para el turno siguiente
    dominant = max(hipotesis, key=lambda h: h.confidence) if hipotesis else None
    pending = {
        "dominant_label":      dominant.label if dominant else "",
        "dominant_confidence": dominant.confidence if dominant else 0.5,
        "turn":                state.turn_count + 1,
    } if dominant else None

    # Paso 5 — actualizar historial de errores y confianza
    error_history = (state.error_history + [error])[-20:]
    model_confidence = float(np.clip(
        1.0 - (sum(error_history) / len(error_history)) if error_history else 0.5,
        0.0, 1.0
    ))

    return PredictiveState(
        intent_hypotheses    = hipotesis,
        expected_effect      = state.expected_effect,  # conservar hasta post_response
        prediction_error_last= error,
        error_history        = error_history,
        model_confidence     = model_confidence,
        pending_prediction   = pending,
        turn_count           = state.turn_count + 1,
        timestamp            = time.time(),
    )


def update_post_response(
    state: PredictiveState,
    respuesta_real: str,
    homeostasis_snap: Optional[dict] = None,
) -> PredictiveState:
    """
    Segunda mitad del ciclo — corre DESPUÉS de que Gemini responde.

    Paso 3: Calcular predict_effect con la respuesta real de ALTER.

    Retorna el estado actualizado con expected_effect preciso.
    """
    effect = predict_effect(
        respuesta_real,
        state.intent_hypotheses,
        homeostasis_snap,
    )
    # Solo actualizar expected_effect — el resto ya está calculado
    return PredictiveState(
        intent_hypotheses    = state.intent_hypotheses,
        expected_effect      = effect,
        prediction_error_last= state.prediction_error_last,
        error_history        = state.error_history,
        model_confidence     = state.model_confidence,
        pending_prediction   = state.pending_prediction,
        turn_count           = state.turn_count,
        timestamp            = state.timestamp,
    )

def update(
    state: PredictiveState,
    texto_input: str,
    historial: list[tuple[str, str]],
    response_candidate: str = "",
    homeostasis_snap: Optional[dict] = None,
) -> PredictiveState:
    """
    Ciclo completo por turno.

    Paso 1: Si hay pending_prediction del turno anterior, calcular error.
    Paso 2: Inferir nuevas hipótesis de intención.
    Paso 3: Predecir efecto de la respuesta candidata.
    Paso 4: Guardar predicción como pending para el turno siguiente.
    Paso 5: Actualizar confianza global del modelo.

    Retorna nuevo PredictiveState.
    """
    # Paso 1 — error del turno anterior
    error = 0.0
    if state.pending_prediction:
        error = compute_prediction_error(
            state.pending_prediction,
            texto_input,
            historial,
        )

    # Paso 2 — nuevas hipótesis
    hipotesis = infer_intent(texto_input, historial, prior_state=state)

    # Paso 3 — efecto esperado
    effect = predict_effect(
        response_candidate or texto_input,
        hipotesis,
        homeostasis_snap,
    )

    # Paso 4 — guardar predicción para el turno siguiente
    dominant = max(hipotesis, key=lambda h: h.confidence) if hipotesis else None
    pending = {
        "dominant_label":      dominant.label if dominant else "",
        "dominant_confidence": dominant.confidence if dominant else 0.5,
        "turn":                state.turn_count + 1,
    } if dominant else None

    # Paso 5 — actualizar historial de errores y confianza global
    error_history = (state.error_history + [error])[-20:]  # últimos 20
    model_confidence = float(np.clip(
        1.0 - (sum(error_history) / len(error_history)) if error_history else 0.5,
        0.0, 1.0
    ))

    return PredictiveState(
        intent_hypotheses    = hipotesis,
        expected_effect      = effect,
        prediction_error_last= error,
        error_history        = error_history,
        model_confidence     = model_confidence,
        pending_prediction   = pending,
        turn_count           = state.turn_count + 1,
        timestamp            = time.time(),
    )


# ============================================================
# CANDIDATOS PARA EL WORKSPACE
# ============================================================

def export_workspace_candidates(state: PredictiveState) -> list[dict]:
    """
    Genera candidatos para el GlobalWorkspace según el estado predictivo.

    Reglas:
        - Hipótesis dominante con confianza > 0.5 → user_hypothesis
        - Error alto (> 0.6) → internal_tension
        - Riesgo de desalineación alto (> 0.55) → constraint
    """
    candidates = []
    dominant = state.dominant_hypothesis()

    # user_hypothesis desde hipótesis dominante
    if dominant and dominant.confidence > 0.50:
        candidates.append({
            "type":         "user_hypothesis",
            "content":      f"Intención probable: {dominant.label} "
                            f"(conf:{dominant.confidence:.2f})",
            "source":       "prediction",
            "relevance":    dominant.confidence,
            "novelty":      0.3,
            "urgency":      0.4,
            "confidence":   dominant.confidence,
        })

    # internal_tension desde error alto
    if state.prediction_error_last > 0.60:
        candidates.append({
            "type":         "internal_tension",
            "content":      f"Error de predicción alto: {state.prediction_error_last:.2f} "
                            f"— modelo desajustado",
            "source":       "prediction",
            "relevance":    0.6,
            "novelty":      0.5,
            "urgency":      0.65,
            "confidence":   0.8,
        })

    # constraint desde riesgo de desalineación
    riesgo = state.expected_effect.get("riesgo_desalineacion", 0.0)
    if riesgo > 0.55:
        candidates.append({
            "type":         "constraint",
            "content":      f"Riesgo de desalineación: {riesgo:.2f} — considerar preguntar",
            "source":       "prediction",
            "relevance":    0.7,
            "novelty":      0.3,
            "urgency":      riesgo,
            "confidence":   state.model_confidence,
        })

    return candidates


# ============================================================
# SERIALIZACIÓN
# ============================================================

def serialize(state: PredictiveState) -> str:
    return json.dumps(state.to_dict(), ensure_ascii=False)


def deserialize(raw: str) -> PredictiveState:
    return PredictiveState.from_dict(json.loads(raw))


# ============================================================
# SNAPSHOT LEGIBLE PARA PROMPT
# ============================================================

def predictive_snapshot_str(state: PredictiveState) -> str:
    """Versión legible para incluir en prompts."""
    lines = []
    dominant = state.dominant_hypothesis()
    if dominant:
        lines.append(
            f"INTENCIÓN PROBABLE: {dominant.label} (conf:{dominant.confidence:.2f})"
        )
    if len(state.intent_hypotheses) > 1:
        otras = state.intent_hypotheses[1:3]
        lines.append(
            "Alternativas: " +
            ", ".join(f"{h.label}({h.confidence:.2f})" for h in otras)
        )
    effect = state.expected_effect
    lines.append(
        f"EFECTO ESPERADO: claridad:{effect.get('claridad',0):.2f} "
        f"satisfacción:{effect.get('satisfaccion',0):.2f} "
        f"riesgo:{effect.get('riesgo_desalineacion',0):.2f}"
    )
    if state.prediction_error_last > 0.3:
        lines.append(
            f"ERROR PREVIO: {state.prediction_error_last:.2f} "
            f"(confianza modelo: {state.model_confidence:.2f})"
        )
    return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    # Test 1: inferencia básica — señales léxicas
    hyps = infer_intent("cómo funciona esto?", [])
    if not hyps:
        errors.append("FAIL: infer_intent retornó lista vacía")
    total_conf = sum(h.confidence for h in hyps)
    if not (0.95 <= total_conf <= 1.05):
        errors.append(f"FAIL: confianzas no suman ~1.0 (suma={total_conf:.2f})")

    # Test 2: quiere_accion detectado
    hyps_accion = infer_intent("implementá el workspace ahora", [])
    labels = [h.label for h in hyps_accion]
    if "quiere_accion" not in labels:
        errors.append("FAIL: 'implementá' no detectó quiere_accion")

    # Test 3: ciclo update completo
    state = PredictiveState()
    state2 = update(state, "cómo funciona?", [], "Funciona así...")
    if not state2.intent_hypotheses:
        errors.append("FAIL: update no generó hipótesis")
    if state2.turn_count != 1:
        errors.append(f"FAIL: turn_count esperado 1, got {state2.turn_count}")

    # Test 4: error de predicción — predijo bien
    state3 = update(state2, "hacé algo concreto", [], "Acá va la implementación...")
    # state2 predijo quiere_analisis (por "cómo funciona")
    # state3 recibe "hacé algo" → debería calcular error
    if state3.prediction_error_last < 0.0 or state3.prediction_error_last > 1.0:
        errors.append(f"FAIL: prediction_error fuera de rango: {state3.prediction_error_last}")

    # Test 5: error alto genera candidato internal_tension
    state_error = PredictiveState(
        prediction_error_last=0.85,
        intent_hypotheses=[UserIntentHypothesis("quiere_exploracion", 0.7, "lexical")]
    )
    candidates = export_workspace_candidates(state_error)
    types = [c["type"] for c in candidates]
    if "internal_tension" not in types:
        errors.append("FAIL: error alto no generó internal_tension en workspace")

    # Test 6: riesgo alto genera constraint
    state_riesgo = PredictiveState(
        expected_effect={"claridad": 0.5, "satisfaccion": 0.5, "riesgo_desalineacion": 0.75},
        intent_hypotheses=[UserIntentHypothesis("quiere_cierre", 0.8, "lexical")]
    )
    candidates2 = export_workspace_candidates(state_riesgo)
    types2 = [c["type"] for c in candidates2]
    if "constraint" not in types2:
        errors.append("FAIL: riesgo alto no generó constraint en workspace")

    # Test 7: serialización round-trip
    raw = serialize(state3)
    state4 = deserialize(raw)
    if state4.turn_count != state3.turn_count:
        errors.append("FAIL: serialización round-trip perdió turn_count")

    return errors


if __name__ == "__main__":
    print("=== alter_predictive.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo ciclo ===")
    state = PredictiveState()
    turnos = [
        ("cómo funciona el workspace?",      "Funciona así: tick() recibe candidatos..."),
        ("hacé una prueba rápida",            "Acá va el test..."),
        ("te parece que está bien diseñado?", "Sí, me parece sólido porque..."),
        ("entonces cerramos la fase 2",       "Perfecto, fase 2 cerrada."),
    ]
    for texto, respuesta in turnos:
        state = update(state, texto, [], respuesta)
        dom = state.dominant_hypothesis()
        print(f"\nTurno {state.turn_count}: '{texto[:40]}'")
        print(f"  Intención: {dom.label if dom else '?'} ({dom.confidence:.2f})")
        print(f"  Error previo: {state.prediction_error_last:.2f}")
        print(f"  Confianza modelo: {state.model_confidence:.2f}")
        ef = state.expected_effect
        print(f"  Efecto esperado: claridad={ef['claridad']:.2f} "
              f"satisfacción={ef['satisfaccion']:.2f} "
              f"riesgo={ef['riesgo_desalineacion']:.2f}")