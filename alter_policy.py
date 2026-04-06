"""
alter_policy.py — Policy Arbiter de AlterB3 (Fase 4)

Centraliza la decisión de acción de ALTER.
Antes: heurísticas distribuidas por procesar_input y procesar_turno.
Ahora: un árbol de prioridades explícito con trazabilidad.

Árbol de decisión (orden de prioridad):
    1. ¿Restricción de pizarra violada?        → ignorar / corregir
    2. ¿Economía crítica?                       → registrar
    3. ¿Riesgo de desalineación > 0.65?         → preguntar
    4. ¿Patrón procedural activo (sr > 0.7)?    → promover candidate_action (Opción B)
    5. ¿Herramienta requerida?                  → usar_herramienta
    6. ¿Council recomienda acción específica?   → respetar
    7. Default                                  → responder

Opción B para patrones procedurales:
    El patrón informa, no decide.
    Si hay patrón con success_rate > 0.7, genera un candidate_action
    en el workspace con source="procedural_memory".
    El Council lo considera junto con las otras señales.

Integración gradual:
    El Arbiter corre DESPUÉS de procesar_turno — valida y puede
    modificar la acción que Gemini eligió.
    En versiones futuras puede correr ANTES para ahorrar el llamado.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ============================================================
# CONSTANTES
# ============================================================

ACCIONES_VALIDAS = {
    "responder",        # respuesta directa
    "preguntar",        # pedir clarificación antes de responder
    "usar_herramienta", # ejecutar tool
    "registrar",        # guardar sin responder
    "ignorar",          # no hacer nada
    "reformular",       # responder pero cambiando el enfoque
    "diferir",          # posponer — sube prioridad de agenda, no promete respuesta
}

UMBRAL_RIESGO_DESALINEACION = 0.65
UMBRAL_PATRON_PROCEDURAL    = 0.70
UMBRAL_ECONOMIA_CRITICA     = 0.15


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class PolicyDecision:
    action:        str            # acción elegida
    confidence:    float          # qué tan seguro está el Arbiter (0..1)
    reason:        str            # por qué eligió esta acción
    source:        str            # módulo que dominó: "pizarra"|"economia"|"prediccion"|
                                  #   "procedural"|"herramienta"|"council"|"default"
    override_data: dict = field(default_factory=dict)
    # override_data puede contener:
    #   "pattern_id":    str — si source=procedural
    #   "tool_name":     str — si action=usar_herramienta
    #   "pregunta":      str — si action=preguntar, la pregunta sugerida
    #   "agenda_item":   str — si action=diferir

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"PolicyDecision(action={self.action}, "
                f"conf={self.confidence:.2f}, "
                f"src={self.source}, "
                f"reason={self.reason[:60]})")


# ============================================================
# POLICY ARBITER
# ============================================================

class PolicyArbiter:
    """
    Árbitro central de decisiones de ALTER.

    Recibe el estado completo de todos los módulos AlterB3
    y elige la acción más apropiada según el árbol de prioridades.

    Integración gradual:
        Fase 4A — corre post-Gemini, puede sobrescribir la acción
        Fase 4B (futura) — corre pre-Gemini, ahorra llamadas innecesarias
    """

    def decide(
        self,
        # Inputs del sistema
        gemini_action:      str,            # lo que Gemini decidió
        gemini_confidence:  float,          # confianza del output de Gemini
        texto_input:        str,
        # Estado de los módulos B3
        workspace_snap:     Optional[dict] = None,
        homeostasis_snap:   Optional[dict] = None,
        predictive_state=   None,           # PredictiveState
        procedural_patterns: list = None,   # list[ProceduralPattern]
        economia:           dict = None,
        pizarra:            dict = None,
        council_tension:    str = "ninguna",
        council_action:     str = "",       # acción que recomendó el Council
        forzar_responder:   bool = False,
        canal:              str = "terminal",
    ) -> PolicyDecision:
        """
        Árbol de prioridades.
        Retorna PolicyDecision con la acción final y su trazabilidad.
        """
        ws   = workspace_snap   or {}
        hs   = homeostasis_snap or {}
        eco  = economia         or {}
        piz  = pizarra          or {}
        proc = procedural_patterns or []

        # ── Paso 1: Restricción de pizarra ────────────────────
        pizarra_violation = self._check_pizarra(texto_input, piz)
        if pizarra_violation:
            return PolicyDecision(
                action     = "ignorar",
                confidence = 0.95,
                reason     = f"Colisión con pizarra: {pizarra_violation}",
                source     = "pizarra",
            )

        # ── Paso 2: Economía crítica ───────────────────────────
        if not forzar_responder:
            eco_critica = self._check_economia(eco)
            if eco_critica:
                return PolicyDecision(
                    action     = "registrar",
                    confidence = 0.85,
                    reason     = f"Economía crítica: {eco_critica}",
                    source     = "economia",
                )

        # ── Paso 3: Riesgo de desalineación ───────────────────
        if not forzar_responder:
            riesgo = 0.0
            if predictive_state:
                riesgo = predictive_state.expected_effect.get(
                    "riesgo_desalineacion", 0.0
                )
            if riesgo > UMBRAL_RIESGO_DESALINEACION:
                pregunta = self._generar_pregunta_clarificacion(
                    texto_input, predictive_state
                )
                return PolicyDecision(
                    action     = "preguntar",
                    confidence = float(np.clip(riesgo, 0.0, 1.0)),
                    reason     = f"Riesgo desalineación: {riesgo:.2f}",
                    source     = "prediccion",
                    override_data = {"pregunta": pregunta}
                )

        # ── Paso 4: Patrón procedural activo (Opción B) ────────
        patron_activo = self._find_active_pattern(proc, ws, hs, predictive_state)
        if patron_activo:
            # No decide — promueve candidate_action al workspace
            # La decisión final la toma el Council
            # Retornamos una señal para que alter_brain inyecte el candidato
            return PolicyDecision(
                action     = gemini_action,  # respeta la decisión de Gemini
                confidence = float(np.clip(gemini_confidence * 0.9 + 0.1, 0.0, 1.0)),
                reason     = f"Patrón procedural activo: '{patron_activo.trigger[:50]}'",
                source     = "procedural",
                override_data = {
                    "pattern_id":      patron_activo.id,
                    "pattern_trigger": patron_activo.trigger,
                    "pattern_response": patron_activo.response,
                    "pattern_sr":      patron_activo.success_rate,
                    "promote_to_workspace": True,  # señal para alter_brain
                }
            )

        # ── Paso 5: Herramienta requerida ─────────────────────
        if gemini_action == "usar_herramienta":
            return PolicyDecision(
                action     = "usar_herramienta",
                confidence = gemini_confidence,
                reason     = "Gemini solicitó herramienta",
                source     = "herramienta",
            )

        # ── Paso 6: Council recomienda acción específica ───────
        if council_action and council_action in ACCIONES_VALIDAS:
            tension_weight = {"ninguna": 0.0, "baja": 0.2, "media": 0.5, "alta": 0.8}.get(
                council_tension, 0.0
            )
            if tension_weight >= 0.5:
                return PolicyDecision(
                    action     = council_action,
                    confidence = float(np.clip(tension_weight, 0.0, 1.0)),
                    reason     = f"Council con tensión {council_tension} recomienda {council_action}",
                    source     = "council",
                )

        # ── Paso 7: Default ────────────────────────────────────
        # Respetar lo que Gemini decidió
        return PolicyDecision(
            action     = gemini_action if gemini_action in ACCIONES_VALIDAS else "responder",
            confidence = gemini_confidence,
            reason     = "Decisión de Gemini sin override",
            source     = "default",
        )

    # ----------------------------------------------------------
    # HELPERS PRIVADOS
    # ----------------------------------------------------------

    def _check_pizarra(self, texto: str, pizarra: dict) -> str:
        """
        Detecta colisión con decisiones inamovibles de la pizarra.
        Retorna descripción de la violación o "" si no hay.
        """
        if not pizarra:
            return ""
        texto_lower = texto.lower()
        for decision in pizarra.get("decisiones", []):
            keywords = decision.get("keywords_colision", [])
            if keywords and any(k.lower() in texto_lower for k in keywords):
                return f"{decision.get('id','?')}: {decision.get('decision','')[:60]}"
        return ""

    def _check_economia(self, economia: dict) -> str:
        """
        Detecta recursos en estado crítico.
        Retorna lista de recursos críticos o "" si todo está bien.
        """
        criticos = [
            k for k, v in economia.items()
            if isinstance(v, (int, float)) and v < UMBRAL_ECONOMIA_CRITICA
        ]
        return ", ".join(criticos) if criticos else ""

    def _find_active_pattern(
        self,
        patterns: list,
        workspace_snap: dict,
        homeostasis_snap: dict,
        predictive_state,
    ):
        """
        Busca un patrón procedural con success_rate > umbral
        que matchee el contexto actual.
        Retorna el patrón o None.
        """
        if not patterns:
            return None

        # Construir contexto para matching
        dominant_hyp = None
        if predictive_state and predictive_state.intent_hypotheses:
            dominant_hyp = predictive_state.dominant_hypothesis()

        context = {
            "user_intent":           dominant_hyp.label if dominant_hyp else "desconocido",
            "prediction_error_high": (
                predictive_state.prediction_error_last > 0.6
                if predictive_state else False
            ),
            "workspace_has_goal":    bool(workspace_snap.get("dominant_goal")),
            "homeostasis_mode":      homeostasis_snap.get("modo_sugerido", "exploracion"),
        }

        candidatos = [
            p for p in patterns
            if p.success_rate > UMBRAL_PATRON_PROCEDURAL
            and p.matches(context) > 0.5
        ]

        if not candidatos:
            return None

        return max(candidatos, key=lambda p: p.success_rate * p.matches(context))

    def _generar_pregunta_clarificacion(
        self,
        texto_input: str,
        predictive_state,
    ) -> str:
        """
        Genera una pregunta de clarificación cuando el riesgo de desalineación es alto.
        Usa las hipótesis de intención para orientar la pregunta.
        """
        if not predictive_state or not predictive_state.intent_hypotheses:
            return "¿Podés contarme un poco más qué necesitás exactamente?"

        dominant = predictive_state.dominant_hypothesis()
        alternativas = predictive_state.intent_hypotheses[1:3]

        if dominant and alternativas:
            alt_labels = [h.label for h in alternativas]
            if "quiere_accion" in alt_labels:
                return "¿Preferís que lo implemente directamente o que primero lo explique?"
            if "quiere_validacion" in alt_labels:
                return "¿Buscás que lo desarrolle o que confirme si el enfoque te parece bien?"
            if "quiere_cierre" in alt_labels:
                return "¿Querés que lo cerremos acá o seguimos desarrollando?"

        return "¿Podés especificar un poco más qué necesitás?"


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []
    arbiter = PolicyArbiter()

    # Test 1: economia crítica → registrar
    decision = arbiter.decide(
        gemini_action="responder",
        gemini_confidence=0.8,
        texto_input="hola",
        economia={"energia": 0.05, "atencion": 0.8},
    )
    if decision.action != "registrar":
        errors.append(f"FAIL: economia crítica debería dar registrar, got {decision.action}")
    if decision.source != "economia":
        errors.append(f"FAIL: source debería ser 'economia', got {decision.source}")

    # Test 2: forzar_responder ignora economia crítica
    decision2 = arbiter.decide(
        gemini_action="responder",
        gemini_confidence=0.8,
        texto_input="respondeme",
        economia={"energia": 0.05},
        forzar_responder=True,
    )
    if decision2.action == "registrar":
        errors.append("FAIL: forzar_responder=True no debería dar registrar")

    # Test 3: riesgo alto → preguntar
    class FakePredictive:
        expected_effect = {"riesgo_desalineacion": 0.80}
        intent_hypotheses = []
        prediction_error_last = 0.3
        def dominant_hypothesis(self): return None

    decision3 = arbiter.decide(
        gemini_action="responder",
        gemini_confidence=0.7,
        texto_input="qué pensás de esto",
        predictive_state=FakePredictive(),
    )
    if decision3.action != "preguntar":
        errors.append(f"FAIL: riesgo alto debería dar preguntar, got {decision3.action}")
    if decision3.source != "prediccion":
        errors.append(f"FAIL: source debería ser 'prediccion', got {decision3.source}")

    # Test 4: patrón procedural activo → source=procedural, acción respetada
    class FakePattern:
        id = "p1"
        trigger = "usuario quiere acción"
        response = "dar código primero"
        success_rate = 0.85
        def matches(self, context, threshold=0.5):
            return 0.9

    class FakePredictive2:
        expected_effect = {"riesgo_desalineacion": 0.2}
        intent_hypotheses = []
        prediction_error_last = 0.2
        def dominant_hypothesis(self): return None

    decision4 = arbiter.decide(
        gemini_action="responder",
        gemini_confidence=0.75,
        texto_input="hacé algo",
        procedural_patterns=[FakePattern()],
        predictive_state=FakePredictive2(),
    )
    if decision4.source != "procedural":
        errors.append(f"FAIL: patrón activo debería dar source=procedural, got {decision4.source}")
    if not decision4.override_data.get("promote_to_workspace"):
        errors.append("FAIL: patrón activo debería tener promote_to_workspace=True")

    # Test 5: herramienta → usar_herramienta
    decision5 = arbiter.decide(
        gemini_action="usar_herramienta",
        gemini_confidence=0.9,
        texto_input="buscame algo",
    )
    if decision5.action != "usar_herramienta":
        errors.append(f"FAIL: herramienta debería dar usar_herramienta, got {decision5.action}")

    # Test 6: default → respeta Gemini
    decision6 = arbiter.decide(
        gemini_action="responder",
        gemini_confidence=0.8,
        texto_input="hola cómo estás",
    )
    if decision6.action != "responder":
        errors.append(f"FAIL: default debería respetar Gemini, got {decision6.action}")
    if decision6.source != "default":
        errors.append(f"FAIL: default debería tener source=default, got {decision6.source}")

    # Test 7: PolicyDecision tiene todos los campos
    d = decision6.to_dict()
    for campo in ["action", "confidence", "reason", "source", "override_data"]:
        if campo not in d:
            errors.append(f"FAIL: PolicyDecision falta campo '{campo}'")

    return errors


if __name__ == "__main__":
    print("=== alter_policy.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo árbol de decisión ===")
    arbiter = PolicyArbiter()
    casos = [
        {
            "nombre": "Economía crítica",
            "kwargs": {
                "gemini_action": "responder", "gemini_confidence": 0.8,
                "texto_input": "contame algo",
                "economia": {"energia": 0.05, "atencion": 0.8},
            }
        },
        {
            "nombre": "Riesgo alto",
            "kwargs": {
                "gemini_action": "responder", "gemini_confidence": 0.7,
                "texto_input": "qué pensás",
                "predictive_state": type("P", (), {
                    "expected_effect": {"riesgo_desalineacion": 0.75},
                    "intent_hypotheses": [],
                    "prediction_error_last": 0.3,
                    "dominant_hypothesis": lambda self: None
                })(),
            }
        },
        {
            "nombre": "Default normal",
            "kwargs": {
                "gemini_action": "responder", "gemini_confidence": 0.85,
                "texto_input": "implementá el workspace",
            }
        },
    ]
    for caso in casos:
        d = arbiter.decide(**caso["kwargs"])
        print(f"\n{caso['nombre']}: {d}")
