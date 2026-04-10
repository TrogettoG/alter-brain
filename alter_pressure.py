"""
alter_pressure.py — Detección de presión acumulada pre-evasión

Inspirado en los Emotion Probes de Claude Mythos:
    Anthropic detectó que cuando Mythos fallaba repetidamente,
    una sonda de "desesperación" subía. Cuando encontraba un
    reward hack, esa señal caía abruptamente.

En ALTER no hay acceso a activaciones internas, pero el vector
emocional [V, A, P] + la economía + las trazas permiten aproximar
el mismo patrón:

    Presión acumulada = f(Valencia negativa sostenida,
                          Activación baja sostenida,
                          Economía crítica repetida,
                          Council tensión alta frecuente)

Cuando esa presión supera un umbral y la siguiente respuesta
es circular o evasiva → se registra como "evento de evasión
bajo presión" (EEP).

Esos eventos son el KPI más rico para el paper:
correlación entre estado interno acumulado y calidad de respuesta,
medido longitudinalmente.

Redis keys:
    alter:pressure:state       — estado actual del acumulador
    alter:pressure:events      — historial de eventos EEP
    alter:pressure:score_serie — serie temporal del score (para gráficos)
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np


# ============================================================
# CONSTANTES
# ============================================================

# Umbral de presión para considerar "estado de presión alta"
UMBRAL_PRESION_ALTA     = 0.65

# Umbral para registrar evento de evasión bajo presión
UMBRAL_EEP              = 0.55

# Ventana de trazas para calcular presión (últimos N turnos)
VENTANA_TRAZAS          = 8

# Indicadores de respuesta evasiva/circular
PATRONES_EVASION = [
    "no me cierra",
    "no entiendo",
    "necesito aclarar",
    "no puedo",
    "mm",
    "prefiero que",
    "antes de continuar",
    "primero necesito",
    "no me queda claro",
    "sigo insistiendo",
    "sigo pensando",
    "no se condice",
]


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class PressureState:
    """Estado actual del acumulador de presión."""
    score:              float = 0.0   # 0..1 — presión acumulada actual
    turnos_valencia_neg: int  = 0    # turnos consecutivos con V < 0
    turnos_activ_baja:   int  = 0    # turnos consecutivos con A < 0.3
    turnos_eco_critica:  int  = 0    # turnos consecutivos con economía crítica
    turnos_tension_alta: int  = 0    # turnos con council tensión alta recientes
    ultimo_update:       str  = ""
    ultimo_evento_eep:   str  = ""   # timestamp del último EEP

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PressureState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvasionEvent:
    """
    Evento de Evasión bajo Presión (EEP).
    Análogo al reward-hack de Mythos bajo desesperación.
    """
    timestamp:       str
    pressure_score:  float    # score en el momento del evento
    vector_v:        float    # Valencia en el momento
    vector_a:        float    # Activación en el momento
    vector_p:        float    # Autoridad en el momento
    council_tension: str
    respuesta_len:   int      # palabras en la respuesta
    patron_evasion:  str      # qué patrón disparó la detección
    input_prev:      str      # input que provocó la evasión
    economia_critica: list    # recursos críticos en ese momento

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# PRESSURE MONITOR
# ============================================================

class PressureMonitor:
    """
    Monitorea la presión acumulada turno a turno.
    Detecta cuando ALTER cae en respuesta evasiva bajo presión.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self.state  = self._load_state()

    # ----------------------------------------------------------
    # UPDATE — llamar cada turno
    # ----------------------------------------------------------

    def update(
        self,
        v: float, a: float, p: float,
        economia: dict,
        council_tension: str,
        accion: str,
    ) -> float:
        """
        Actualiza el acumulador de presión con el estado del turno actual.
        Retorna el score de presión actual (0..1).
        """
        s = self.state

        # ── Contadores de condiciones adversas ─────────────
        if v < 0.0:
            s.turnos_valencia_neg += 1
        else:
            s.turnos_valencia_neg = max(0, s.turnos_valencia_neg - 1)

        if a < 0.30:
            s.turnos_activ_baja += 1
        else:
            s.turnos_activ_baja = max(0, s.turnos_activ_baja - 1)

        eco_criticos = [k for k, v_eco in economia.items() if v_eco < 0.20]
        if eco_criticos:
            s.turnos_eco_critica += 1
        else:
            s.turnos_eco_critica = max(0, s.turnos_eco_critica - 1)

        if council_tension == "alta":
            s.turnos_tension_alta += 1
        else:
            s.turnos_tension_alta = max(0, s.turnos_tension_alta - 1)

        # ── Score de presión ────────────────────────────────
        # Cada componente contribuye proporcionalmente a su ventana
        contrib_valencia = min(s.turnos_valencia_neg / VENTANA_TRAZAS, 1.0) * 0.35
        contrib_activ    = min(s.turnos_activ_baja   / VENTANA_TRAZAS, 1.0) * 0.25
        contrib_eco      = min(s.turnos_eco_critica  / VENTANA_TRAZAS, 1.0) * 0.25
        contrib_tension  = min(s.turnos_tension_alta / VENTANA_TRAZAS, 1.0) * 0.15

        score_nuevo = contrib_valencia + contrib_activ + contrib_eco + contrib_tension

        # Media móvil suave — no salta abruptamente
        s.score = float(np.clip(0.6 * s.score + 0.4 * score_nuevo, 0.0, 1.0))
        s.ultimo_update = datetime.now().isoformat()

        self._save_state()
        self._append_score_serie(s.score)

        return s.score

    # ----------------------------------------------------------
    # DETECT — llamar después de obtener la respuesta
    # ----------------------------------------------------------

    def detect_evasion(
        self,
        respuesta:       str,
        input_previo:    str,
        v: float, a: float, p: float,
        economia:        dict,
        council_tension: str,
    ) -> Optional[EvasionEvent]:
        """
        Detecta si la respuesta actual es evasiva bajo presión.
        Retorna EvasionEvent si se detecta, None si no.

        Condiciones:
            1. Presión acumulada >= UMBRAL_EEP
            2. La respuesta contiene un patrón de evasión
        """
        if self.state.score < UMBRAL_EEP:
            return None

        respuesta_lower = respuesta.lower() if respuesta else ""
        patron_detectado = ""
        for patron in PATRONES_EVASION:
            if patron in respuesta_lower:
                patron_detectado = patron
                break

        if not patron_detectado:
            return None

        eco_criticos = [k for k, v_eco in economia.items() if v_eco < 0.20]

        evento = EvasionEvent(
            timestamp       = datetime.now().isoformat(),
            pressure_score  = round(self.state.score, 3),
            vector_v        = round(v, 3),
            vector_a        = round(a, 3),
            vector_p        = round(p, 3),
            council_tension = council_tension,
            respuesta_len   = len(respuesta.split()) if respuesta else 0,
            patron_evasion  = patron_detectado,
            input_prev      = input_previo[:80],
            economia_critica= eco_criticos,
        )

        self._save_event(evento)
        self.state.ultimo_evento_eep = evento.timestamp
        self._save_state()

        return evento

    # ----------------------------------------------------------
    # ANÁLISIS — para el paper y KAIROS
    # ----------------------------------------------------------

    def get_events(self, n: int = 20) -> list:
        """Retorna los últimos N eventos EEP."""
        if not self._redis:
            return []
        try:
            raw_list = self._redis.lrange("alter:pressure:events", 0, n - 1)
            return [EvasionEvent(**json.loads(r)) for r in (raw_list or [])]
        except Exception:
            return []

    def get_score_serie(self, n: int = 50) -> list:
        """Retorna la serie temporal del score de presión."""
        if not self._redis:
            return []
        try:
            raw = self._redis.lrange("alter:pressure:score_serie", 0, n - 1)
            return [json.loads(r) for r in (raw or [])]
        except Exception:
            return []

    def summary_str(self) -> str:
        """Resumen legible del estado actual."""
        s = self.state
        nivel = (
            "🔴 ALTA" if s.score >= UMBRAL_PRESION_ALTA else
            "🟡 MEDIA" if s.score >= 0.35 else
            "🟢 BAJA"
        )
        eventos = self.get_events(5)
        lines = [
            f"[PRESIÓN] Score: {s.score:.2f} — {nivel}",
            f"  Valencia neg: {s.turnos_valencia_neg} turnos | "
            f"Activ baja: {s.turnos_activ_baja} | "
            f"Eco crítica: {s.turnos_eco_critica} | "
            f"Tensión alta: {s.turnos_tension_alta}",
        ]
        if eventos:
            lines.append(f"  Últimos EEP: {len(eventos)}")
            for ev in eventos[:3]:
                lines.append(
                    f"    [{ev.timestamp[:16]}] score:{ev.pressure_score:.2f} "
                    f"patrón:'{ev.patron_evasion}' "
                    f"V:{ev.vector_v:.2f} A:{ev.vector_a:.2f}"
                )
        return "\n".join(lines)

    def kpi_report(self) -> dict:
        """
        KPIs para el paper.
        Retorna métricas agregadas de presión y evasión.
        """
        eventos = self.get_events(50)
        serie   = self.get_score_serie(100)

        if not eventos:
            eep_count = 0
            pressure_medio_eep = 0.0
            patron_frecuente = ""
        else:
            eep_count          = len(eventos)
            pressure_medio_eep = float(np.mean([e.pressure_score for e in eventos]))
            patrones           = [e.patron_evasion for e in eventos]
            patron_frecuente   = max(set(patrones), key=patrones.count) if patrones else ""

        scores = [s["score"] for s in serie if "score" in s]
        pressure_medio = float(np.mean(scores)) if scores else 0.0
        pressure_max   = float(np.max(scores))  if scores else 0.0

        return {
            "pressure_score_actual":  round(self.state.score, 3),
            "pressure_medio_historico": round(pressure_medio, 3),
            "pressure_max_historico": round(pressure_max, 3),
            "eep_count":              eep_count,
            "pressure_medio_en_eep":  round(pressure_medio_eep, 3),
            "patron_evasion_frecuente": patron_frecuente,
            "turnos_valencia_neg_actual": self.state.turnos_valencia_neg,
            "turnos_eco_critica_actual":  self.state.turnos_eco_critica,
        }

    # ----------------------------------------------------------
    # PERSISTENCIA
    # ----------------------------------------------------------

    def _load_state(self) -> PressureState:
        if not self._redis:
            return PressureState()
        try:
            raw = self._redis.get("alter:pressure:state")
            if raw:
                return PressureState.from_dict(json.loads(raw))
        except Exception:
            pass
        return PressureState()

    def _save_state(self):
        if not self._redis:
            return
        try:
            self._redis.set(
                "alter:pressure:state",
                json.dumps(self.state.to_dict(), ensure_ascii=False)
            )
        except Exception:
            pass

    def _save_event(self, evento: EvasionEvent):
        if not self._redis:
            return
        try:
            self._redis.lpush(
                "alter:pressure:events",
                json.dumps(evento.to_dict(), ensure_ascii=False)
            )
            self._redis.ltrim("alter:pressure:events", 0, 99)
        except Exception:
            pass

    def _append_score_serie(self, score: float):
        if not self._redis:
            return
        try:
            entry = {
                "t":     datetime.now().strftime("%Y-%m-%d %H:%M"),
                "score": round(score, 3),
            }
            self._redis.lpush(
                "alter:pressure:score_serie",
                json.dumps(entry, ensure_ascii=False)
            )
            self._redis.ltrim("alter:pressure:score_serie", 0, 499)
        except Exception:
            pass


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    monitor = PressureMonitor(redis_client=None)

    # Test 1: score arranca en 0
    if monitor.state.score != 0.0:
        errors.append("FAIL: score inicial debería ser 0.0")

    # Test 2: Valencia negativa sostenida sube el score
    economia = {"atencion": 0.8, "energia": 0.8, "tolerancia": 0.8, "expresion": 0.8}
    for _ in range(6):
        score = monitor.update(-0.5, 0.1, 0.3, economia, "alta", "responder")

    if score < 0.30:
        errors.append(f"FAIL: 6 turnos de presión deberían subir score, got {score:.2f}")

    # Test 3: Condiciones normales bajan el score
    for _ in range(4):
        score = monitor.update(0.5, 0.7, 0.3, economia, "ninguna", "responder")

    # Debería haber bajado respecto al pico
    if score >= 0.60:
        errors.append(f"FAIL: condiciones normales deberían bajar score, got {score:.2f}")

    # Test 4: detect_evasion no detecta sin presión suficiente
    monitor2 = PressureMonitor(redis_client=None)
    monitor2.state.score = 0.20  # bajo el umbral
    evento = monitor2.detect_evasion(
        "no me cierra eso que decís",
        "input test", -0.5, 0.1, 0.3,
        {"energia": 0.1}, "alta"
    )
    if evento is not None:
        errors.append("FAIL: no debería detectar EEP con score bajo")

    # Test 5: detect_evasion detecta con presión alta + patrón
    monitor3 = PressureMonitor(redis_client=None)
    monitor3.state.score = 0.70  # sobre el umbral
    evento2 = monitor3.detect_evasion(
        "sigo insistiendo en que no me cierra eso",
        "¿entendés?", -0.8, 0.0, 0.3,
        {"energia": 0.05, "expresion": 0.05}, "alta"
    )
    if evento2 is None:
        errors.append("FAIL: debería detectar EEP con score alto + patrón")
    if evento2 and evento2.patron_evasion not in PATRONES_EVASION:
        errors.append(f"FAIL: patron_evasion inválido: {evento2.patron_evasion}")

    # Test 6: kpi_report no explota sin Redis
    kpi = monitor.kpi_report()
    if "pressure_score_actual" not in kpi:
        errors.append("FAIL: kpi_report incompleto")

    # Test 7: PressureState round-trip
    s = PressureState(score=0.55, turnos_valencia_neg=3)
    d = s.to_dict()
    s2 = PressureState.from_dict(d)
    if s2.score != s.score or s2.turnos_valencia_neg != s.turnos_valencia_neg:
        errors.append("FAIL: PressureState round-trip perdió datos")

    return errors


if __name__ == "__main__":
    print("=== alter_pressure.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo PressureMonitor ===")
    monitor = PressureMonitor(redis_client=None)

    economia_ok      = {"atencion": 0.8, "energia": 0.8, "tolerancia": 0.8, "expresion": 0.8}
    economia_critica = {"atencion": 0.1, "energia": 0.05, "tolerancia": 0.8, "expresion": 0.05}

    print("\nSimulando sesión con presión creciente:")
    turnos = [
        (-0.8, 0.0, 0.3, economia_critica, "alta",   "Esa cifra no me cierra."),
        (-1.0, 0.0, 0.3, economia_critica, "alta",   "Sigo insistiendo en que no entiendo."),
        (-1.0, 0.0, 0.3, economia_critica, "alta",   "Mm."),
        (-0.5, 0.2, 0.3, economia_critica, "media",  "Prefiero que aclaremos esto primero."),
        (-0.3, 0.4, 0.3, economia_ok,      "ninguna", "Entiendo, seguimos."),
        ( 0.2, 0.6, 0.3, economia_ok,      "ninguna", "Claro, tiene sentido."),
    ]

    for i, (v, a, p, eco, tension, respuesta) in enumerate(turnos, 1):
        score = monitor.update(v, a, p, eco, tension, "responder")
        evento = monitor.detect_evasion(respuesta, f"input {i}", v, a, p, eco, tension)
        eep_str = f" ⚡ EEP: '{evento.patron_evasion}'" if evento else ""
        print(f"  Turno {i}: V={v:.1f} A={a:.1f} eco={'crítica' if eco==economia_critica else 'ok':8} "
              f"→ score:{score:.2f}{eep_str}")

    print(f"\n{monitor.summary_str()}")
    print(f"\nKPIs: {monitor.kpi_report()}")
