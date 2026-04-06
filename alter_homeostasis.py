"""
alter_homeostasis.py — Capa de Homeostasis de AlterB3 (Fase 1A)

Estado fisiológico-cognitivo unificado de ALTER.

Unifica en un solo sistema coherente:
  - Vector emocional E=[V, A, P]  (alter_brain.py)
  - Economía mental               (alter_mind.py: ECONOMIA_DEFAULT)
  - Drives                        (alter_mind.py: DRIVES_DEFAULT)
  - Estado basal del campo mental (alter_mind.py: CAMPO_DEFAULT)

Regla de migración:
  Primero adapter, después reemplazo.
  El sistema viejo sigue funcionando. export_compat_view()
  le devuelve a alter_brain.py la vista que espera.

Invariantes:
  - Todos los campos en 0.0..1.0 excepto valencia (-1.0..1.0)
  - mayor fatiga nunca mejora claridad sola
  - menor energia nunca aumenta carga cognitiva disponible
  - presion extrema reduce estabilidad

AlterB3 Fase 1A — no incluye:
  - acoplamiento con workspace (Fase 1B)
  - aprendizaje homeostático largo (Fase 3)
  - emociones complejas diferenciadas
"""

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np


# ============================================================
# CONTRATO CENTRAL
# ============================================================

@dataclass
class HomeostasisState:
    # Capacidad disponible para actuar
    energia: float = 0.75

    # Desgaste acumulado — sube lento, baja lento
    fatiga: float = 0.20

    # Calidad de procesamiento — función de energia, fatiga, presion
    claridad: float = 0.70

    # Nivel de arousal / activación
    activacion: float = 0.50

    # Tono basal afectivo — EXCEPCIÓN: rango -1.0..1.0
    valencia: float = 0.30

    # Presión interna / social acumulada
    presion: float = 0.30

    # Impulso exploratorio
    curiosidad: float = 0.60

    # Deseo de resolver loops abiertos
    necesidad_cierre: float = 0.35

    # Conflicto acumulado entre señales o metas
    tension_interna: float = 0.20

    # Cohesión general del sistema
    estabilidad: float = 0.72

    # Costo actual del contexto activo
    carga_cognitiva: float = 0.25

    # Timestamp de último update
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self._clamp()

    def _clamp(self):
        """Aplica invariantes de rango."""
        self.energia          = float(np.clip(self.energia,          0.0,  1.0))
        self.fatiga           = float(np.clip(self.fatiga,           0.0,  1.0))
        self.claridad         = float(np.clip(self.claridad,         0.0,  1.0))
        self.activacion       = float(np.clip(self.activacion,       0.0,  1.0))
        self.valencia         = float(np.clip(self.valencia,        -1.0,  1.0))
        self.presion          = float(np.clip(self.presion,          0.0,  1.0))
        self.curiosidad       = float(np.clip(self.curiosidad,       0.0,  1.0))
        self.necesidad_cierre = float(np.clip(self.necesidad_cierre, 0.0,  1.0))
        self.tension_interna  = float(np.clip(self.tension_interna,  0.0,  1.0))
        self.estabilidad      = float(np.clip(self.estabilidad,      0.0,  1.0))
        self.carga_cognitiva  = float(np.clip(self.carga_cognitiva,  0.0,  1.0))

    def validate(self) -> list[str]:
        """Verifica invariantes. Retorna lista de violaciones."""
        violations = []
        if self.fatiga > 0.7 and self.claridad > 0.8:
            violations.append("fatiga alta no debería coexistir con claridad muy alta")
        if self.energia < 0.2 and self.carga_cognitiva < 0.1:
            violations.append("energia crítica con carga cognitiva casi nula es inconsistente")
        if self.presion > 0.85 and self.estabilidad > 0.8:
            violations.append("presion extrema no debería coexistir con estabilidad muy alta")
        return violations

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HomeostasisState":
        campos_validos = {k for k in d if k in cls.__dataclass_fields__}
        return cls(**{k: d[k] for k in campos_validos})


# ============================================================
# ADAPTER DESDE ESTADO LEGACY
# ============================================================

def load_legacy_state(
    v: float = 0.3,
    a: float = 0.5,
    p: float = 0.3,
    economia: Optional[dict] = None,
    drives: Optional[dict] = None,
    nivel_cansancio: float = 0.0,
) -> dict:
    """
    Lee el estado actual del sistema viejo y lo normaliza
    en un dict compatible con build_homeostasis_state().

    Parámetros:
        v, a, p         — vector emocional actual
        economia        — dict con atención, energía, tolerancia, expresión
        drives          — dict con curiosidad, expresion, conexion, eficiencia
        nivel_cansancio — float 0..1
    """
    eco = economia or {
        "atencion": 0.7, "energia": 0.75,
        "tolerancia": 0.8, "expresion": 0.6
    }
    drv = drives or {
        "curiosidad": 0.6, "expresion": 0.4,
        "conexion": 0.5, "eficiencia": 0.3
    }
    return {
        "v": v, "a": a, "p": p,
        "economia": eco,
        "drives": drv,
        "nivel_cansancio": nivel_cansancio,
    }


def build_homeostasis_state(legacy: dict) -> HomeostasisState:
    """
    Construye HomeostasisState desde el estado legacy normalizado.

    Mapeo:
        V -> valencia          (directo, -1..1)
        A -> activacion        (directo, 0..1)
        P -> presion           (directo, 0..1)
        economia.energia     -> energia
        economia.atencion    -> claridad (parcial)
        economia.tolerancia  -> estabilidad (parcial)
        economia.expresion   -> parte de tension_interna (inversa)
        drives.curiosidad    -> curiosidad
        drives.eficiencia    -> necesidad_cierre (proxy)
        nivel_cansancio      -> fatiga
        historial implícito  -> carga_cognitiva
    """
    v   = legacy.get("v", 0.3)
    a   = legacy.get("a", 0.5)
    p   = legacy.get("p", 0.3)
    eco = legacy.get("economia", {})
    drv = legacy.get("drives", {})
    cansancio = legacy.get("nivel_cansancio", 0.0)

    energia    = eco.get("energia", 0.75)
    atencion   = eco.get("atencion", 0.70)
    tolerancia = eco.get("tolerancia", 0.80)
    expresion  = eco.get("expresion", 0.60)

    # Claridad: función de atención, energía y cansancio inverso
    claridad = float(np.clip(
        0.4 * atencion + 0.4 * energia + 0.2 * (1.0 - cansancio),
        0.0, 1.0
    ))

    # Estabilidad: función de tolerancia y valencia normalizada
    valencia_norm = (v + 1.0) / 2.0  # -1..1 → 0..1 solo para este cálculo
    estabilidad = float(np.clip(
        0.5 * tolerancia + 0.3 * valencia_norm + 0.2 * (1.0 - p),
        0.0, 1.0
    ))

    # Tensión interna: presión alta + baja expresión + baja tolerancia
    tension_interna = float(np.clip(
        0.4 * p + 0.3 * (1.0 - expresion) + 0.3 * (1.0 - tolerancia),
        0.0, 1.0
    ))

    # Necesidad de cierre: proxy desde eficiencia
    necesidad_cierre = float(np.clip(drv.get("eficiencia", 0.3) * 1.1, 0.0, 1.0))

    # Carga cognitiva: estimada desde cansancio y presión
    carga_cognitiva = float(np.clip(
        0.5 * cansancio + 0.3 * p + 0.2 * (1.0 - energia),
        0.0, 1.0
    ))

    state = HomeostasisState(
        energia          = energia,
        fatiga           = float(np.clip(cansancio, 0.0, 1.0)),
        claridad         = claridad,
        activacion       = a,
        valencia         = v,
        presion          = p,
        curiosidad       = float(np.clip(drv.get("curiosidad", 0.6), 0.0, 1.0)),
        necesidad_cierre = necesidad_cierre,
        tension_interna  = tension_interna,
        estabilidad      = estabilidad,
        carga_cognitiva  = carga_cognitiva,
        timestamp        = time.time(),
    )

    return state


# ============================================================
# IMPACTO POR TURNO
# ============================================================

def apply_turn_impact(
    state: HomeostasisState,
    turn_metrics: dict
) -> HomeostasisState:
    """
    Aplica el costo de un turno sobre el estado homeostático.

    turn_metrics esperados:
        input_complexity    0..1  — complejidad semántica del input
        response_length     0..1  — longitud relativa de la respuesta
        conflict_level      0..1  — nivel de conflicto detectado
        novelty             0..1  — qué tan nuevo fue el contenido
        tool_usage_cost     0..1  — costo de usar herramientas
        council_invoked     0|1   — si el Council corrió
        open_loops_created  0..1  — nuevos loops sin cerrar
        open_loops_closed   0..1  — loops cerrados en este turno
    """
    m = turn_metrics
    ic  = float(m.get("input_complexity",   0.3))
    rl  = float(m.get("response_length",    0.3))
    cf  = float(m.get("conflict_level",     0.0))
    nov = float(m.get("novelty",            0.3))
    tc  = float(m.get("tool_usage_cost",    0.0))
    ci  = float(m.get("council_invoked",    0.0))
    olc = float(m.get("open_loops_created", 0.0))
    olk = float(m.get("open_loops_closed",  0.0))

    # Costo total del turno — base + multiplicadores
    costo_base = 0.3 * ic + 0.2 * rl + 0.15 * cf + 0.1 * tc + 0.25 * ci

    # Energía baja con esfuerzo
    delta_energia = -costo_base * 0.08

    # Fatiga sube más lento que baja energía — acumulativa
    delta_fatiga = costo_base * 0.04

    # Claridad depende de energía restante y fatiga
    nueva_energia = state.energia + delta_energia
    nueva_fatiga  = state.fatiga  + delta_fatiga
    nueva_claridad = float(np.clip(
        0.5 * nueva_energia + 0.3 * (1.0 - nueva_fatiga) + 0.2 * (1.0 - state.carga_cognitiva),
        0.0, 1.0
    ))

    # Curiosidad: sube con novedad, baja con saturación
    delta_curiosidad = 0.04 * nov - 0.02 * ic
    if nueva_fatiga > 0.7:
        delta_curiosidad -= 0.03  # saturación inhibe curiosidad

    # Necesidad de cierre: loops abiertos la suben, cerrados la bajan
    delta_cierre = 0.06 * olc - 0.08 * olk

    # Tensión interna: conflicto la sube, cierre la baja
    delta_tension = 0.07 * cf - 0.05 * olk

    # Presión: conflicto la sube, respuesta exitosa la baja levemente
    delta_presion = 0.04 * cf - 0.02 * (1.0 - cf)

    # Estabilidad: conflicto y tensión la erosionan, cierre la recupera
    delta_estabilidad = -0.04 * cf - 0.02 * (nueva_fatiga > 0.7) + 0.03 * olk

    # Carga cognitiva: sube con complejidad y council, baja con cierre
    delta_carga = 0.05 * ic + 0.04 * ci - 0.04 * olk

    new_state = HomeostasisState(
        energia          = state.energia          + delta_energia,
        fatiga           = state.fatiga           + delta_fatiga,
        claridad         = nueva_claridad,
        activacion       = state.activacion,       # actualizado por vector E externamente
        valencia         = state.valencia,         # actualizado por vector E externamente
        presion          = state.presion          + delta_presion,
        curiosidad       = state.curiosidad       + delta_curiosidad,
        necesidad_cierre = state.necesidad_cierre + delta_cierre,
        tension_interna  = state.tension_interna  + delta_tension,
        estabilidad      = state.estabilidad      + delta_estabilidad,
        carga_cognitiva  = state.carga_cognitiva  + delta_carga,
        timestamp        = time.time(),
    )

    return new_state


# ============================================================
# RECUPERACIÓN OFFLINE
# ============================================================

def recover_state(
    state: HomeostasisState,
    delta_time: float  # en horas
) -> HomeostasisState:
    """
    Recuperación con el paso del tiempo sin conversación.
    Simula descanso, consolidación, restablecimiento basal.

    Tasas por hora:
        energia          +0.12 / hora
        fatiga           -0.08 / hora (baja lento)
        claridad         +0.10 / hora
        presion          -0.06 / hora
        tension_interna  -0.05 / hora
        carga_cognitiva  -0.10 / hora
        estabilidad      +0.05 / hora
        curiosidad       +0.08 / hora (crece con el tiempo sin estímulo)
        necesidad_cierre: no cambia sola — depende de loops
    """
    dt = max(0.0, delta_time)

    new_state = HomeostasisState(
        energia          = state.energia          + 0.12 * dt,
        fatiga           = state.fatiga           - 0.08 * dt,
        claridad         = state.claridad         + 0.10 * dt,
        activacion       = state.activacion,
        valencia         = state.valencia,
        presion          = state.presion          - 0.06 * dt,
        curiosidad       = state.curiosidad       + 0.08 * dt,
        necesidad_cierre = state.necesidad_cierre,
        tension_interna  = state.tension_interna  - 0.05 * dt,
        estabilidad      = state.estabilidad      + 0.05 * dt,
        carga_cognitiva  = state.carga_cognitiva  - 0.10 * dt,
        timestamp        = time.time(),
    )
    return new_state


# ============================================================
# VISTA DE COMPATIBILIDAD (COMPAT VIEW)
# ============================================================

def export_compat_view(state: HomeostasisState) -> dict:
    """
    Exporta HomeostasisState como el formato que espera alter_brain.py.
    Permite que el sistema viejo siga funcionando sin modificaciones.

    Retorna:
        v, a, p          — vector emocional legacy
        economia         — dict con atención, energía, tolerancia, expresión
        drives           — dict con curiosidad, expresion, conexion, eficiencia
        nivel_cansancio  — float
    """
    # Economia derivada
    economia = {
        "atencion":   float(np.clip(state.claridad, 0.0, 1.0)),
        "energia":    state.energia,
        "tolerancia": float(np.clip(state.estabilidad, 0.0, 1.0)),
        "expresion":  float(np.clip(1.0 - state.tension_interna, 0.0, 1.0)),
    }

    # Drives derivados
    drives = {
        "curiosidad":  state.curiosidad,
        "expresion":   float(np.clip(1.0 - state.tension_interna * 0.5, 0.0, 1.0)),
        "conexion":    float(np.clip(state.estabilidad * 0.8, 0.2, 1.0)),
        "eficiencia":  float(np.clip(state.necesidad_cierre, 0.0, 1.0)),
    }

    return {
        "v":               state.valencia,
        "a":               state.activacion,
        "p":               state.presion,
        "economia":        economia,
        "drives":          drives,
        "nivel_cansancio": state.fatiga,
    }


# ============================================================
# SNAPSHOT PARA WORKSPACE (Fase 1B)
# ============================================================

def homeostasis_snapshot(state: HomeostasisState) -> dict:
    """
    Exporta un snapshot compacto para que el Global Workspace
    lo use en TTL, decay y prioridad de items.

    Usado en Fase 1B — aquí solo se define el contrato.
    """
    return {
        "energia":          state.energia,
        "fatiga":           state.fatiga,
        "claridad":         state.claridad,
        "curiosidad":       state.curiosidad,
        "necesidad_cierre": state.necesidad_cierre,
        "tension_interna":  state.tension_interna,
        "presion":          state.presion,
        # Señales derivadas útiles para el workspace
        "modo_sugerido":    _inferir_modo(state),
        "council_warranted": _council_warranted(state),
    }


def _inferir_modo(state: HomeostasisState) -> str:
    """Infiere el modo cognitivo desde el estado homeostático."""
    if state.presion > 0.65 or state.tension_interna > 0.65:
        return "defensa"
    if state.curiosidad > 0.7 and state.energia > 0.5:
        return "exploracion"
    if state.necesidad_cierre > 0.65 and state.claridad > 0.5:
        return "sintesis"
    if state.fatiga > 0.65:
        return "conservacion"
    return "exploracion"


def _council_warranted(state: HomeostasisState) -> bool:
    """True si el estado justifica invocar el Inner Council."""
    return (
        state.tension_interna > 0.5 or
        state.presion > 0.6 or
        (state.curiosidad > 0.75 and state.energia > 0.5)
    )


# ============================================================
# SERIALIZACIÓN PARA REDIS
# ============================================================

def serialize(state: HomeostasisState) -> str:
    return json.dumps(state.to_dict(), ensure_ascii=False)


def deserialize(raw: str) -> HomeostasisState:
    return HomeostasisState.from_dict(json.loads(raw))


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    """
    Corre una suite mínima de tests de invariantes.
    Retorna lista de errores encontrados — vacía si todo pasa.
    """
    errors = []

    # Test 1: fatiga alta no mejora claridad
    s = HomeostasisState(energia=0.3, fatiga=0.9)
    s2 = apply_turn_impact(s, {"input_complexity": 0.1})
    if s2.claridad > s.claridad + 0.05:
        errors.append("FAIL: fatiga alta mejora claridad con input bajo")

    # Test 2: recovery sube energia
    s = HomeostasisState(energia=0.3, fatiga=0.8)
    s2 = recover_state(s, delta_time=2.0)
    if s2.energia <= s.energia:
        errors.append("FAIL: recovery no sube energia")
    if s2.fatiga >= s.fatiga:
        errors.append("FAIL: recovery no baja fatiga")

    # Test 3: presion extrema no coexiste con estabilidad alta
    s = HomeostasisState(presion=0.9, estabilidad=0.9)
    violations = s.validate()
    if not violations:
        errors.append("FAIL: presion extrema + estabilidad alta no detecta violación")

    # Test 4: compat view mantiene rango de economia
    s = HomeostasisState()
    compat = export_compat_view(s)
    for k, v in compat["economia"].items():
        if not (0.0 <= v <= 1.0):
            errors.append(f"FAIL: compat view economia.{k} fuera de rango: {v}")

    # Test 5: build_homeostasis_state desde legacy
    legacy = load_legacy_state(v=0.5, a=0.6, p=0.2)
    state = build_homeostasis_state(legacy)
    if state.valencia != 0.5:
        errors.append(f"FAIL: build_homeostasis_state no mapea valencia correctamente")
    if not (-1.0 <= state.valencia <= 1.0):
        errors.append("FAIL: valencia fuera de rango -1..1")

    return errors


if __name__ == "__main__":
    print("=== alter_homeostasis.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Snapshot desde legacy ===")
    legacy = load_legacy_state(v=0.4, a=0.5, p=0.3,
                                economia={"atencion": 0.7, "energia": 0.75,
                                          "tolerancia": 0.8, "expresion": 0.6},
                                drives={"curiosidad": 1.0, "expresion": 1.0,
                                        "conexion": 0.2, "eficiencia": 0.3})
    state = build_homeostasis_state(legacy)
    snap  = homeostasis_snapshot(state)
    compat = export_compat_view(state)
    print(f"Estado: energia={state.energia:.2f} fatiga={state.fatiga:.2f} "
          f"claridad={state.claridad:.2f} valencia={state.valencia:.2f}")
    print(f"Modo sugerido: {snap['modo_sugerido']}")
    print(f"Council warranted: {snap['council_warranted']}")
    print(f"Compat V={compat['v']:.2f} A={compat['a']:.2f} P={compat['p']:.2f}")
    violations = state.validate()
    print(f"Violaciones: {violations if violations else 'ninguna'}")
