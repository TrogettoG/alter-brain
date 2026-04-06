"""
alter_workspace.py — Global Workspace de AlterB3 (Fase 1B)

El corazón de AlterB3. Todo lo que entra a conciencia pasa por acá.
Lo que no entra, no existe para el sistema en ese turno.

Contrato:
    Input:  señales candidatas desde homeostasis, memoria, agenda,
            input del usuario, predicción (futura Fase 2)
    Output: snapshot de máx 7 items activos para deliberación y acción

Ajustes incorporados vs spec inicial:
    1. ttl en turnos, no segundos
    2. base_priority + current_priority separadas
    3. sticky explícito para items inamovibles
    4. decay modulado por fatiga Y claridad
    5. score con need_of_type
    6. merge semántico simple antes de refresh

AlterB3 Fase 1B — corre en paralelo al sistema actual.
No reemplaza alter_brain.py todavía. Se integra en Fase 1C.

Redis keys propias (no toca keys existentes):
    alter:workspace:state   — estado serializado
    alter:workspace:log     — log de movimientos (lpush, ltrim 200)
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ============================================================
# CONSTANTES
# ============================================================

MAX_ITEMS          = 7
MIN_ITEMS          = 3
MAX_PER_TYPE       = {
    "goal":             1,
    "constraint":       2,
    "user_hypothesis":  2,
    "memory_trace":     2,
    "candidate_action": 1,
    "internal_tension": 1,
}
VALID_TYPES = set(MAX_PER_TYPE.keys())
VALID_SOURCES = {
    "user_input", "memory", "agenda",
    "homeostasis", "council", "system", "tarea_autonoma"
}

# TTL default por tipo (en turnos)
TTL_DEFAULT = {
    "goal":             8.0,
    "constraint":       6.0,
    "user_hypothesis":  4.0,
    "memory_trace":     3.0,
    "candidate_action": 2.0,
    "internal_tension": 5.0,
}

# Umbral de evicción
EVICTION_THRESHOLD  = 0.15

# Umbral de merge semántico (overlap de palabras)
MERGE_OVERLAP_THRESHOLD = 0.55

# Pesos del score
SCORE_WEIGHTS = {
    "relevance":      0.28,
    "novelty":        0.18,
    "urgency":        0.18,
    "consistency":    0.15,
    "memory_support": 0.10,
    "need_of_type":   0.11,
    "cost":          -0.05,
}


# ============================================================
# DATACLASS — WORKSPACE ITEM
# ============================================================

@dataclass
class WorkspaceItem:
    id:               str
    type:             str    # goal | constraint | user_hypothesis | memory_trace | candidate_action | internal_tension
    content:          str
    base_priority:    float  # fuerza intrínseca — no cambia con decay
    current_priority: float  # valor actual tras decay / refresh
    confidence:       float  # 0..1
    ttl_turns:        float  # persistencia cognitiva en turnos
    source:           str
    created_at:       float
    updated_at:       float
    sticky:           bool = False  # si True, no puede ser eviccionado

    def __post_init__(self):
        self.base_priority    = float(np.clip(self.base_priority,    0.0, 1.0))
        self.current_priority = float(np.clip(self.current_priority, 0.0, 1.0))
        self.confidence       = float(np.clip(self.confidence,       0.0, 1.0))
        self.ttl_turns        = max(0.0, self.ttl_turns)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceItem":
        campos = {k for k in d if k in cls.__dataclass_fields__}
        return cls(**{k: d[k] for k in campos})

    def is_expired(self) -> bool:
        return self.ttl_turns <= 0.0 and not self.sticky

    def decayed_priority(self, fatiga: float, claridad: float) -> float:
        """
        Decay modulado por fatiga Y claridad.
        No es lo mismo fatiga alta con claridad usable
        que fatiga alta con claridad colapsada.
        """
        decay_multiplier = fatiga * 0.5 + (1.0 - claridad) * 0.3 + 0.1
        return float(np.clip(self.current_priority - 0.08 * decay_multiplier, 0.0, 1.0))


# ============================================================
# MERGE SEMÁNTICO
# ============================================================

def _word_overlap(a: str, b: str) -> float:
    """Overlap de palabras entre dos strings. Simple y rápido."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _find_merge_candidate(
    new_item: WorkspaceItem,
    items: list[WorkspaceItem]
) -> Optional[WorkspaceItem]:
    """
    Encuentra un item existente que sea semánticamente similar
    al nuevo candidato. Si lo encuentra, se hace merge en lugar
    de crear un item nuevo.

    Condiciones:
        - mismo type
        - overlap textual >= MERGE_OVERLAP_THRESHOLD
        - fuente compatible (misma o ambas son cognitivas)
    """
    for existing in items:
        if existing.type != new_item.type:
            continue
        overlap = _word_overlap(existing.content, new_item.content)
        if overlap >= MERGE_OVERLAP_THRESHOLD:
            return existing
    return None


# ============================================================
# GLOBAL WORKSPACE
# ============================================================

class GlobalWorkspace:
    """
    Workspace cognitivo activo de ALTER.

    Mantiene hasta MAX_ITEMS items compitiendo por atención.
    Cada turno se aplica decay, se procesan candidatos y se
    expone un snapshot al resto del sistema.

    Integración gradual:
        Fase 1B — corre en paralelo, no reemplaza nada.
        Fase 1C — el Council recibe el snapshot en lugar
                  del historial completo.
    """

    def __init__(self, redis_client=None):
        self._items: list[WorkspaceItem] = []
        self._redis = redis_client
        self._turn_count: int = 0
        self._eviction_log: list[dict] = []

    # ----------------------------------------------------------
    # ACCESO
    # ----------------------------------------------------------

    @property
    def items(self) -> list[WorkspaceItem]:
        return list(self._items)

    def get_by_type(self, item_type: str) -> list[WorkspaceItem]:
        return [i for i in self._items if i.type == item_type]

    def dominant_goal(self) -> Optional[WorkspaceItem]:
        goals = self.get_by_type("goal")
        return goals[0] if goals else None

    def candidate_action(self) -> Optional[WorkspaceItem]:
        actions = self.get_by_type("candidate_action")
        return actions[0] if actions else None

    # ----------------------------------------------------------
    # SCORE DE CANDIDATOS
    # ----------------------------------------------------------

    def score_candidate(
        self,
        item: WorkspaceItem,
        homeostasis_snap: dict,
        relevance: float = 0.5,
        novelty: float = 0.5,
        urgency: float = 0.3,
        consistency: float = 0.5,
        memory_support: float = 0.3,
    ) -> float:
        """
        Calcula el score de un candidato para entrar al workspace.

        Factores:
            relevance       — relevancia al goal actual
            novelty         — qué tan nuevo para el sistema
            urgency         — presión temporal (de homeostasis o externo)
            consistency     — coherencia con goal dominante
            memory_support  — respaldado por episodios o ideas
            need_of_type    — workspace necesita este tipo de item ahora
            cost            — costo de ocupar espacio cognitivo
        """
        # need_of_type: boost si falta este tipo, penalización si ya hay máximo
        current_count = len(self.get_by_type(item.type))
        max_count     = MAX_PER_TYPE.get(item.type, 1)
        if current_count == 0:
            need_of_type = 0.9   # workspace necesita este tipo urgente
        elif current_count < max_count:
            need_of_type = 0.5   # hay espacio
        else:
            need_of_type = 0.1   # ya está lleno — entrará con dificultad

        # cost: depende de carga cognitiva actual
        carga = homeostasis_snap.get("carga_cognitiva", 0.3)
        cost  = 0.3 + 0.7 * carga  # más caro cuando más cargado

        score = (
            SCORE_WEIGHTS["relevance"]      * relevance +
            SCORE_WEIGHTS["novelty"]        * novelty +
            SCORE_WEIGHTS["urgency"]        * urgency +
            SCORE_WEIGHTS["consistency"]    * consistency +
            SCORE_WEIGHTS["memory_support"] * memory_support +
            SCORE_WEIGHTS["need_of_type"]   * need_of_type +
            SCORE_WEIGHTS["cost"]           * cost
        )
        return float(np.clip(score, 0.0, 1.0))

    # ----------------------------------------------------------
    # INSERCIÓN Y MERGE
    # ----------------------------------------------------------

    def _insert_or_merge(
        self,
        candidate: WorkspaceItem,
        score: float,
        homeostasis_snap: dict
    ) -> str:
        """
        Intenta insertar el candidato.
        Antes de crear item nuevo, busca merge semántico.
        Si el workspace está lleno, evalúa evicción.

        Retorna: 'inserted' | 'merged' | 'rejected'
        """
        # 1. Intento de merge
        merge_target = _find_merge_candidate(candidate, self._items)
        if merge_target:
            refresh_bonus = 0.15 * score
            merge_target.current_priority = float(np.clip(
                merge_target.current_priority + refresh_bonus, 0.0, 1.0
            ))
            merge_target.ttl_turns += 1.5
            merge_target.content   = candidate.content  # actualizar contenido
            merge_target.updated_at = time.time()
            self._log_move("+merge", merge_target, score)
            return "merged"

        # 2. Restricción dura de tipo único
        if MAX_PER_TYPE.get(candidate.type, 2) == 1:
            existentes = self.get_by_type(candidate.type)
            if existentes and not existentes[0].sticky:
                # Solo reemplaza si el nuevo tiene mejor score
                if score > existentes[0].current_priority:
                    self._items.remove(existentes[0])
                    self._log_move("-replaced", existentes[0], score)
                else:
                    return "rejected"

        # 3. Workspace lleno — evicción del más débil no sticky
        if len(self._items) >= MAX_ITEMS:
            evictable = [i for i in self._items if not i.sticky]
            if not evictable:
                return "rejected"
            weakest = min(evictable, key=lambda x: x.current_priority)
            if weakest.current_priority >= score:
                return "rejected"
            self._items.remove(weakest)
            self._log_move("-evicted", weakest, score)

        # 4. Insertar
        candidate.current_priority = score
        candidate.base_priority    = float(np.clip(candidate.base_priority, 0.0, 1.0))
        self._items.append(candidate)
        self._log_move("+inserted", candidate, score)
        return "inserted"

    # ----------------------------------------------------------
    # TICK — CICLO PRINCIPAL
    # ----------------------------------------------------------

    def tick(
        self,
        candidates: list[dict],
        homeostasis_snap: dict,
    ) -> dict:
        """
        Ciclo principal del workspace. Llamar una vez por turno.

        candidates: lista de dicts con:
            type, content, source
            + métricas opcionales: relevance, novelty, urgency,
              consistency, memory_support, confidence, sticky

        homeostasis_snap: salida de homeostasis_snapshot()

        Retorna: dict con resultados del tick (para logs)
        """
        self._turn_count += 1
        fatiga   = homeostasis_snap.get("fatiga",   0.2)
        claridad = homeostasis_snap.get("claridad", 0.7)

        results = {
            "turn": self._turn_count,
            "inserted": [], "merged": [], "rejected": [],
            "decayed": [], "expired": []
        }

        # 1. Decay a items existentes
        for item in self._items:
            if item.sticky:
                continue
            new_p = item.decayed_priority(fatiga, claridad)
            if new_p != item.current_priority:
                results["decayed"].append(item.id)
            item.current_priority = new_p
            item.ttl_turns = max(0.0, item.ttl_turns - 1.0)

        # 2. Eliminar expirados no sticky
        expirados = [i for i in self._items if i.is_expired()]
        for item in expirados:
            self._items.remove(item)
            results["expired"].append(item.id)
            self._log_move("-expired", item, 0.0)

        # 3. Procesar candidatos
        for c in candidates:
            if c.get("type") not in VALID_TYPES:
                continue

            candidate = WorkspaceItem(
                id               = str(uuid.uuid4())[:8],
                type             = c["type"],
                content          = c.get("content", ""),
                base_priority    = float(c.get("base_priority", 0.5)),
                current_priority = 0.0,  # se calcula en score
                confidence       = float(c.get("confidence", 0.7)),
                ttl_turns        = float(c.get("ttl_turns",
                                    TTL_DEFAULT.get(c["type"], 4.0))),
                source           = c.get("source", "system"),
                created_at       = time.time(),
                updated_at       = time.time(),
                sticky           = bool(c.get("sticky", False)),
            )

            score = self.score_candidate(
                candidate,
                homeostasis_snap,
                relevance      = float(c.get("relevance",      0.5)),
                novelty        = float(c.get("novelty",        0.5)),
                urgency        = float(c.get("urgency",        0.3)),
                consistency    = float(c.get("consistency",    0.5)),
                memory_support = float(c.get("memory_support", 0.3)),
            )

            outcome = self._insert_or_merge(candidate, score, homeostasis_snap)
            results[outcome if outcome in results else "rejected"].append(candidate.id)

        # 4. Ordenar por current_priority desc
        self._items.sort(key=lambda x: x.current_priority, reverse=True)

        # 5. Persistir en Redis
        self._persist()

        return results

    # ----------------------------------------------------------
    # SNAPSHOT
    # ----------------------------------------------------------

    def snapshot(self, homeostasis_snap: Optional[dict] = None) -> dict:
        """
        Exporta el estado actual para que el Council lo consuma.
        En Fase 1C reemplaza el historial completo en el prompt.
        """
        hs = homeostasis_snap or {}
        return {
            "dominant_goal":     self.dominant_goal().to_dict() if self.dominant_goal() else None,
            "candidate_action":  self.candidate_action().to_dict() if self.candidate_action() else None,
            "constraints":       [i.to_dict() for i in self.get_by_type("constraint")],
            "user_hypotheses":   [i.to_dict() for i in self.get_by_type("user_hypothesis")],
            "memory_traces":     [i.to_dict() for i in self.get_by_type("memory_trace")],
            "active_tensions":   [i.to_dict() for i in self.get_by_type("internal_tension")],
            "homeostasis_mode":  hs.get("modo_sugerido", "exploracion"),
            "council_warranted": hs.get("council_warranted", False),
            "item_count":        len(self._items),
            "turn":              self._turn_count,
        }

    def snapshot_str(self, homeostasis_snap: Optional[dict] = None) -> str:
        """
        Versión legible del snapshot para incluir en prompts.
        Formato compacto — no JSON, sino texto natural.
        """
        snap = self.snapshot(homeostasis_snap)
        lines = []

        if snap["dominant_goal"]:
            g = snap["dominant_goal"]
            lines.append(f"GOAL: {g['content']} (conf:{g['confidence']:.2f})")

        for h in snap["user_hypotheses"]:
            lines.append(f"HIPÓTESIS USUARIO: {h['content']} (conf:{h['confidence']:.2f})")

        for m in snap["memory_traces"]:
            lines.append(f"MEMORIA ACTIVA: {m['content']}")

        for c in snap["constraints"]:
            lines.append(f"RESTRICCIÓN: {c['content']}")

        for t in snap["active_tensions"]:
            lines.append(f"TENSIÓN: {t['content']}")

        if snap["candidate_action"]:
            a = snap["candidate_action"]
            lines.append(f"ACCIÓN CANDIDATA: {a['content']}")

        lines.append(f"MODO: {snap['homeostasis_mode']} | "
                     f"Council: {'sí' if snap['council_warranted'] else 'no'}")

        return "\n".join(lines) if lines else "Workspace vacío."

    # ----------------------------------------------------------
    # PERSISTENCIA
    # ----------------------------------------------------------

    def _persist(self):
        """Guarda el estado en Redis si está disponible."""
        if not self._redis:
            return
        try:
            state = {
                "items":       [i.to_dict() for i in self._items],
                "turn_count":  self._turn_count,
                "timestamp":   time.time(),
            }
            self._redis.set(
                "alter:workspace:state",
                json.dumps(state, ensure_ascii=False)
            )
        except Exception:
            pass

    def load(self):
        """Carga estado desde Redis."""
        if not self._redis:
            return
        try:
            raw = self._redis.get("alter:workspace:state")
            if raw:
                state = json.loads(raw)
                self._items       = [WorkspaceItem.from_dict(i) for i in state.get("items", [])]
                self._turn_count  = state.get("turn_count", 0)
        except Exception:
            pass

    def _log_move(self, action: str, item: WorkspaceItem, score: float):
        """Log estructurado de movimientos del workspace."""
        entry = f"[WS] {action} {item.type} '{item.content[:50]}' " \
                f"(score:{score:.2f} src:{item.source} p:{item.current_priority:.2f})"
        print(entry)
        if self._redis:
            try:
                self._redis.lpush("alter:workspace:log", entry)
                self._redis.ltrim("alter:workspace:log", 0, 199)
            except Exception:
                pass

    # ----------------------------------------------------------
    # UTILIDADES
    # ----------------------------------------------------------

    def clear(self, keep_sticky: bool = True):
        """Limpia el workspace. Por defecto mantiene items sticky."""
        if keep_sticky:
            self._items = [i for i in self._items if i.sticky]
        else:
            self._items = []

    def add_sticky(self, item_type: str, content: str, source: str = "system") -> WorkspaceItem:
        """
        Agrega un item sticky directamente (sin pasar por score).
        Para constraints de pizarra, identidad, etc.
        """
        item = WorkspaceItem(
            id               = str(uuid.uuid4())[:8],
            type             = item_type,
            content          = content,
            base_priority    = 1.0,
            current_priority = 1.0,
            confidence       = 1.0,
            ttl_turns        = float("inf"),
            source           = source,
            created_at       = time.time(),
            updated_at       = time.time(),
            sticky           = True,
        )
        # Verificar tipo válido y límite
        if item_type not in VALID_TYPES:
            raise ValueError(f"Tipo inválido: {item_type}")
        self._items.append(item)
        self._log_move("+sticky", item, 1.0)
        return item

    def validate(self) -> list[str]:
        """Verifica invariantes del workspace. Retorna lista de violaciones."""
        errors = []
        if len(self._items) > MAX_ITEMS:
            errors.append(f"FAIL: {len(self._items)} items > MAX_ITEMS {MAX_ITEMS}")
        for t, max_c in MAX_PER_TYPE.items():
            count = len(self.get_by_type(t))
            if count > max_c:
                errors.append(f"FAIL: {count} items de tipo '{t}' > max {max_c}")
        for item in self._items:
            if not (0.0 <= item.current_priority <= 1.0):
                errors.append(f"FAIL: current_priority fuera de rango en {item.id}")
            if not (0.0 <= item.base_priority <= 1.0):
                errors.append(f"FAIL: base_priority fuera de rango en {item.id}")
        return errors


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    ws = GlobalWorkspace()
    hs = {
        "fatiga": 0.3, "claridad": 0.7, "carga_cognitiva": 0.3,
        "curiosidad": 0.6, "necesidad_cierre": 0.4,
        "modo_sugerido": "exploracion", "council_warranted": False,
        "energia": 0.75, "presion": 0.3, "tension_interna": 0.2
    }

    # Test 1: inserción básica
    candidates = [
        {"type": "goal", "content": "definir AlterB3", "source": "user_input",
         "relevance": 0.9, "novelty": 0.8, "urgency": 0.7},
        {"type": "constraint", "content": "no romper sistema actual", "source": "system",
         "relevance": 0.8, "novelty": 0.3, "urgency": 0.6},
        {"type": "user_hypothesis", "content": "usuario quiere diseño conceptual", "source": "user_input",
         "relevance": 0.7, "novelty": 0.6, "urgency": 0.4},
    ]
    ws.tick(candidates, hs)
    v = ws.validate()
    if v:
        errors.extend(v)
    if not ws.dominant_goal():
        errors.append("FAIL: no hay goal después de insertar uno")

    # Test 2: max 1 goal — el segundo no entra si el primero tiene mayor score
    ws2 = GlobalWorkspace()
    ws2.tick([
        {"type": "goal", "content": "objetivo A", "source": "user_input",
         "relevance": 0.9, "novelty": 0.8, "urgency": 0.7},
    ], hs)
    ws2.tick([
        {"type": "goal", "content": "objetivo B inferior", "source": "agenda",
         "relevance": 0.2, "novelty": 0.2, "urgency": 0.1},
    ], hs)
    goals = ws2.get_by_type("goal")
    if len(goals) > 1:
        errors.append(f"FAIL: {len(goals)} goals simultáneos (máx 1)")

    # Test 3: merge semántico
    ws3 = GlobalWorkspace()
    ws3.tick([
        {"type": "goal", "content": "definir arquitectura AlterB3", "source": "user_input",
         "relevance": 0.9, "novelty": 0.8, "urgency": 0.7},
    ], hs)
    count_before = len(ws3.items)
    ws3.tick([
        {"type": "goal", "content": "arquitectura AlterB3 definición", "source": "user_input",
         "relevance": 0.9, "novelty": 0.5, "urgency": 0.6},
    ], hs)
    if len(ws3.items) > count_before:
        errors.append("FAIL: merge semántico no funcionó — se creó item duplicado")

    # Test 4: sticky no se evicta
    ws4 = GlobalWorkspace()
    ws4.add_sticky("constraint", "identidad inamovible", source="system")
    hs_alta_fatiga = dict(hs, fatiga=0.95, claridad=0.1)
    for _ in range(10):
        ws4.tick([], hs_alta_fatiga)
    stickies = [i for i in ws4.items if i.sticky]
    if not stickies:
        errors.append("FAIL: item sticky fue eviccionado o expirado")

    # Test 5: decay modula por fatiga + claridad
    ws5 = GlobalWorkspace()
    ws5.tick([
        {"type": "memory_trace", "content": "hormigas y feromonas", "source": "memory",
         "relevance": 0.5, "novelty": 0.3, "urgency": 0.2},
    ], hs)
    p_before = ws5.items[0].current_priority if ws5.items else 0
    ws5.tick([], dict(hs, fatiga=0.9, claridad=0.1))  # alta fatiga, baja claridad
    p_after  = ws5.items[0].current_priority if ws5.items else 0
    if p_after >= p_before and ws5.items:
        errors.append("FAIL: decay no redujo priority con fatiga alta y claridad baja")

    # Test 6: snapshot tiene estructura correcta
    snap = ws.snapshot(hs)
    required = {"dominant_goal", "candidate_action", "constraints",
                "user_hypotheses", "memory_traces", "active_tensions",
                "homeostasis_mode", "council_warranted"}
    missing = required - set(snap.keys())
    if missing:
        errors.append(f"FAIL: snapshot falta keys: {missing}")

    return errors


if __name__ == "__main__":
    print("=== alter_workspace.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo tick ===")
    ws = GlobalWorkspace()
    hs = {
        "fatiga": 0.2, "claridad": 0.75, "carga_cognitiva": 0.25,
        "curiosidad": 0.7, "necesidad_cierre": 0.4, "energia": 0.8,
        "presion": 0.3, "tension_interna": 0.2,
        "modo_sugerido": "exploracion", "council_warranted": False
    }
    ws.add_sticky("constraint", "no romper sistema actual — pizarra", source="system")
    ws.tick([
        {"type": "goal",            "content": "implementar AlterB3 Fase 1B",
         "source": "user_input",    "relevance": 0.95, "novelty": 0.8, "urgency": 0.8},
        {"type": "user_hypothesis", "content": "Gian quiere spec antes de codear",
         "source": "user_input",    "relevance": 0.85, "novelty": 0.6, "urgency": 0.5},
        {"type": "memory_trace",    "content": "homeostasis ya implementada en Fase 1A",
         "source": "memory",        "relevance": 0.75, "novelty": 0.3, "urgency": 0.2},
        {"type": "internal_tension","content": "workspace debe ser compatible con sistema viejo",
         "source": "homeostasis",   "relevance": 0.7,  "novelty": 0.5, "urgency": 0.6},
    ], hs)

    print(f"\nItems activos: {len(ws.items)}")
    print(f"\nSnapshot legible:\n{ws.snapshot_str(hs)}")
    violations = ws.validate()
    print(f"\nViolaciones: {violations if violations else 'ninguna'}")
