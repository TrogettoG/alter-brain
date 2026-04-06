"""
alter_memory.py — Memory Layers de AlterB3 (Fase 3)

Separación explícita de memoria en 4 capas funcionales.
Working Memory ya vive en GlobalWorkspace (Fase 1B).

Capas:
    EpisodicMemory   — eventos con contexto emocional y outcome
    SemanticMemory   — conocimiento del usuario, mundo e ideas propias
    ProceduralMemory — patrones aprendidos (Opción A: solo error alto o feedback)
    IdentityMemory   — narrativa, principios, pizarra, estilo

Principio de migración:
    Adaptadores sobre Redis existente — no se reescriben keys.
    El sistema viejo sigue funcionando intacto.

Redis keys que lee (sin modificar):
    alter:episodio:*         → EpisodicMemory
    alter:episodios:idx      → EpisodicMemory
    alter:autobiografia      → IdentityMemory
    alter:config:pizarra     → IdentityMemory
    alter:mundo:nodos        → SemanticMemory
    alter:mundo:aristas      → SemanticMemory
    alter:interlocutor:*     → SemanticMemory
    alter:ideas              → SemanticMemory

Redis keys propias (nuevas):
    alter:memory:procedural  — PatternStore serializado

ProceduralMemory — Opción A:
    Aprende SOLO cuando:
    1. prediction_error > 0.6
    2. feedback explícito del usuario ("no era eso", "más concreto", etc.)
    3. señal muy clara de éxito
    No aprende de cada turno — evita drift prematuro.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Episode:
    id:           str
    tema:         str
    tension:      str
    outcome:      str      # "resuelto" | "pendiente" | "abierto"
    relevancia:   float
    sintesis:     str
    vector:       list     # [V, A, P]
    timestamp:    str
    interlocutor: str = "gian"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SemanticNode:
    id:        str
    tipo:      str    # "persona" | "idea" | "concepto" | "lugar" | "proyecto"
    nombre:    str
    atributos: dict = field(default_factory=dict)
    peso:      float = 0.5
    updated:   str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProceduralPattern:
    id:               str
    trigger:          str    # descripción del trigger
    response:         str    # qué hace ALTER cuando se activa
    success_rate:     float  # 0..1 — historial de éxito
    uses:             int    # cuántas veces se usó
    last_used:        str
    context_signature: dict = field(default_factory=dict)
    # Ejemplo de context_signature:
    # {
    #   "user_intent": "quiere_accion",
    #   "prediction_error_high": True,
    #   "workspace_has_goal": True,
    #   "homeostasis_mode": "presion"
    # }

    def to_dict(self) -> dict:
        return asdict(self)

    def matches(self, context: dict, threshold: float = 0.5) -> float:
        """
        Calcula qué tan bien este patrón coincide con el contexto actual.
        Retorna score 0..1.
        """
        if not self.context_signature:
            return 0.2  # patrón sin firma — baja relevancia por defecto
        sig = self.context_signature
        matches = 0
        total   = len(sig)
        for k, v in sig.items():
            if k in context and context[k] == v:
                matches += 1
        return matches / total if total > 0 else 0.0


@dataclass
class IdentityMemory:
    narrativa:   str = ""
    principios:  list = field(default_factory=list)
    decisiones:  list = field(default_factory=list)  # pizarra
    estilo:      dict = field(default_factory=dict)
    updated:     str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# SEÑALES DE FEEDBACK EXPLÍCITO
# ============================================================

FEEDBACK_NEGATIVO = [
    "no era eso", "no es lo que", "no me sirve", "no lo necesito",
    "eso no", "dejá la teoría", "más concreto", "sin rodeos",
    "al punto", "no me interesa", "no era lo que pedí",
    "te equivocaste", "no coincido", "eso no ayuda",
]

FEEDBACK_POSITIVO = [
    "sí, eso era", "justo eso", "perfecto", "exacto",
    "eso quería", "genial", "me cerró", "me copa",
    "muy bien", "lo clavaste", "dale así",
]

def detectar_feedback(texto: str) -> str:
    """
    Detecta señal de feedback explícito del usuario.
    Retorna: "positivo" | "negativo" | "ninguno"
    """
    tl = texto.lower()
    if any(s in tl for s in FEEDBACK_NEGATIVO):
        return "negativo"
    if any(s in tl for s in FEEDBACK_POSITIVO):
        return "positivo"
    return "ninguno"


# ============================================================
# EPISODIC MEMORY
# ============================================================

class EpisodicMemory:
    """
    Adaptador sobre las keys Redis existentes de episodios.
    No modifica las keys — solo las lee con la interfaz correcta.
    """

    def __init__(self, redis_client):
        self._redis = redis_client

    def get_recent(self, n: int = 5) -> list[Episode]:
        """Últimos N episodios."""
        if not self._redis:
            return []
        try:
            keys = self._redis.lrange("alter:episodios:idx", 0, n - 1)
            episodes = []
            for k in (keys or []):
                raw = self._redis.get(k)
                if raw:
                    d = json.loads(raw)
                    episodes.append(Episode(
                        id           = k,
                        tema         = d.get("tema", ""),
                        tension      = d.get("tension", ""),
                        outcome      = d.get("outcome", "abierto"),
                        relevancia   = float(d.get("relevancia", 0.5)),
                        sintesis     = d.get("sintesis", ""),
                        vector       = d.get("vector", [0.3, 0.5, 0.3]),
                        timestamp    = d.get("t", ""),
                        interlocutor = d.get("interlocutor", "gian"),
                    ))
            return episodes
        except Exception:
            return []

    def get_by_relevance(self, threshold: float = 0.7) -> list[Episode]:
        """Episodios con relevancia >= threshold."""
        return [e for e in self.get_recent(20) if e.relevancia >= threshold]

    def search(self, query: str, n: int = 5) -> list[Episode]:
        """Búsqueda simple por tema o síntesis."""
        query_lower = query.lower()
        all_episodes = self.get_recent(20)
        matches = [
            e for e in all_episodes
            if query_lower in e.tema.lower() or query_lower in e.sintesis.lower()
        ]
        return matches[:n]

    def get_pending(self) -> list[Episode]:
        """Episodios con outcome pendiente."""
        return [e for e in self.get_recent(20) if e.outcome == "pendiente"]

    def snapshot_str(self, n: int = 3) -> str:
        """Formato compacto para incluir en prompt."""
        episodes = self.get_recent(n)
        if not episodes:
            return "Sin episodios recientes."
        lines = []
        for e in episodes:
            estado = "✓" if e.outcome == "resuelto" else "…" if e.outcome == "pendiente" else "○"
            lines.append(f"{estado} [{e.timestamp[:10]}] {e.tema}: {e.sintesis[:80]}")
        return "\n".join(lines)


# ============================================================
# SEMANTIC MEMORY
# ============================================================

class SemanticMemory:
    """
    Adaptador sobre grafo del mundo, modelo del interlocutor e ideas propias.
    """

    def __init__(self, redis_client):
        self._redis = redis_client

    def get_all_nodes(self) -> list[SemanticNode]:
        if not self._redis:
            return []
        try:
            raw = self._redis.get("alter:mundo:nodos")
            if not raw:
                return []
            nodes_raw = json.loads(raw)
            return [
                SemanticNode(
                    id        = n.get("id", str(uuid.uuid4())[:8]),
                    tipo      = n.get("tipo", "concepto"),
                    nombre    = n.get("nombre", ""),
                    atributos = n.get("atributos", {}),
                    peso      = float(n.get("peso", 0.5)),
                    updated   = n.get("updated", ""),
                )
                for n in nodes_raw
            ]
        except Exception:
            return []

    def get_node(self, nombre: str) -> Optional[SemanticNode]:
        nodes = self.get_all_nodes()
        nombre_lower = nombre.lower()
        return next(
            (n for n in nodes if n.nombre.lower() == nombre_lower),
            None
        )

    def get_by_tipo(self, tipo: str) -> list[SemanticNode]:
        return [n for n in self.get_all_nodes() if n.tipo == tipo]

    def get_ideas_propias(self, n: int = 5) -> list[dict]:
        if not self._redis:
            return []
        try:
            raw = self._redis.get("alter:ideas")
            if not raw:
                return []
            ideas = json.loads(raw)
            return ideas[-n:] if isinstance(ideas, list) else []
        except Exception:
            return []

    def get_interlocutor_model(self, interlocutor_id: str) -> dict:
        if not self._redis:
            return {}
        try:
            raw = self._redis.get(f"alter:persona:{interlocutor_id}")
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def snapshot_str(self, max_nodes: int = 6) -> str:
        """Formato compacto para prompt — nodos top por peso."""
        nodes = sorted(self.get_all_nodes(), key=lambda x: x.peso, reverse=True)
        if not nodes:
            return "Grafo vacío."
        top = nodes[:max_nodes]
        lines = [f"[{n.tipo}] {n.nombre} (peso:{n.peso:.1f})" for n in top]
        return "\n".join(lines)


# ============================================================
# PROCEDURAL MEMORY
# ============================================================

class ProceduralMemory:
    """
    Patrones aprendidos — políticas implícitas que se activan sin deliberación.

    Opción A: solo aprende cuando:
        1. prediction_error > THRESHOLD_ERROR
        2. feedback explícito del usuario
        3. señal muy clara de éxito/fracaso

    No aprende de cada turno — evita drift prematuro.
    """

    THRESHOLD_ERROR   = 0.60
    MAX_PATTERNS      = 30
    DECAY_RATE        = 0.02   # success_rate baja levemente con el tiempo si no se usa
    REDIS_KEY         = "alter:memory:procedural"

    def __init__(self, redis_client):
        self._redis = redis_client
        self._patterns: list[ProceduralPattern] = []
        self._load()

    def _load(self):
        if not self._redis:
            return
        try:
            raw = self._redis.get(self.REDIS_KEY)
            if raw:
                patterns_raw = json.loads(raw)
                self._patterns = [
                    ProceduralPattern(**p) for p in patterns_raw
                ]
        except Exception:
            self._patterns = []

    def _save(self):
        if not self._redis:
            return
        try:
            self._redis.set(
                self.REDIS_KEY,
                json.dumps([p.to_dict() for p in self._patterns], ensure_ascii=False)
            )
        except Exception:
            pass

    def get_matching(
        self,
        context: dict,
        threshold: float = 0.5,
        n: int = 3
    ) -> list[ProceduralPattern]:
        """
        Retorna los N patrones más relevantes al contexto actual.
        Solo retorna patrones con success_rate > 0.4 — no aplica patrones con historial malo.
        """
        scored = [
            (p, p.matches(context, threshold))
            for p in self._patterns
            if p.success_rate > 0.40
        ]
        scored = [(p, s) for p, s in scored if s >= threshold]
        scored.sort(key=lambda x: x[1] * x[0].success_rate, reverse=True)
        return [p for p, _ in scored[:n]]

    def add_pattern(
        self,
        trigger: str,
        response: str,
        context_signature: dict,
        initial_success: float = 0.5,
    ) -> ProceduralPattern:
        """
        Agrega un nuevo patrón aprendido.
        Si ya existe uno con el mismo trigger, llama reinforce() en lugar de crear duplicado.
        """
        # Buscar si ya existe un patrón similar
        existing = next(
            (p for p in self._patterns if p.trigger.lower() == trigger.lower()),
            None
        )
        if existing:
            return self.reinforce(existing.id, success=initial_success > 0.5)

        if len(self._patterns) >= self.MAX_PATTERNS:
            # Eliminar el patrón con menor success_rate * uses
            worst = min(self._patterns, key=lambda p: p.success_rate * max(p.uses, 1))
            self._patterns.remove(worst)

        pattern = ProceduralPattern(
            id                = str(uuid.uuid4())[:8],
            trigger           = trigger,
            response          = response,
            success_rate      = float(np.clip(initial_success, 0.0, 1.0)),
            uses              = 1,
            last_used         = time.strftime("%Y-%m-%d %H:%M"),
            context_signature = context_signature,
        )
        self._patterns.append(pattern)
        self._save()
        return pattern

    def reinforce(self, pattern_id: str, success: bool) -> Optional[ProceduralPattern]:
        """
        Sube o baja el success_rate de un patrón.
        success=True → +0.10 (hasta 0.95)
        success=False → -0.15 (hasta 0.10)
        """
        pattern = next((p for p in self._patterns if p.id == pattern_id), None)
        if not pattern:
            return None

        if success:
            pattern.success_rate = float(np.clip(pattern.success_rate + 0.10, 0.0, 0.95))
        else:
            pattern.success_rate = float(np.clip(pattern.success_rate - 0.15, 0.10, 1.0))

        pattern.uses      += 1
        pattern.last_used  = time.strftime("%Y-%m-%d %H:%M")
        self._save()
        return pattern

    def learn_from_error(
        self,
        prediction_error: float,
        user_intent: str,
        alter_action: str,
        context_signature: dict,
    ) -> Optional[ProceduralPattern]:
        """
        Aprende un nuevo patrón cuando el error de predicción es alto.
        Solo actúa si prediction_error > THRESHOLD_ERROR.
        """
        if prediction_error <= self.THRESHOLD_ERROR:
            return None

        trigger  = f"usuario quiere {user_intent} pero ALTER da respuesta incorrecta"
        response = f"cuando detecto {user_intent}, ajustar hacia: {alter_action}"

        sig = dict(context_signature)
        sig["prediction_error_high"] = True
        sig["user_intent"] = user_intent

        return self.add_pattern(trigger, response, sig, initial_success=0.40)

    def learn_from_feedback(
        self,
        feedback: str,
        last_alter_action: str,
        context_signature: dict,
    ) -> Optional[ProceduralPattern]:
        """
        Aprende o refuerza desde feedback explícito del usuario.
        feedback: "positivo" | "negativo"
        """
        if feedback == "ninguno":
            return None

        trigger  = f"contexto similar a: {context_signature.get('user_intent', '?')}"
        response = last_alter_action[:100]
        success  = feedback == "positivo"

        existing = next(
            (p for p in self._patterns
             if p.context_signature.get("user_intent") ==
                context_signature.get("user_intent")),
            None
        )
        if existing:
            return self.reinforce(existing.id, success=success)

        if feedback == "negativo":
            return self.add_pattern(
                trigger, f"EVITAR: {response}", context_signature,
                initial_success=0.20
            )
        else:
            return self.add_pattern(
                trigger, f"REPETIR: {response}", context_signature,
                initial_success=0.75
            )

    def snapshot_str(self, context: dict, n: int = 2) -> str:
        """Patrones relevantes al contexto actual."""
        matching = self.get_matching(context, n=n)
        if not matching:
            return ""
        lines = []
        for p in matching:
            lines.append(
                f"Patrón aprendido: '{p.trigger}' → '{p.response[:60]}' "
                f"(éxito:{p.success_rate:.2f} usos:{p.uses})"
            )
        return "\n".join(lines)


# ============================================================
# IDENTITY MEMORY
# ============================================================

class IdentityMemoryStore:
    """
    Adaptador sobre autobiografía y pizarra.
    """

    def __init__(self, redis_client):
        self._redis = redis_client

    def get(self) -> IdentityMemory:
        narrativa  = ""
        principios = []
        decisiones = []

        if self._redis:
            try:
                raw_auto = self._redis.get("alter:autobiografia")
                if raw_auto:
                    auto = json.loads(raw_auto)
                    narrativa = auto.get("narrativa", "")
            except Exception:
                pass

            try:
                raw_pizarra = self._redis.get("alter:config:pizarra")
                if raw_pizarra:
                    pizarra = json.loads(raw_pizarra)
                    decisiones = pizarra.get("decisiones", [])
                    principios = pizarra.get("principios", [])
            except Exception:
                pass

        return IdentityMemory(
            narrativa   = narrativa,
            principios  = principios,
            decisiones  = decisiones,
            updated     = time.strftime("%Y-%m-%d %H:%M"),
        )

    def snapshot_str(self) -> str:
        identity = self.get()
        lines = []
        if identity.narrativa:
            lines.append(f"Narrativa: {identity.narrativa[:150]}")
        if identity.decisiones:
            lines.append(f"Decisiones inamovibles: {len(identity.decisiones)} activas")
        return "\n".join(lines) if lines else "Identidad en construcción."


# ============================================================
# MEMORY SYSTEM — PUNTO DE ACCESO ÚNICO
# ============================================================

class MemorySystem:
    """
    Punto de acceso único a todas las capas de memoria.
    Working Memory vive en GlobalWorkspace (Fase 1B) — no se duplica acá.
    """

    def __init__(self, redis_client):
        self.episodic   = EpisodicMemory(redis_client)
        self.semantic   = SemanticMemory(redis_client)
        self.procedural = ProceduralMemory(redis_client)
        self.identity   = IdentityMemoryStore(redis_client)

    def query(self, tipo: str, **kwargs) -> list:
        """Enruta queries al tipo correcto."""
        if tipo == "episodic":
            n = kwargs.get("n", 5)
            query_str = kwargs.get("query")
            if query_str:
                return self.episodic.search(query_str, n)
            return self.episodic.get_recent(n)

        elif tipo == "semantic":
            node_tipo = kwargs.get("node_tipo")
            if node_tipo:
                return self.semantic.get_by_tipo(node_tipo)
            return self.semantic.get_all_nodes()

        elif tipo == "procedural":
            context = kwargs.get("context", {})
            return self.procedural.get_matching(context)

        elif tipo == "identity":
            return [self.identity.get()]

        return []

    def snapshot_for_prompt(
        self,
        procedural_context: Optional[dict] = None,
        n_episodes: int = 3,
        n_nodes: int = 5,
    ) -> str:
        """
        Reemplaza los _formatear_* de alter_brain.py.
        Entrega exactamente lo que cada capa considera relevante ahora.

        En Fase 3 este método se usa en lugar de los formateos individuales.
        Por ahora convive con ellos durante la transición.
        """
        sections = []

        # Identity — siempre presente
        id_str = self.identity.snapshot_str()
        if id_str:
            sections.append(f"[IDENTIDAD]\n{id_str}")

        # Episodic — últimos N relevantes
        ep_str = self.episodic.snapshot_str(n_episodes)
        if ep_str and ep_str != "Sin episodios recientes.":
            sections.append(f"[EPISODIOS]\n{ep_str}")

        # Semantic — nodos top
        sem_str = self.semantic.snapshot_str(n_nodes)
        if sem_str and sem_str != "Grafo vacío.":
            sections.append(f"[MUNDO]\n{sem_str}")

        # Procedural — solo si hay patrones relevantes
        if procedural_context:
            proc_str = self.procedural.snapshot_str(procedural_context, n=2)
            if proc_str:
                sections.append(f"[PATRONES APRENDIDOS]\n{proc_str}")

        return "\n\n".join(sections) if sections else "Sin memoria disponible."

    def learn(
        self,
        prediction_error: float,
        user_intent: str,
        alter_action: str,
        user_feedback: str,
        context_signature: dict,
    ):
        """
        Punto único de aprendizaje procedural.
        Opción A: solo aprende si hay error alto o feedback explícito.
        """
        if prediction_error > ProceduralMemory.THRESHOLD_ERROR:
            self.procedural.learn_from_error(
                prediction_error, user_intent, alter_action, context_signature
            )

        if user_feedback != "ninguno":
            self.procedural.learn_from_feedback(
                user_feedback, alter_action, context_signature
            )


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    # Test 1: ProceduralPattern.matches()
    p = ProceduralPattern(
        id="t1", trigger="test", response="resp",
        success_rate=0.7, uses=3, last_used="2026-04-01",
        context_signature={
            "user_intent": "quiere_accion",
            "prediction_error_high": True,
            "homeostasis_mode": "presion"
        }
    )
    context_match = {
        "user_intent": "quiere_accion",
        "prediction_error_high": True,
        "homeostasis_mode": "presion"
    }
    context_no_match = {
        "user_intent": "quiere_exploracion",
        "prediction_error_high": False,
    }
    score_match    = p.matches(context_match)
    score_no_match = p.matches(context_no_match)
    if score_match < 0.9:
        errors.append(f"FAIL: matches perfecto debería ser ~1.0, got {score_match}")
    if score_no_match > 0.2:
        errors.append(f"FAIL: matches sin coincidencia debería ser bajo, got {score_no_match}")

    # Test 2: ProceduralMemory sin Redis
    pm = ProceduralMemory(None)
    pat = pm.add_pattern(
        "usuario quiere acción pero ALTER analiza",
        "dar código primero, análisis después",
        {"user_intent": "quiere_accion", "prediction_error_high": True},
        initial_success=0.40
    )
    if not pat:
        errors.append("FAIL: add_pattern retornó None")
    if len(pm._patterns) != 1:
        errors.append(f"FAIL: esperaba 1 patrón, got {len(pm._patterns)}")

    # Test 3: reinforce sube y baja
    sr_antes = pat.success_rate
    pm.reinforce(pat.id, success=True)
    p_updated = next(p for p in pm._patterns if p.id == pat.id)
    if p_updated.success_rate <= sr_antes:
        errors.append("FAIL: reinforce(True) no subió success_rate")
    sr_tras_true = p_updated.success_rate
    pm.reinforce(pat.id, success=False)
    pm.reinforce(pat.id, success=False)
    p_updated2 = next(p for p in pm._patterns if p.id == pat.id)
    if p_updated2.success_rate >= sr_tras_true:
        errors.append("FAIL: dos reinforce(False) no bajaron success_rate")

    # Test 4: learn_from_error — solo si > threshold
    result_bajo  = pm.learn_from_error(0.3, "quiere_exploracion", "acción X", {})
    result_alto  = pm.learn_from_error(0.8, "quiere_cierre", "análisis largo", {})
    if result_bajo is not None:
        errors.append("FAIL: learn_from_error con error bajo debería retornar None")
    if result_alto is None:
        errors.append("FAIL: learn_from_error con error alto debería crear patrón")

    # Test 5: learn_from_feedback positivo y negativo
    pm2 = ProceduralMemory(None)
    pm2.learn_from_feedback("positivo", "dar código directo", {"user_intent": "quiere_accion"})
    pm2.learn_from_feedback("negativo", "dar análisis largo", {"user_intent": "quiere_cierre"})
    if len(pm2._patterns) < 2:
        errors.append(f"FAIL: esperaba 2 patrones de feedback, got {len(pm2._patterns)}")

    # Test 6: get_matching filtra por success_rate
    pm3 = ProceduralMemory(None)
    pm3.add_pattern("trigger malo", "resp", {"user_intent": "quiere_accion"}, initial_success=0.2)
    pm3.add_pattern("trigger bueno", "resp2", {"user_intent": "quiere_accion"}, initial_success=0.8)
    matching = pm3.get_matching({"user_intent": "quiere_accion"})
    sr_values = [p.success_rate for p in matching]
    if any(sr < 0.4 for sr in sr_values):
        errors.append("FAIL: get_matching devolvió patrón con success_rate < 0.4")

    # Test 7: MemorySystem sin Redis — no explota
    ms = MemorySystem(None)
    snap = ms.snapshot_for_prompt(procedural_context={"user_intent": "quiere_accion"})
    if snap is None:
        errors.append("FAIL: snapshot_for_prompt retornó None sin Redis")

    # Test 8: detectar_feedback
    if detectar_feedback("no era eso") != "negativo":
        errors.append("FAIL: 'no era eso' no detectó feedback negativo")
    if detectar_feedback("perfecto, eso quería") != "positivo":
        errors.append("FAIL: 'perfecto' no detectó feedback positivo")
    if detectar_feedback("seguí contándome") != "ninguno":
        errors.append("FAIL: frase neutral detectó feedback")

    return errors


if __name__ == "__main__":
    print("=== alter_memory.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo MemorySystem (sin Redis) ===")
    ms = MemorySystem(None)

    # Simular aprendizaje procedural
    ms.learn(
        prediction_error  = 0.75,
        user_intent       = "quiere_accion",
        alter_action      = "dar análisis conceptual",
        user_feedback     = "negativo",
        context_signature = {
            "user_intent":         "quiere_accion",
            "prediction_error_high": True,
            "homeostasis_mode":    "presion"
        }
    )
    ms.learn(
        prediction_error  = 0.2,
        user_intent       = "quiere_exploracion",
        alter_action      = "preguntar y explorar juntos",
        user_feedback     = "positivo",
        context_signature = {
            "user_intent":      "quiere_exploracion",
            "homeostasis_mode": "exploracion"
        }
    )

    context_actual = {
        "user_intent":         "quiere_accion",
        "prediction_error_high": True,
        "homeostasis_mode":    "presion"
    }
    matching = ms.procedural.get_matching(context_actual)
    print(f"\nPatrones relevantes al contexto actual ({len(matching)}):")
    for p in matching:
        print(f"  [{p.success_rate:.2f}] {p.trigger[:60]}")
        print(f"         → {p.response[:60]}")

    snap = ms.snapshot_for_prompt(procedural_context=context_actual)
    print(f"\nSnapshot para prompt:\n{snap}")
