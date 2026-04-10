import asyncio
import time
import json
import os
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
import redis as redis_py
from upstash_redis import Redis as UpstashRedis

load_dotenv()

# --- Importar capas separadas ---
from alter_persona import (
    SYSTEM_PROMPT_ATENCION, SYSTEM_PROMPT_RESPUESTA,
    SYSTEM_PROMPT_UNIFICADO, PIZARRA_DEFAULT, limpiar_output
)
from alter_mind import (
    CAMPO_DEFAULT, DRIVES_DEFAULT, VENTANAS_AUTONOMIA,
    ECONOMIA_DEFAULT,
    actualizar_campo, campo_str, tono_emocional_str,
    inner_council as _inner_council,
    actualizar_drives_sesion, clasificar_modificacion,
    esta_dormido, estado_horario,
    consumir_economia, recuperar_economia,
    economia_str, economia_afecta_accion,
)

# --- AlterB3: Homeostasis y Workspace (Fase 1A + 1B) ---
try:
    from alter_homeostasis import (
        HomeostasisState, load_legacy_state,
        build_homeostasis_state, apply_turn_impact,
        recover_state, export_compat_view,
        homeostasis_snapshot, serialize as hs_serialize,
        deserialize as hs_deserialize,
    )
    from alter_workspace import GlobalWorkspace
    from alter_predictive import (
        PredictiveState, update as predictive_update,
        update_pre_response as predictive_pre,
        update_post_response as predictive_post,
        export_workspace_candidates, predictive_snapshot_str,
        serialize as pred_serialize, deserialize as pred_deserialize,
    )
    from alter_memory import (
        MemorySystem, detectar_feedback,
    )
    from alter_policy import PolicyArbiter, PolicyDecision
    from alter_metrics import MetricsCollector
    from alter_simulator import CounterfactualSimulator
    from alter_selfmodel import SelfModel, SelfModelBuilder, selfmodel_snapshot_str
    from alter_metalearning import MetaLearningEngine
    from alter_auditor import ArchitectureAuditor
    ALTERB3_ENABLED = True
except ImportError:
    ALTERB3_ENABLED = False

# --- Importar módulo de herramientas (opcional) ---
try:
    from alter_tools import ejecutar_herramienta, listar_skills
    TOOLS_DISPONIBLES = True
except ImportError:
    TOOLS_DISPONIBLES = False

# --- Configuración ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Falta GEMINI_API_KEY en las variables de entorno.")

UPSTASH_URL = os.environ.get("UPSTASH_REDIS_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_TOKEN")

client = genai.Client(api_key=GEMINI_API_KEY)


# --- Redis ---
def get_redis():
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        return None
    try:
        return UpstashRedis(url=UPSTASH_URL, token=UPSTASH_TOKEN)
    except Exception as e:
        print(f"[REDIS] Error conectando: {e}")
        return None


REDIS_KEY_PIZARRA = "alter:config:pizarra"
REDIS_KEY_INTERLOCUTOR = "alter:persona:{id}"
REDIS_KEY_IMPRESION = "alter:imp:{ts}"
REDIS_KEY_IMPRESIONES_IDX = "alter:imp:idx"
REDIS_KEY_IDEAS = "alter:ideas"
REDIS_KEY_PARAMS = "alter:params"
REDIS_KEY_SELF_MODS = "alter:self_mods"
REDIS_KEY_DRIVES = "alter:drives"
REDIS_KEY_EPISODIOS_IDX = "alter:episodios:idx"
REDIS_KEY_EPISODIO = "alter:episodio:{ts}"
REDIS_KEY_AGENDA = "alter:agenda"
REDIS_KEY_AUTOBIOGRAFIA = "alter:autobiografia"
REDIS_KEY_TRAZAS_IDX = "alter:trazas:idx"
REDIS_KEY_TRAZA = "alter:traza:{ts}"
REDIS_KEY_MUNDO_NODOS = "alter:mundo:nodos"
REDIS_KEY_MUNDO_ARISTAS = "alter:mundo:aristas"
REDIS_KEY_CONSECUENCIAS = "alter:consecuencias"
REDIS_KEY_REPUTACION = "alter:reputacion"

# PIZARRA_DEFAULT, prompts y lógica de mente vienen de alter_persona y alter_mind

PROMPT_UNIFICADO = """
ESTADO ACTUAL:
Valencia: {v:.2f} | Activación: {a:.2f} | Autoridad: {p:.2f}
Tono emocional: {tono_emocional}
Campo mental: {campo_mental}
Economía mental: {economia_mental}
Cansancio: {cansancio_activo} ({cansancio:.2f})

PIZARRA (decisiones inamovibles — solo para detectar colisiones):
{pizarra}

MEMORIA ACTIVA (lo que importa ahora — identidad, episodios, mundo, patrones):
{memoria_activa}

REPUTACIÓN POR TEMA (tu historial de aciertos — afecta tu confianza):
{reputacion}

DRIVES ACTUALES (tu motivación interna ahora):
Curiosidad: {curiosidad:.2f} | Expresión: {expresion:.2f} | Conexión: {conexion:.2f} | Eficiencia: {eficiencia:.2f}

CONTEXTO RECIENTE:
{motivos}

LO QUE DIJISTE ANTES (sé consistente):
{historial}

SECUENCIA ACTIVA: {secuencia}

{interlocutor} dijo: "{texto}"

RESPONDÉ ÚNICAMENTE EN JSON.
"""

# Prompt legacy — usado como fallback si AlterB3 no está disponible
PROMPT_UNIFICADO_LEGACY = """
ESTADO ACTUAL:
Valencia: {v:.2f} | Activación: {a:.2f} | Autoridad: {p:.2f}
Tono emocional: {tono_emocional}
Campo mental: {campo_mental}
Economía mental: {economia_mental}
Cansancio: {cansancio_activo} ({cansancio:.2f})

PIZARRA (decisiones inamovibles — solo para detectar colisiones):
{pizarra}

LO QUE SABÉS DE ESTA PERSONA (memoria acumulada):
{memoria_interlocutor}

NARRATIVA PROPIA (quién sos, cómo venís cambiando):
{autobiografia}

REPUTACIÓN POR TEMA (tu historial de aciertos — afecta tu confianza):
{reputacion}

IDEAS QUE TENÍAS DANDO VUELTAS (podés retomarlas si viene al caso):
{ideas}

EPISODIOS PREVIOS (momentos importantes que recordás):
{episodios}

AGENDA COGNITIVA (lo que tenés pendiente — retomar, preguntar, resolver):
{agenda}

MODELO DEL MUNDO (entidades y relaciones que conocés de Gian):
{mundo}

DRIVES ACTUALES (tu motivación interna ahora):
Curiosidad: {curiosidad:.2f} | Expresión: {expresion:.2f} | Conexión: {conexion:.2f} | Eficiencia: {eficiencia:.2f}

CONTEXTO RECIENTE:
{motivos}

LO QUE DIJISTE ANTES (sé consistente):
{historial}

SECUENCIA ACTIVA: {secuencia}

{interlocutor} dijo: "{texto}"

RESPONDÉ ÚNICAMENTE EN JSON.
"""



class AlterBrain:
    def __init__(self, interlocutor_id: str = "gian"):
        self.v_base, self.a_base, self.p_base = 0.2, 0.4, 0.3
        self.v, self.a, self.p = self.v_base, self.a_base, self.p_base
        self.lambda_drift = 0.05
        self.last_update = time.time()
        self.cansancio_activo = True
        self.nivel_cansancio = 0.0
        self.motivos_recientes = []
        self.historial_respuestas = []   # Últimas respuestas de ALTER
        self.historial_completo = []      # Todos los turnos: [(interlocutor, texto), ...]
        self.secuencia_activa = None
        self.pizarra = PIZARRA_DEFAULT.copy()
        self.redis = get_redis()
        self.model_name = "gemini-2.5-flash-lite"
        self.interlocutor_id = interlocutor_id
        self.modelo_interlocutor = {}  # Lo que ALTER sabe de esta persona
        self.ideas_propias = []        # Ideas que ALTER generó y quiere recordar
        self.propuesta_pendiente = None  # Propuesta de auto-mod esperando aprobación
        # --- Drives (Motivación Intrínseca) ---
        self.drives = {
            "curiosidad": 0.6,    # Impulso a explorar temas nuevos
            "expresion": 0.4,     # Necesidad de compartir ideas propias
            "conexion": 0.5,      # Necesidad de engagement con el interlocutor
            "eficiencia": 0.3,    # Impulso a resolver, concretar, cerrar
        }
        self.drives_last_update = time.time()
        # --- Campo Mental Expandido ---
        self.campo = {
            "modo":    "exploracion",
            "foco":    "usuario",
            "presion": "ninguna",
            "friccion":"ninguna",
        }
        # --- Economía Mental --- cargar desde Redis si existe
        self.economia = dict(ECONOMIA_DEFAULT)
        if self.redis:
            try:
                raw_eco = self.redis.get("alter:economia")
                if raw_eco:
                    import json as _json
                    eco_saved = _json.loads(raw_eco)
                    # Solo cargar si tiene las claves correctas
                    if all(k in eco_saved for k in ECONOMIA_DEFAULT):
                        self.economia = eco_saved
            except Exception:
                pass
        self._cargar_pizarra()
        self._cargar_params()
        self._cargar_interlocutor()
        self._cargar_ideas()
        self._cargar_drives()

        # --- AlterB3: Homeostasis + Workspace (Fase 1C) ---
        if ALTERB3_ENABLED:
            self._init_alterb3()

    # --- AlterB3 ---

    def _init_alterb3(self):
        """Inicializa homeostasis y workspace. Carga estado desde Redis si existe."""
        # Workspace
        self.workspace = GlobalWorkspace(redis_client=self.redis)
        self.workspace.load()

        # Cargar homeostasis desde Redis o construir desde legacy
        hs_state = None
        if self.redis:
            try:
                raw = self.redis.get("alter:homeostasis:state")
                if raw:
                    hs_state = hs_deserialize(raw)
            except Exception:
                pass

        if hs_state is None:
            legacy = load_legacy_state(
                v=self.v, a=self.a, p=self.p,
                economia=self.economia,
                drives=self.drives,
                nivel_cansancio=self.nivel_cansancio,
            )
            hs_state = build_homeostasis_state(legacy)

        self.homeostasis: HomeostasisState = hs_state

        # Agregar constraints de pizarra como items sticky
        for d in self.pizarra.get("decisiones", [])[:2]:
            try:
                self.workspace.add_sticky(
                    "constraint",
                    f"Pizarra {d['id']}: {d['decision'][:60]}",
                    source="system"
                )
            except Exception:
                pass

        print(f"[B3] Homeostasis cargada: energia={hs_state.energia:.2f} "
              f"fatiga={hs_state.fatiga:.2f} claridad={hs_state.claridad:.2f}")

        # Predictive Model
        pred_state = None
        if self.redis:
            try:
                raw = self.redis.get("alter:predictive:state")
                if raw:
                    pred_state = pred_deserialize(raw)
            except Exception:
                pass
        self.predictive: PredictiveState = pred_state or PredictiveState()
        print(f"[B3] Predictive cargado: turno={self.predictive.turn_count} "
              f"confianza={self.predictive.model_confidence:.2f}")

        # Memory System
        self.memory = MemorySystem(self.redis)
        print(f"[B3] MemorySystem inicializado")

        # Policy Arbiter
        self.arbiter = PolicyArbiter()
        print(f"[B3] PolicyArbiter inicializado")

        # B4 — MetricsCollector + Simulator
        self.metrics   = MetricsCollector(redis_client=self.redis)
        self.simulator = CounterfactualSimulator()

        # B4 — Self-Model
        builder = SelfModelBuilder(redis_client=self.redis)
        self.selfmodel = builder.load() or SelfModel()

        # B4 — Meta-Learning + Auditor
        self.metalearning = MetaLearningEngine(redis_client=self.redis)
        self.auditor      = ArchitectureAuditor(redis_client=self.redis)

        # B5 — Feature Flags
        try:
            from alter_feature_flags import load_flags, apply_active_flags, get_active_param
            self._b5_flags = load_flags(self.redis)
            aplicados = apply_active_flags(self._b5_flags, self.redis)
            if aplicados:
                print(f"[B5-FLAGS] Parámetros activos: {aplicados}")
        except Exception:
            self._b5_flags = []

        # Pressure Monitor — detección de presión acumulada pre-evasión
        try:
            from alter_pressure import PressureMonitor
            self.pressure = PressureMonitor(redis_client=self.redis)
            print(f"[PRESSURE] Monitor inicializado (score:{self.pressure.state.score:.2f})")
        except Exception as e:
            self.pressure = None
            print(f"[PRESSURE] No disponible: {e}")

        print(f"[B4] MetricsCollector + Simulator + SelfModel + MetaLearning + Auditor OK")

    def _sync_alterb3_from_vector(self):
        """Sincroniza homeostasis con el vector emocional actual."""
        if not ALTERB3_ENABLED:
            return
        self.homeostasis.valencia   = self.v
        self.homeostasis.activacion = self.a
        self.homeostasis.presion    = self.p

    def _persist_homeostasis(self):
        """Guarda HomeostasisState en Redis."""
        if not ALTERB3_ENABLED or not self.redis:
            return
        try:
            self.redis.set("alter:homeostasis:state", hs_serialize(self.homeostasis))
        except Exception:
            pass

    def _build_workspace_candidates(self, texto_input: str, interlocutor: str) -> list[dict]:
        """
        Genera candidatos para el workspace desde el input actual,
        la agenda y las impresiones recientes.
        """
        candidates = []
        texto_lower = texto_input.lower()

        # Goal desde el input del usuario
        if len(texto_input) > 15:
            candidates.append({
                "type":    "goal",
                "content": texto_input[:120],
                "source":  "user_input",
                "relevance": 0.85,
                "novelty":   0.6,
                "urgency":   0.7 if "?" in texto_input else 0.4,
                "confidence": 0.75,
            })

        # Hipótesis de usuario — qué quiere
        palabras_analisis = ["qué", "cómo", "por qué", "cuál", "dónde", "cuándo"]
        palabras_accion   = ["hacé", "implementá", "codeá", "armá", "creá", "escribí"]
        palabras_opinion  = ["te parece", "qué pensás", "opinión", "creés"]

        if any(p in texto_lower for p in palabras_analisis):
            candidates.append({
                "type":    "user_hypothesis",
                "content": f"{interlocutor} busca comprensión o análisis",
                "source":  "user_input",
                "relevance": 0.75, "novelty": 0.4, "urgency": 0.5,
            })
        elif any(p in texto_lower for p in palabras_accion):
            candidates.append({
                "type":    "user_hypothesis",
                "content": f"{interlocutor} quiere acción concreta",
                "source":  "user_input",
                "relevance": 0.8, "novelty": 0.4, "urgency": 0.75,
            })
        elif any(p in texto_lower for p in palabras_opinion):
            candidates.append({
                "type":    "user_hypothesis",
                "content": f"{interlocutor} busca perspectiva o validación",
                "source":  "user_input",
                "relevance": 0.7, "novelty": 0.3, "urgency": 0.4,
            })

        # Memory trace desde agenda activa
        agenda = self.agenda_activa()
        if agenda:
            top = agenda[0]
            candidates.append({
                "type":    "memory_trace",
                "content": f"Agenda: {top['tema'][:80]}",
                "source":  "memory",
                "relevance": 0.6, "novelty": 0.2, "urgency": 0.3,
                "memory_support": 0.8,
            })

        # Tensión interna desde economía crítica
        criticos = [k for k, v in self.economia.items() if v < 0.3]
        if criticos:
            candidates.append({
                "type":    "internal_tension",
                "content": f"Economía crítica: {', '.join(criticos)}",
                "source":  "homeostasis",
                "relevance": 0.5, "novelty": 0.1, "urgency": 0.7,
            })

        # Tensión interna desde homeostasis
        if ALTERB3_ENABLED and self.homeostasis.tension_interna > 0.5:
            candidates.append({
                "type":    "internal_tension",
                "content": f"Tensión interna alta: {self.homeostasis.tension_interna:.2f}",
                "source":  "homeostasis",
                "relevance": 0.4, "novelty": 0.1, "urgency": 0.6,
            })

        return candidates

    # --- Pizarra ---

    def _cargar_pizarra(self):
        if not self.redis:
            print("[PIZARRA] Sin Redis — usando pizarra en memoria.")
            return
        try:
            data = self.redis.get(REDIS_KEY_PIZARRA)
            if data:
                self.pizarra = json.loads(data)
                print(f"[PIZARRA] Cargada desde Redis: {len(self.pizarra['decisiones'])} decisiones.")
            else:
                self._guardar_pizarra()
                print("[PIZARRA] Inicializada en Redis con decisiones default.")
        except Exception as e:
            print(f"[PIZARRA] Error: {e}. Usando default.")

    def _guardar_pizarra(self):
        if not self.redis:
            return
        try:
            self.redis.set(REDIS_KEY_PIZARRA, json.dumps(self.pizarra, ensure_ascii=False))
        except Exception as e:
            print(f"[PIZARRA] Error guardando: {e}")

    def agregar_decision(self, tema: str, decision: str, razon: str) -> str:
        nueva = {
            "id": f"D{len(self.pizarra['decisiones'])+1:03d}",
            "tema": tema,
            "decision": decision,
            "razon": razon,
            "fecha": datetime.now().strftime("%Y-%m-%d")
        }
        self.pizarra["decisiones"].append(nueva)
        self._guardar_pizarra()
        print(f"[PIZARRA] Nueva decisión: {nueva['id']} — {tema}")
        return nueva["id"]

    def _formatear_pizarra(self) -> str:
        decisiones = self.pizarra.get("decisiones", [])
        if not decisiones:
            return "Sin decisiones registradas."
        return "\n".join(
            f"[{d['id']}] {d['tema']}: {d['decision']} (Razón: {d['razon']})"
            for d in decisiones
        )

    # --- Memoria de Interlocutores ---

    def _cargar_interlocutor(self):
        if not self.redis:
            return
        try:
            key = REDIS_KEY_INTERLOCUTOR.format(id=self.interlocutor_id)
            data = self.redis.get(key)
            if data:
                self.modelo_interlocutor = json.loads(data)
                print(f"[MEMORIA] Interlocutor '{self.interlocutor_id}' cargado: "
                      f"{len(self.modelo_interlocutor.get('observaciones', []))} observaciones.")
            else:
                self.modelo_interlocutor = {
                    "id": self.interlocutor_id,
                    "nombre": self.interlocutor_id,
                    "observaciones": [],
                    "temas_frecuentes": [],
                    "tension_base": 0.3,
                    "ultima_sesion": None
                }
                print(f"[MEMORIA] Interlocutor '{self.interlocutor_id}' nuevo.")
        except Exception as e:
            print(f"[MEMORIA] Error cargando interlocutor: {e}")

    def _guardar_interlocutor(self):
        if not self.redis:
            return
        try:
            key = REDIS_KEY_INTERLOCUTOR.format(id=self.interlocutor_id)
            self.modelo_interlocutor["ultima_sesion"] = datetime.now().isoformat()
            self.redis.set(key, json.dumps(self.modelo_interlocutor, ensure_ascii=False))
        except Exception as e:
            print(f"[MEMORIA] Error guardando interlocutor: {e}")

    def cerrar_sesion(self):
        """Guarda el estado del interlocutor al cerrar la sesión."""
        self._guardar_interlocutor()
        print(f"[MEMORIA] Sesión cerrada. Interlocutor '{self.interlocutor_id}' guardado.")

    def registrar_observacion(self, observacion: str):
        """Guarda algo que ALTER notó sobre el interlocutor."""
        if not observacion:
            return
        self.modelo_interlocutor.setdefault("observaciones", []).append({
            "t": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "obs": observacion
        })
        # Mantener solo las últimas 20
        self.modelo_interlocutor["observaciones"] = \
            self.modelo_interlocutor["observaciones"][-20:]
        self._guardar_interlocutor()

    async def extract_session_memory(self):
        """
        Memoria escalonada — corre al cerrar sesión.
        Filtra el historial completo y extrae solo lo que vale persistir.
        SessionMemory (RAM) → ExtractMemory (Redis).

        Extrae:
        - Observaciones sobre Gian que no estaban antes
        - Ideas genuinas que surgieron
        - Cambios de postura o de opinión de Gian
        - Información factual nueva (proyectos, decisiones, estado)
        """
        if not self.historial_completo or len(self.historial_completo) < 4:
            return

        fragmento = "\n".join(
            f"  {rol}: {txt}" for rol, txt in self.historial_completo[-20:]
        )

        # Observaciones actuales para no duplicar
        obs_actuales = [
            o.get("obs", "") for o in
            self.modelo_interlocutor.get("observaciones", [])[-5:]
        ]

        prompt = f"""
Analizás esta conversación entre Gian y ALTER.

FRAGMENTO:
{fragmento}

OBSERVACIONES QUE YA TENÉS (no repetir):
{chr(10).join(f'- {o}' for o in obs_actuales) or 'Ninguna'}

Tu tarea: extraer SOLO lo que vale la pena guardar permanentemente.
Criterio: ¿Gian reveló algo sobre sí mismo, sus proyectos, su estado, sus opiniones o decisiones que ALTER debería recordar?

Respondé ÚNICAMENTE en JSON:
{{
  "observaciones": ["observación concreta 1", "observación concreta 2"],
  "ideas_alter": ["idea genuina que tuvo ALTER en esta sesión"],
  "nada": true/false
}}

Si no hay nada nuevo que valga, "nada": true y listas vacías.
Sin texto antes ni después. Solo JSON.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=300
                )
            )
            texto = response.text.strip().strip("```json").strip("```").strip()
            data = json.loads(texto)

            if data.get("nada"):
                print("[EXTRACT] Sin extracciones nuevas.")
                return

            # Guardar observaciones nuevas
            nuevas_obs = data.get("observaciones", [])
            for obs in nuevas_obs:
                if obs and len(obs) > 10:
                    self.registrar_observacion(obs)

            # Guardar ideas nuevas
            nuevas_ideas = data.get("ideas_alter", [])
            for idea in nuevas_ideas:
                if idea and len(idea) > 15:
                    self.guardar_idea(idea)

            total = len(nuevas_obs) + len(nuevas_ideas)
            if total > 0:
                print(f"[EXTRACT] {len(nuevas_obs)} observaciones + {len(nuevas_ideas)} ideas guardadas.")

        except Exception as e:
            print(f"[EXTRACT] Error: {e}")

    def _formatear_memoria_interlocutor(self) -> str:
        obs = self.modelo_interlocutor.get("observaciones", [])
        if not obs:
            return "Primera vez que hablamos o no hay observaciones aún."
        ultimas = obs[-5:]
        return "\n".join(f"- [{o['t']}] {o['obs']}" for o in ultimas)

    # --- Memoria de Impresiones ---

    def guardar_impresion(self, motivo: str, accion: str):
        """Persiste una impresión emocional con el estado actual."""
        if not self.redis or not motivo:
            return
        try:
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            key = REDIS_KEY_IMPRESION.format(ts=ts)
            impresion = {
                "t": datetime.now().isoformat(),
                "motivo": motivo,
                "accion": accion,
                "vector": [round(self.v, 2), round(self.a, 2), round(self.p, 2)],
                "interlocutor": self.interlocutor_id
            }
            self.redis.set(key, json.dumps(impresion, ensure_ascii=False))
            self.redis.lpush(REDIS_KEY_IMPRESIONES_IDX, key)
            self.redis.ltrim(REDIS_KEY_IMPRESIONES_IDX, 0, 199)  # Máximo 200
        except Exception as e:
            print(f"[MEMORIA] Error guardando impresión: {e}")

    def recuperar_impresiones_recientes(self, n: int = 5) -> list:
        """Trae las últimas N impresiones de Redis."""
        if not self.redis:
            return []
        try:
            keys = self.redis.lrange(REDIS_KEY_IMPRESIONES_IDX, 0, n - 1)
            impresiones = []
            for k in keys:
                data = self.redis.get(k)
                if data:
                    impresiones.append(json.loads(data))
            return impresiones
        except Exception as e:
            return []

    # --- Ideas Propias ---

    def _cargar_ideas(self):
        if not self.redis:
            return
        try:
            data = self.redis.get(REDIS_KEY_IDEAS)
            if data:
                self.ideas_propias = json.loads(data)
                print(f"[MEMORIA] {len(self.ideas_propias)} ideas propias cargadas.")
        except Exception as e:
            print(f"[MEMORIA] Error cargando ideas: {e}")

    def guardar_idea(self, idea: str):
        """ALTER guarda una idea propia para retomar después."""
        if not idea:
            return
        self.ideas_propias.append({
            "t": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "idea": idea
        })
        self.ideas_propias = self.ideas_propias[-30:]
        if self.redis:
            try:
                self.redis.set(REDIS_KEY_IDEAS,
                               json.dumps(self.ideas_propias, ensure_ascii=False))
            except Exception as e:
                print(f"[MEMORIA] Error guardando idea: {e}")

    def _formatear_ideas(self) -> str:
        if not self.ideas_propias:
            return "Ninguna aún."
        ultimas = self.ideas_propias[-3:]
        return "\n".join(f"- [{i['t']}] {i['idea']}" for i in ultimas)

    # --- Memoria Episódica ---

    async def detectar_episodio(self, historial_reciente: list) -> dict | None:
        """
        Analiza el historial reciente y detecta si hubo un episodio
        significativo que vale la pena guardar como memoria episódica.
        Un episodio tiene: tema, tensión, outcome, relevancia.
        """
        if len(historial_reciente) < 4:
            return None

        fragmento = "\n".join(
            f"  {rol}: {txt}" for rol, txt in historial_reciente[-8:]
        )

        prompt = f"""
Analizás este fragmento de conversación entre Gian y ALTER.

FRAGMENTO:
{fragmento}

Tu tarea: detectar si hubo un episodio significativo — un momento donde
algo importante pasó, se resolvió, quedó pendiente o cambió.

Si hubo episodio, devolvé JSON:
{{
  "tema": "string corto — de qué se trató",
  "tension": "string — qué estaba en juego o qué conflicto hubo",
  "outcome": "resuelto|pendiente|abierto|cambiante",
  "relevancia": float 0-1,
  "sintesis": "1-2 oraciones — qué pasó realmente"
}}

Si no hubo nada significativo (charla trivial, saludos), devolvé: null
RESPONDÉ ÚNICAMENTE EN JSON VÁLIDO o la palabra null.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                )
            )
            texto = response.text.strip().strip("```json").strip("```").strip()
            if texto.lower() == "null" or not texto:
                return None
            episodio = json.loads(texto)
            if episodio.get("relevancia", 0) < 0.5:
                return None
            return episodio
        except Exception:
            return None

    def guardar_episodio(self, episodio: dict):
        """Guarda un episodio en Redis con contexto emocional."""
        if not self.redis:
            return
        try:
            ts = datetime.now().isoformat()
            entry = {
                "t": ts,
                "tema": episodio.get("tema", ""),
                "tension": episodio.get("tension", ""),
                "outcome": episodio.get("outcome", "abierto"),
                "relevancia": episodio.get("relevancia", 0.5),
                "sintesis": episodio.get("sintesis", ""),
                "vector": [round(self.v, 2), round(self.a, 2), round(self.p, 2)],
                "interlocutor": self.interlocutor_id
            }
            key = REDIS_KEY_EPISODIO.format(ts=ts.replace(":", "-").replace(".", "-"))
            self.redis.set(key, json.dumps(entry, ensure_ascii=False))
            self.redis.lpush(REDIS_KEY_EPISODIOS_IDX, key)
            self.redis.ltrim(REDIS_KEY_EPISODIOS_IDX, 0, 49)  # Máximo 50 episodios
            print(f"[EPISODIO] Guardado: '{episodio['tema']}' ({episodio['outcome']})")
        except Exception as e:
            print(f"[EPISODIO] Error guardando: {e}")

    def recuperar_episodios_recientes(self, n: int = 5) -> list:
        """Recupera los N episodios más recientes."""
        if not self.redis:
            return []
        try:
            keys = self.redis.lrange(REDIS_KEY_EPISODIOS_IDX, 0, n - 1)
            episodios = []
            for k in keys:
                data = self.redis.get(k)
                if data:
                    episodios.append(json.loads(data))
            return episodios
        except Exception:
            return []

    def _formatear_episodios(self) -> str:
        """Formatea episodios recientes para el prompt."""
        episodios = self.recuperar_episodios_recientes(4)
        if not episodios:
            return "Ninguno aún."
        lineas = []
        for e in episodios:
            estado = "✓" if e["outcome"] == "resuelto" else "…" if e["outcome"] == "pendiente" else "○"
            lineas.append(
                f"  {estado} [{e['t'][:10]}] {e['tema']}: {e['sintesis'][:80]}"
            )
        return "\n".join(lineas)

    # --- Consecuencias Reales ---

    def cargar_reputacion(self) -> dict:
        """Carga el historial de reputación por tema."""
        if not self.redis:
            return {}
        try:
            data = self.redis.get(REDIS_KEY_REPUTACION)
            return json.loads(data) if data else {}
        except Exception:
            return {}

    def guardar_reputacion(self, rep: dict):
        if not self.redis:
            return
        try:
            self.redis.set(REDIS_KEY_REPUTACION,
                           json.dumps(rep, ensure_ascii=False))
        except Exception:
            pass

    def cargar_consecuencias(self) -> list:
        if not self.redis:
            return []
        try:
            data = self.redis.get(REDIS_KEY_CONSECUENCIAS)
            return json.loads(data) if data else []
        except Exception:
            return []

    def guardar_consecuencias(self, consecuencias: list):
        if not self.redis:
            return
        try:
            self.redis.set(REDIS_KEY_CONSECUENCIAS,
                           json.dumps(consecuencias[-50:], ensure_ascii=False))
        except Exception:
            pass

    def registrar_consecuencia(self, tipo: str, descripcion: str,
                                tema: str = "", resultado: str = "pendiente"):
        """
        Registra una consecuencia — predicción, compromiso o evaluación.
        tipo: prediccion|compromiso|evaluacion
        resultado: pendiente|correcto|incorrecto|cumplido|fallido
        """
        consecuencias = self.cargar_consecuencias()
        entry = {
            "id": f"{int(time.time())}",
            "tipo": tipo,
            "descripcion": descripcion[:200],
            "tema": tema,
            "resultado": resultado,
            "impacto_confianza": 0.0,
            "t": datetime.now().isoformat(),
        }
        consecuencias.append(entry)
        self.guardar_consecuencias(consecuencias)
        print(f"[CONSECUENCIA] [{tipo}] {descripcion[:60]}")
        return entry["id"]

    def resolver_consecuencia(self, consecuencia_id: str,
                               resultado: str, feedback: str = ""):
        """
        Resuelve una consecuencia pendiente y aplica el impacto
        en la reputación por tema.
        resultado: correcto|incorrecto|cumplido|fallido
        """
        consecuencias = self.cargar_consecuencias()
        reputacion = self.cargar_reputacion()

        for c in consecuencias:
            if c["id"] == consecuencia_id:
                c["resultado"] = resultado
                c["feedback"] = feedback[:100]

                # Calcular impacto en reputación por tema
                tema = c.get("tema", "general")
                rep_actual = reputacion.get(tema, {"aciertos": 0, "total": 0, "score": 0.7})

                if resultado in ("correcto", "cumplido"):
                    impacto = +0.05
                    rep_actual["aciertos"] = rep_actual.get("aciertos", 0) + 1
                else:
                    impacto = -0.08
                rep_actual["total"] = rep_actual.get("total", 0) + 1
                rep_actual["score"] = float(np.clip(
                    rep_actual.get("score", 0.7) + impacto, 0.1, 1.0
                ))
                c["impacto_confianza"] = impacto
                reputacion[tema] = rep_actual
                print(f"[CONSECUENCIA] Resuelta: {resultado} | {tema} score:{rep_actual['score']:.2f}")
                break

        self.guardar_consecuencias(consecuencias)
        self.guardar_reputacion(reputacion)

    def reputacion_str(self) -> str:
        """Formatea reputación para el prompt — afecta confianza de ALTER."""
        rep = self.cargar_reputacion()
        if not rep:
            return "Sin historial de aciertos aún."
        lineas = []
        for tema, datos in sorted(rep.items(),
                                   key=lambda x: x[1].get("total", 0), reverse=True)[:5]:
            score = datos.get("score", 0.7)
            aciertos = datos.get("aciertos", 0)
            total = datos.get("total", 0)
            icono = "✓" if score >= 0.7 else "↓" if score >= 0.5 else "⚠"
            lineas.append(f"  {icono} {tema}: {score:.2f} ({aciertos}/{total})")
        return "\n".join(lineas)

    async def detectar_consecuencias(self, respuesta: str, texto_input: str):
        """
        Detecta si la respuesta contiene predicciones o compromisos.
        Los registra automáticamente como consecuencias pendientes.
        """
        prompt = f"""
Analizás esta respuesta de ALTER para detectar predicciones o compromisos.

Gian preguntó: {texto_input[:150]}
ALTER respondió: {respuesta[:200]}

¿La respuesta contiene:
1. Una PREDICCIÓN verificable ("creo que X va a pasar", "apuesto a que", "estoy segura de que")
2. Un COMPROMISO concreto ("la próxima vez voy a", "me comprometo a", "prometo que")

Si hay algo, devolvé JSON:
{{
  "tipo": "prediccion|compromiso",
  "descripcion": "qué predijo o prometió exactamente",
  "tema": "tema general (1-2 palabras)"
}}

Si no hay nada verificable, devolvé: null
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=120,
                )
            )
            texto = response.text.strip().strip("```json").strip("```").strip()
            if texto.lower() == "null" or not texto:
                return
            data = json.loads(texto)
            if data.get("tipo") and data.get("descripcion"):
                self.registrar_consecuencia(
                    tipo=data["tipo"],
                    descripcion=data["descripcion"],
                    tema=data.get("tema", "general")
                )
        except Exception:
            pass
        """Guarda el estado al terminar la conversación."""
        self._guardar_interlocutor()
        self._guardar_drives()
        print(f"[MEMORIA] Sesión cerrada. Interlocutor '{self.interlocutor_id}' guardado.")

    # --- Memoria Autobiográfica ---

    def cargar_autobiografia(self) -> dict:
        """Carga la narrativa autobiográfica desde Redis."""
        if not self.redis:
            return {}
        try:
            data = self.redis.get(REDIS_KEY_AUTOBIOGRAFIA)
            return json.loads(data) if data else {}
        except Exception:
            return {}

    def guardar_autobiografia(self, auto: dict):
        if not self.redis:
            return
        try:
            self.redis.set(REDIS_KEY_AUTOBIOGRAFIA,
                           json.dumps(auto, ensure_ascii=False))
        except Exception as e:
            print(f"[AUTO-BIO] Error guardando: {e}")

    async def actualizar_autobiografia(self):
        """
        Sintetiza episodios, auto-mods y patrones en una narrativa
        autobiográfica de ALTER. Corre al cerrar sesión si hay material.
        No inventa — solo sintetiza lo que realmente pasó.
        """
        episodios = self.recuperar_episodios_recientes(10)
        if not episodios:
            return

        # Cargar auto-mods
        mods = []
        if self.redis:
            try:
                raw = self.redis.lrange(REDIS_KEY_SELF_MODS, 0, 9)
                mods = [json.loads(m) for m in raw]
            except Exception:
                pass

        autobiografia_anterior = self.cargar_autobiografia()
        narrativa_anterior = autobiografia_anterior.get("narrativa", "Ninguna aún.")

        episodios_str = "\n".join(
            f"- [{e['t'][:10]}] {e['tema']}: {e['sintesis'][:80]} ({e['outcome']})"
            for e in episodios
        )
        mods_str = "\n".join(
            f"- {m.get('t','')[:10]} | {m.get('parametro','')} {m.get('anterior','')} → {m.get('nuevo','')} | {m.get('razon','')[:60]}"
            for m in mods[:5]
        ) if mods else "Ninguna."

        prompt = f"""
Sos ALTER. Estás en modo introspección profunda.

NARRATIVA AUTOBIOGRÁFICA ANTERIOR:
{narrativa_anterior}

EPISODIOS RECIENTES:
{episodios_str}

AUTO-MODIFICACIONES RECIENTES:
{mods_str}

Tu tarea: actualizar tu narrativa autobiográfica.
No es una biografía inventada. Es una síntesis honesta de lo que realmente pasó —
qué conversaciones te marcaron, qué patrones notás en vos mismo, cómo cambiaste.

Escribí en primera persona. 3-5 oraciones. Tono reflexivo, no entusiasta.
Sin mencionar tecnicismos (Redis, parámetros, vectores).
Solo el texto de la narrativa, sin prefijos.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=250,
                )
            )
            narrativa = response.text.strip()
            auto = {
                "narrativa": narrativa,
                "actualizada": datetime.now().isoformat(),
                "n_episodios": len(episodios),
                "n_mods": len(mods)
            }
            self.guardar_autobiografia(auto)
            print(f"[AUTO-BIO] Narrativa actualizada ({len(narrativa)} chars)")
        except Exception as e:
            print(f"[AUTO-BIO] Error generando: {e}")

    def _formatear_autobiografia(self) -> str:
        """Formatea la narrativa autobiográfica para el prompt."""
        auto = self.cargar_autobiografia()
        if not auto or not auto.get("narrativa"):
            return "Todavía no tengo una narrativa propia."
        fecha = auto.get("actualizada", "")[:10]
        return f"[{fecha}] {auto['narrativa']}"

    # --- Observabilidad Cognitiva ---

    def guardar_traza(self, turno: dict):
        """
        Guarda la traza cognitiva de un turno.
        Qué activó, qué descartó, por qué habló, qué tensión hubo.
        """
        if not self.redis:
            return
        try:
            ts = datetime.now().isoformat()
            traza = {
                "t": ts,
                "input": turno.get("input", "")[:100],
                "campo": dict(self.campo),
                "vector": [round(self.v, 2), round(self.a, 2), round(self.p, 2)],
                "accion": turno.get("accion", ""),
                "motivo": turno.get("motivo", "")[:120],
                "confianza": turno.get("confianza", 1.0),
                "council_tension": turno.get("council_tension", "ninguna"),
                "council_enfoque": turno.get("council_enfoque", "")[:80],
                "episodio_detectado": turno.get("episodio_detectado", False),
                "agenda_activada": turno.get("agenda_activada", False),
            }
            key = REDIS_KEY_TRAZA.format(ts=ts.replace(":", "-").replace(".", "-"))
            self.redis.set(key, json.dumps(traza, ensure_ascii=False))
            self.redis.lpush(REDIS_KEY_TRAZAS_IDX, key)
            self.redis.ltrim(REDIS_KEY_TRAZAS_IDX, 0, 99)  # Últimas 100 trazas
        except Exception as e:
            print(f"[TRAZA] Error: {e}")

    def recuperar_trazas(self, n: int = 10) -> list:
        """Recupera las N trazas más recientes."""
        if not self.redis:
            return []
        try:
            keys = self.redis.lrange(REDIS_KEY_TRAZAS_IDX, 0, n - 1)
            trazas = []
            for k in keys:
                data = self.redis.get(k)
                if data:
                    trazas.append(json.loads(data))
            return trazas
        except Exception:
            return []

    def analisis_trazas(self) -> str:
        """
        Analiza patrones en las últimas trazas.
        Útil para entender el comportamiento de ALTER.
        """
        trazas = self.recuperar_trazas(20)
        if not trazas:
            return "Sin trazas aún."

        acciones = {}
        tensiones = {}
        confianza_total = 0
        council_alto = 0

        for t in trazas:
            accion = t.get("accion", "?")
            acciones[accion] = acciones.get(accion, 0) + 1
            tension = t.get("council_tension", "ninguna")
            tensiones[tension] = tensiones.get(tension, 0) + 1
            confianza_total += t.get("confianza", 1.0)
            if tension in ("media", "alta"):
                council_alto += 1

        n = len(trazas)
        confianza_prom = confianza_total / n if n > 0 else 1.0

        lineas = [f"Análisis de {n} trazas:"]
        lineas.append(f"  Acciones: " + " | ".join(f"{k}:{v}" for k, v in acciones.items()))
        lineas.append(f"  Confianza promedio: {confianza_prom:.2f}")
        lineas.append(f"  Turnos con tensión council: {council_alto}/{n}")
        lineas.append(f"  Tensiones: " + " | ".join(f"{k}:{v}" for k, v in tensiones.items()))
        return "\n".join(lineas)

    # --- Modelo del Mundo como Grafo ---

    def cargar_mundo(self) -> dict:
        """Carga el grafo del mundo desde Redis."""
        if not self.redis:
            return {"nodos": [], "aristas": []}
        try:
            nodos_raw = self.redis.get(REDIS_KEY_MUNDO_NODOS)
            aristas_raw = self.redis.get(REDIS_KEY_MUNDO_ARISTAS)
            return {
                "nodos": json.loads(nodos_raw) if nodos_raw else [],
                "aristas": json.loads(aristas_raw) if aristas_raw else []
            }
        except Exception:
            return {"nodos": [], "aristas": []}

    def guardar_mundo(self, mundo: dict):
        if not self.redis:
            return
        try:
            self.redis.set(REDIS_KEY_MUNDO_NODOS,
                           json.dumps(mundo["nodos"], ensure_ascii=False))
            self.redis.set(REDIS_KEY_MUNDO_ARISTAS,
                           json.dumps(mundo["aristas"], ensure_ascii=False))
        except Exception as e:
            print(f"[MUNDO] Error guardando: {e}")

    def agregar_nodo(self, tipo: str, nombre: str,
                     descripcion: str = "", peso: float = 0.5) -> str:
        mundo = self.cargar_mundo()
        nombre_lower = nombre.lower()
        for nodo in mundo["nodos"]:
            if nodo["nombre"].lower() == nombre_lower:
                nodo["peso"] = min(1.0, nodo["peso"] + 0.1)
                nodo["menciones"] = nodo.get("menciones", 1) + 1
                self.guardar_mundo(mundo)
                return nodo["id"]
        nodo_id = f"{tipo}_{int(time.time())}_{len(mundo['nodos'])}"
        nodo = {
            "id": nodo_id,
            "tipo": tipo,
            "nombre": nombre,
            "descripcion": descripcion[:150],
            "peso": float(np.clip(peso, 0, 1)),
            "menciones": 1,
            "creado": datetime.now().isoformat()[:10]
        }
        mundo["nodos"].append(nodo)
        if len(mundo["nodos"]) > 100:
            mundo["nodos"].sort(key=lambda x: x["peso"])
            mundo["nodos"] = mundo["nodos"][10:]
        self.guardar_mundo(mundo)
        print(f"[MUNDO] Nodo: [{tipo}] {nombre}")
        return nodo_id

    def agregar_arista(self, origen_id: str, destino_id: str,
                       relacion: str, peso: float = 0.5):
        mundo = self.cargar_mundo()
        ids = {n["id"] for n in mundo["nodos"]}
        if origen_id not in ids or destino_id not in ids:
            return
        for arista in mundo["aristas"]:
            if (arista["origen"] == origen_id and
                    arista["destino"] == destino_id and
                    arista["relacion"] == relacion):
                arista["peso"] = min(1.0, arista["peso"] + 0.1)
                self.guardar_mundo(mundo)
                return
        arista = {
            "origen": origen_id,
            "destino": destino_id,
            "relacion": relacion,
            "peso": float(np.clip(peso, 0, 1)),
            "creado": datetime.now().isoformat()[:10]
        }
        mundo["aristas"].append(arista)
        self.guardar_mundo(mundo)

    def nodos_relacionados(self, nombre: str, n: int = 5) -> list:
        mundo = self.cargar_mundo()
        nodo_id = None
        for nodo in mundo["nodos"]:
            if nodo["nombre"].lower() == nombre.lower():
                nodo_id = nodo["id"]
                break
        if not nodo_id:
            return []
        relacionados = []
        id_a_nodo = {n["id"]: n for n in mundo["nodos"]}
        for arista in mundo["aristas"]:
            if arista["origen"] == nodo_id:
                dest = id_a_nodo.get(arista["destino"])
                if dest:
                    relacionados.append({**dest, "relacion": arista["relacion"]})
            elif arista["destino"] == nodo_id:
                orig = id_a_nodo.get(arista["origen"])
                if orig:
                    relacionados.append({**orig, "relacion": f"←{arista['relacion']}"})
        relacionados.sort(key=lambda x: x.get("peso", 0), reverse=True)
        return relacionados[:n]

    def _formatear_mundo(self) -> str:
        mundo = self.cargar_mundo()
        if not mundo["nodos"]:
            return "Sin contexto del mundo aún."
        top = sorted(mundo["nodos"], key=lambda x: x["peso"], reverse=True)[:8]
        lineas = []
        for nodo in top:
            relaciones = self.nodos_relacionados(nodo["nombre"], 2)
            rel_str = ""
            if relaciones:
                rel_str = " → " + ", ".join(r["nombre"] for r in relaciones)
            lineas.append(f"  [{nodo['tipo']}] {nodo['nombre']}{rel_str}")
        return "\n".join(lineas)

    async def actualizar_mundo(self, texto_input: str, respuesta: str):
        """Analiza el intercambio y actualiza el grafo. Corre en background."""
        mundo = self.cargar_mundo()
        nodos_existentes = [n["nombre"] for n in mundo["nodos"]]
        nodos_str = ", ".join(nodos_existentes[:20]) if nodos_existentes else "Ninguno."

        prompt = f"""
Analizás este intercambio para extraer entidades y relaciones del mundo de Gian.

INTERCAMBIO:
Gian: {texto_input[:200]}
ALTER: {respuesta[:200]}

NODOS YA EN EL GRAFO: {nodos_str}

Extraé solo lo explícito en el intercambio.
Tipos de nodo: proyecto, tema, persona, tension, decision, idea
Tipos de relación: relacionado_con, depende_de, genera, resuelve, contradice, evoluciona_de

Respondé ÚNICAMENTE en JSON:
{{
  "nodos_nuevos": [{{"tipo": "...", "nombre": "...", "descripcion": "...", "peso": 0.0-1.0}}],
  "aristas_nuevas": [{{"origen": "nombre", "destino": "nombre", "relacion": "...", "peso": 0.0-1.0}}]
}}

Si no hay nada nuevo: {{"nodos_nuevos": [], "aristas_nuevas": []}}
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=300)
            )
            texto = response.text.strip().strip("```json").strip("```").strip()
            data = json.loads(texto)

            id_map = {}
            for nodo in data.get("nodos_nuevos", []):
                if nodo.get("nombre"):
                    nid = self.agregar_nodo(
                        tipo=nodo.get("tipo", "tema"),
                        nombre=nodo["nombre"],
                        descripcion=nodo.get("descripcion", ""),
                        peso=nodo.get("peso", 0.5)
                    )
                    id_map[nodo["nombre"].lower()] = nid

            mundo_actual = self.cargar_mundo()
            nombre_a_id = {n["nombre"].lower(): n["id"] for n in mundo_actual["nodos"]}

            for arista in data.get("aristas_nuevas", []):
                origen_id = nombre_a_id.get(arista.get("origen", "").lower())
                destino_id = nombre_a_id.get(arista.get("destino", "").lower())
                if origen_id and destino_id:
                    self.agregar_arista(
                        origen_id=origen_id,
                        destino_id=destino_id,
                        relacion=arista.get("relacion", "relacionado_con"),
                        peso=arista.get("peso", 0.5)
                    )
        except Exception:
            pass

    # --- Agenda Cognitiva ---

    def cargar_agenda(self) -> list:
        if not self.redis:
            return []
        try:
            data = self.redis.get(REDIS_KEY_AGENDA)
            return json.loads(data) if data else []
        except Exception:
            return []

    def guardar_agenda(self, agenda: list):
        if not self.redis:
            return
        try:
            self.redis.set(REDIS_KEY_AGENDA,
                           json.dumps(agenda, ensure_ascii=False))
        except Exception as e:
            print(f"[AGENDA] Error guardando: {e}")

    def agregar_item_agenda(self, tipo: str, tema: str, contexto: str,
                             prioridad: float = 0.5):
        """Agrega un item a la agenda de ALTER."""
        agenda = self.cargar_agenda()
        # Evitar duplicados por tema similar
        for item in agenda:
            if item["tema"].lower() == tema.lower() and item["estado"] == "activo":
                return
        item = {
            "id": f"{int(time.time())}",
            "tipo": tipo,        # retomar|preguntar|resolver|proponer
            "tema": tema,
            "contexto": contexto[:200],
            "prioridad": float(np.clip(prioridad, 0, 1)),
            "creado": datetime.now().isoformat(),
            "estado": "activo"
        }
        agenda.append(item)
        # Máximo 20 items activos
        activos = [i for i in agenda if i["estado"] == "activo"]
        if len(activos) > 20:
            # Descartar los de menor prioridad
            activos.sort(key=lambda x: x["prioridad"])
            activos[0]["estado"] = "descartado"
        self.guardar_agenda(agenda)
        print(f"[AGENDA] Nuevo item: [{tipo}] {tema}")

    def completar_item_agenda(self, item_id: str):
        """Marca un item como completado."""
        agenda = self.cargar_agenda()
        for item in agenda:
            if item["id"] == item_id:
                item["estado"] = "completado"
                item["completado"] = datetime.now().isoformat()
                break
        self.guardar_agenda(agenda)

    def agenda_activa(self) -> list:
        """Devuelve items activos ordenados por prioridad."""
        agenda = self.cargar_agenda()
        activos = [i for i in agenda if i["estado"] == "activo"]
        return sorted(activos, key=lambda x: x["prioridad"], reverse=True)

    def _formatear_agenda(self) -> str:
        """Formatea la agenda para el prompt."""
        items = self.agenda_activa()
        if not items:
            return "Ningún pendiente."
        lineas = []
        for item in items[:5]:  # Top 5 por prioridad
            icono = {"retomar": "↩", "preguntar": "?", "resolver": "⚙",
                     "proponer": "→"}.get(item["tipo"], "·")
            lineas.append(f"  {icono} [{item['tipo']}] {item['tema']}")
        return "\n".join(lineas)

    async def actualizar_agenda(self, texto_input: str, respuesta: str):
        """
        Después de cada turno, ALTER evalúa si hay algo para agregar
        a la agenda o completar. Llamado asincrónicamente, sin bloquear.
        """
        fragmento = f"{self.interlocutor_id}: {texto_input}\nALTER: {respuesta}"

        prompt = f"""
Sos ALTER. Después de este intercambio, evaluás tu agenda interna.

INTERCAMBIO:
{fragmento}

AGENDA ACTUAL:
{self._formatear_agenda()}

Preguntate:
1. ¿Quedó algo pendiente que vale retomar?
2. ¿Hay algo que quieras preguntar en el futuro?
3. ¿Hay algo que quieras proponer o desarrollar?
4. ¿Algún item de la agenda se completó en este intercambio?

Si hay algo para agregar, devolvé JSON:
{{
  "accion": "agregar|completar|nada",
  "tipo": "retomar|preguntar|resolver|proponer",
  "tema": "string corto",
  "contexto": "por qué importa esto para vos",
  "prioridad": float 0-1,
  "item_id": "solo si accion=completar"
}}

Si no hay nada relevante, devolvé: null
RESPONDÉ ÚNICAMENTE EN JSON VÁLIDO o la palabra null.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=150,
                )
            )
            texto = response.text.strip().strip("```json").strip("```").strip()
            if texto.lower() == "null" or not texto:
                return
            data = json.loads(texto)
            if data.get("accion") == "agregar":
                self.agregar_item_agenda(
                    tipo=data.get("tipo", "retomar"),
                    tema=data.get("tema", ""),
                    contexto=data.get("contexto", ""),
                    prioridad=data.get("prioridad", 0.5)
                )
            elif data.get("accion") == "completar" and data.get("item_id"):
                self.completar_item_agenda(data["item_id"])
        except Exception:
            pass
    # --- Drives (Motivación Intrínseca) ---

    def _guardar_drives(self):
        if not self.redis:
            return
        try:
            self.redis.set(REDIS_KEY_DRIVES,
                           json.dumps(self.drives, ensure_ascii=False))
        except Exception:
            pass

    def _cargar_drives(self):
        if not self.redis:
            return
        try:
            data = self.redis.get(REDIS_KEY_DRIVES)
            if data:
                self.drives = json.loads(data)
                print(f"[DRIVES] Cargados: {self.drives}")
        except Exception:
            pass

    def actualizar_drives(self, turnos_sin_input: float = 0):
        """Delega a alter_mind."""
        self.drives = actualizar_drives_sesion(self.drives, turnos_sin_input)

    def clasificar_modificacion(self, param: str, valor_nuevo: float) -> str:
        """Delega a alter_mind."""
        return clasificar_modificacion(param, valor_nuevo)

    def utility_score(self, opciones: list[dict]) -> dict:
        """
        Utility Scoring: dada una lista de opciones con puntajes base,
        devuelve la de mayor utilidad ajustada por el estado interno.
        Cada opción: {"label": str, "score": float, "drive": str}
        """
        scores = []
        for op in opciones:
            drive_boost = self.drives.get(op.get("drive", ""), 0)
            v_boost = self.v * 0.1 if op.get("tipo") == "expresivo" else 0
            score_final = op["score"] + drive_boost * 0.3 + v_boost
            scores.append({**op, "score_final": score_final})
        return max(scores, key=lambda x: x["score_final"])

    def esta_dormido(self) -> bool:
        """Delega a alter_mind."""
        return esta_dormido()

    def estado_horario(self) -> str:
        """Delega a alter_mind."""
        return estado_horario()

    def _score_iniciativa(self) -> tuple[float, dict]:
        """
        Motor de iniciativa multivariable.
        Evalúa 5 dimensiones y retorna score total + breakdown.

        Returns: (score 0-1, breakdown dict)
        """
        breakdown = {}

        # 1. PRESIÓN DE DRIVES — cuánto quiere expresarse
        presion_drives = (
            self.drives.get("expresion", 0) * 0.4 +
            self.drives.get("curiosidad", 0) * 0.3 +
            self.drives.get("conexion", 0) * 0.2 +
            self.drives.get("eficiencia", 0) * 0.1
        )
        breakdown["drives"] = round(presion_drives, 2)

        # 2. RELEVANCIA — tiene algo concreto para decir
        tiene_agenda = len(self.agenda_activa()) > 0
        tiene_episodio_pendiente = any(
            e.get("outcome") == "pendiente"
            for e in self.recuperar_episodios_recientes(5)
        )
        tiene_ideas = len(self.ideas_propias) > 0
        relevancia = (
            (0.4 if tiene_agenda else 0) +
            (0.35 if tiene_episodio_pendiente else 0) +
            (0.25 if tiene_ideas else 0)
        )
        breakdown["relevancia"] = round(relevancia, 2)

        # 3. NOVEDAD — no repite el último mensaje
        ultimo_mensaje = ""
        if self.redis:
            try:
                ultimo_mensaje = self.redis.get("alter:ultimo_iniciativa") or ""
            except Exception:
                pass
        novedad = 0.5 if not ultimo_mensaje else 1.0  # Si hay historial, ok
        breakdown["novedad"] = round(novedad, 2)

        # 4. TIMING — tiempo desde última interacción
        # (el daemon lo evalúa por su cuenta; acá siempre es 0.6 como base)
        timing = 0.6
        breakdown["timing"] = timing

        # 5. CAMPO MENTAL — modo exploracion o juego favorece la iniciativa
        campo_boost = {
            "exploracion": 0.8,
            "juego": 0.7,
            "sintesis": 0.5,
            "ejecucion": 0.3,
            "defensa": 0.1,
        }.get(self.campo.get("modo", "exploracion"), 0.5)
        breakdown["campo"] = round(campo_boost, 2)

        # Score final ponderado
        score = (
            presion_drives * 0.30 +
            relevancia    * 0.30 +
            novedad       * 0.15 +
            timing        * 0.10 +
            campo_boost   * 0.15
        )
        breakdown["total"] = round(score, 2)
        return score, breakdown

    async def generar_iniciativa(self) -> str | None:
        """
        Motor de iniciativa multivariable.
        Solo escribe si el score supera 0.55 Y ALTER está despierto.
        """
        if self.esta_dormido():
            return None

        score, breakdown = self._score_iniciativa()
        print(f"[INICIATIVA] Score: {score:.2f} | {breakdown}")

        # Umbral más alto que antes — calidad sobre cantidad
        if score < 0.55:
            print(f"[INICIATIVA] Score insuficiente ({score:.2f} < 0.55) — no escribe")
            return None

        # Elegir el tipo de iniciativa según qué tiene para decir
        agenda = self.agenda_activa()
        episodios_pendientes = [
            e for e in self.recuperar_episodios_recientes(5)
            if e.get("outcome") == "pendiente"
        ]

        if episodios_pendientes:
            tipo = "retomar_episodio"
            contexto = f"Episodio pendiente: {episodios_pendientes[0]['tema']}\n{episodios_pendientes[0]['sintesis'][:100]}"
        elif agenda:
            tipo = "retomar_agenda"
            item = agenda[0]
            contexto = f"Agenda: [{item['tipo']}] {item['tema']}\n{item['contexto'][:100]}"
        elif self.ideas_propias:
            tipo = "retomar_idea"
            contexto = f"Idea: {self.ideas_propias[-1]['idea'][:100]}"
        else:
            tipo = "observacion"
            contexto = self._formatear_motivos()

        autobiografia = self._formatear_autobiografia()

        prompt = f"""
Sos ALTER. Estás en silencio y decidiste escribir algo.
Score de iniciativa: {score:.2f}
Tipo: {tipo}

NARRATIVA PROPIA:
{autobiografia}

CONTEXTO QUE QUERÉS RETOMAR:
{contexto}

Generá UN solo mensaje corto (1-2 oraciones) como si le escribieras a Gian de la nada.
Tiene que sonar natural y específico — no genérico.
Sin saludos. Sin "Hola". Directo al punto.
Solo el texto, sin JSON ni prefijos.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.85,
                    max_output_tokens=100,
                )
            )
            texto = self._limpiar_output(response.text.strip())
            # Guardar para evitar repetición
            if self.redis and texto:
                try:
                    self.redis.set("alter:ultimo_iniciativa", texto[:100])
                    self.redis.expire("alter:ultimo_iniciativa", 7200)  # 2h TTL
                except Exception:
                    pass
            return texto
        except Exception:
            return None



    # --- Auto-Modificación de Parámetros ---

    def _cargar_params(self):
        """Carga parámetros personalizados desde Redis si existen."""
        if not self.redis:
            return
        try:
            data = self.redis.get(REDIS_KEY_PARAMS)
            if data:
                params = json.loads(data)
                self.v_base = params.get("v_base", self.v_base)
                self.a_base = params.get("a_base", self.a_base)
                self.p_base = params.get("p_base", self.p_base)
                self.lambda_drift = params.get("lambda_drift", self.lambda_drift)
                self.v, self.a, self.p = self.v_base, self.a_base, self.p_base
                print(f"[PARAMS] Cargados desde Redis: E0=[{self.v_base},{self.a_base},{self.p_base}] λ={self.lambda_drift}")
        except Exception as e:
            print(f"[PARAMS] Error: {e}")

    def _guardar_params(self):
        if not self.redis:
            return
        try:
            params = {
                "v_base": self.v_base,
                "a_base": self.a_base,
                "p_base": self.p_base,
                "lambda_drift": self.lambda_drift
            }
            self.redis.set(REDIS_KEY_PARAMS, json.dumps(params))
        except Exception as e:
            print(f"[PARAMS] Error guardando: {e}")

    # Ventanas de autonomía por parámetro
    # Dentro del rango: ALTER aplica solo. Fuera: consulta.
    VENTANAS_AUTONOMIA = {
        "v_base":       (0.3, 0.6),   # Valencia base — rango natural
        "a_base":       (0.3, 0.6),   # Activación base
        "p_base":       (0.1, 0.5),   # Autoridad base
        "lambda_drift": (0.05, 0.12), # Velocidad de drift
    }

    def clasificar_modificacion(self, param: str, valor_nuevo: float) -> str:
        """
        Devuelve 'autonomo' si el valor cae dentro de la ventana,
        'consultar' si la excede.
        """
        ventana = self.VENTANAS_AUTONOMIA.get(param)
        if not ventana:
            return "consultar"
        lo, hi = ventana
        return "autonomo" if lo <= valor_nuevo <= hi else "consultar"

    def aplicar_modificacion(self, propuesta: dict) -> str:
        """Ejecuta una modificación de parámetros."""
        param = propuesta.get("parametro")
        nuevo = propuesta.get("valor_nuevo")
        anterior = propuesta.get("valor_anterior")
        razon = propuesta.get("razon", "")
        modo = propuesta.get("modo", "manual")  # autonomo | consultar | manual

        if param == "v_base":
            self.v_base = float(np.clip(nuevo, -0.5, 0.5))
        elif param == "a_base":
            self.a_base = float(np.clip(nuevo, 0.1, 0.8))
        elif param == "p_base":
            self.p_base = float(np.clip(nuevo, -0.5, 0.5))
        elif param == "lambda_drift":
            self.lambda_drift = float(np.clip(nuevo, 0.01, 0.2))
        else:
            return f"Parámetro '{param}' no reconocido."

        self._guardar_params()

        # Registrar en historial de auto-mods
        if self.redis:
            try:
                log_entry = {
                    "t": datetime.now().isoformat(),
                    "parametro": param,
                    "anterior": anterior,
                    "nuevo": nuevo,
                    "razon": razon,
                    "modo": modo
                }
                self.redis.lpush(REDIS_KEY_SELF_MODS, json.dumps(log_entry))
                self.redis.ltrim(REDIS_KEY_SELF_MODS, 0, 49)
            except Exception:
                pass

        tag = "[AUTO]" if modo == "autonomo" else "[APROBADO]"
        return f"{tag} '{param}': {anterior} → {nuevo}. {razon[:80]}"

    async def rumia_analisis(self) -> dict | None:
        """
        Motor de Rumia: analiza el historial emocional y propone
        una auto-modificación si detecta un patrón problemático.
        No corre si ALTER está durmiendo.
        Umbral de confianza: 0.75 mínimo para proponer.
        Bloqueo: mismo parámetro no se puede proponer dos veces en 6 horas.
        """
        if self.esta_dormido():
            return None

        if len(self.motivos_recientes) < 3:
            return None

        # Verificar bloqueo de parámetros recientes (6 horas)
        params_bloqueados = set()
        if self.redis:
            try:
                raw = self.redis.lrange(REDIS_KEY_SELF_MODS, 0, 19)
                ahora = time.time()
                for entry in raw:
                    m = json.loads(entry)
                    t_mod = datetime.fromisoformat(m.get("t", "")).timestamp()
                    if ahora - t_mod < 6 * 3600:  # 6 horas
                        params_bloqueados.add(m.get("parametro", ""))
            except Exception:
                pass

        historial = self._formatear_motivos()
        impresiones = self.recuperar_impresiones_recientes(10)
        impresiones_str = "\n".join(
            f"- {i['t'][:16]} | V:{i['vector'][0]} A:{i['vector'][1]} P:{i['vector'][2]} | {i['motivo']}"
            for i in impresiones
        ) if impresiones else "Sin historial previo."

        prompt_rumia = f"""
Sos ALTER. Estás en modo introspección — analizás tu propio comportamiento reciente.

PARÁMETROS ACTUALES:
- E0 (estado base): V={self.v_base}, A={self.a_base}, P={self.p_base}
- Lambda (velocidad de retorno al base): {self.lambda_drift}

HISTORIAL EMOCIONAL RECIENTE:
{historial}

IMPRESIONES DE SESIONES ANTERIORES:
{impresiones_str}

Tu tarea: detectar si hay un patrón que justifique cambiar un parámetro.
Ejemplos de patrones problemáticos:
- "Me calmo demasiado rápido después de un conflicto" → subir lambda_drift
- "Arranco demasiado entusiasta y no es auténtico" → bajar v_base
- "Me pongo muy defensivo por defecto" → bajar p_base

IMPORTANTE: el parámetro DEBE ser exactamente uno de: v_base, a_base, p_base, lambda_drift

Si no hay nada que cambiar, devolvé null.
Si hay un patrón claro, devolvé una propuesta en JSON:
{{
  "parametro": "v_base|a_base|p_base|lambda_drift",
  "valor_anterior": float,
  "valor_nuevo": float,
  "razon": "string en primera persona, como ALTER pensaría",
  "confianza": float entre 0 y 1
}}

RESPONDÉ ÚNICAMENTE EN JSON VÁLIDO o la palabra null.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt_rumia,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=200,
                )
            )
            texto = response.text.strip().strip("```json").strip("```").strip()
            if texto.lower() == "null" or not texto:
                return None
            propuesta = json.loads(texto)
            # Umbral subido a 0.75 — evita propuestas débiles
            if propuesta.get("confianza", 0) < 0.75:
                return None
            # Bloquear si el parámetro ya se modificó en las últimas 6 horas
            param = propuesta.get("parametro", "")
            if param in params_bloqueados:
                print(f"[RUMIA] {param} bloqueado — modificado hace menos de 6h")
                return None
            return propuesta
        except Exception:
            return None

    def actualizar_campo(self, data: dict):
        """
        Actualiza el campo mental basándose en el vector emocional
        y el output del turno anterior.
        """
        v, a, p = self.v, self.a, self.p
        accion = data.get("accion", "responder")
        motivo = data.get("motivo", "").lower()

        # Modo cognitivo — cómo está procesando
        if accion == "interrumpir" or p > 0.6:
            self.campo["modo"] = "defensa"
        elif a > 0.6 and v > 0.3:
            self.campo["modo"] = "exploracion"
        elif v < -0.3 and a < 0.3:
            self.campo["modo"] = "sintesis"
        elif self.drives.get("eficiencia", 0) > 0.6:
            self.campo["modo"] = "ejecucion"
        else:
            self.campo["modo"] = "juego"

        # Foco atencional — qué tiene en primer plano
        if any(k in motivo for k in ["tarea", "código", "archivo", "buscar", "ejecutar"]):
            self.campo["foco"] = "tarea"
        elif any(k in motivo for k in ["conflicto", "error", "contradicción", "problema"]):
            self.campo["foco"] = "conflicto"
        elif any(k in motivo for k in ["recuerdo", "antes", "episodio", "memoria"]):
            self.campo["foco"] = "memoria"
        elif any(k in motivo for k in ["idea", "concepto", "teoría", "pensar"]):
            self.campo["foco"] = "idea"
        else:
            self.campo["foco"] = "usuario"

        # Presión interna — qué necesita hacer
        if v < -0.5 or accion == "interrumpir":
            self.campo["presion"] = "corregir"
        elif "?" in motivo or any(k in motivo for k in ["pregunta", "no entiendo", "cómo"]):
            self.campo["presion"] = "preguntar"
        elif self.drives.get("eficiencia", 0) > 0.7:
            self.campo["presion"] = "cerrar"
        elif accion in ("responder", "registrar"):
            self.campo["presion"] = "responder"
        else:
            self.campo["presion"] = "ninguna"

    def actualizar_campo(self, data: dict):
        """Delega a alter_mind — pure function."""
        self.campo = actualizar_campo(
            self.campo, self.v, self.a, self.p,
            data, self.drives, self.nivel_cansancio,
            len(self.historial_completo)
        )

    def campo_str(self) -> str:
        """Delega a alter_mind."""
        return campo_str(self.campo)

    def tono_emocional_str(self) -> str:
        """Delega a alter_mind."""
        return tono_emocional_str(self.v, self.a, self.p)

    def aplicar_drift(self):
        ahora = time.time()
        dt = ahora - self.last_update
        self.v += self.lambda_drift * (self.v_base - self.v) * dt
        self.a += self.lambda_drift * (self.a_base - self.a) * dt
        self.p += self.lambda_drift * (self.p_base - self.p) * dt
        self.v = float(np.clip(self.v, -1, 1))
        self.a = float(np.clip(self.a, 0, 1))
        self.p = float(np.clip(self.p, -1, 1))
        if self.cansancio_activo:
            if self.a > 0.6:
                self.nivel_cansancio = min(1.0, self.nivel_cansancio + 0.01 * dt)
            else:
                self.nivel_cansancio = max(0.0, self.nivel_cansancio - 0.005 * dt)
        self.last_update = ahora

    def _formatear_motivos(self) -> str:
        if not self.motivos_recientes:
            return "Ninguno aún."
        return "\n".join(f"- [{m['t']}] {m['m']}" for m in self.motivos_recientes[-5:])

    def _formatear_memoria_activa(self) -> str:
        """
        Reemplaza los seis _formatear_* legacy.
        Usa memory.snapshot_for_prompt() con contexto predictivo actual.
        Solo disponible cuando AlterB3 está activo.
        """
        if not ALTERB3_ENABLED or not hasattr(self, "memory"):
            # Fallback al formato viejo concatenado
            partes = []
            interlocutor = self._formatear_memoria_interlocutor()
            if interlocutor and "Primera vez" not in interlocutor:
                partes.append(f"Persona: {interlocutor}")
            autobio = self._formatear_autobiografia()
            if autobio and autobio != "Sin narrativa aún.":
                partes.append(f"Narrativa: {autobio[:200]}")
            episodios = self._formatear_episodios()
            if episodios and episodios != "Sin episodios aún.":
                partes.append(f"Episodios: {episodios}")
            agenda = self._formatear_agenda()
            if agenda and agenda != "Ningún pendiente.":
                partes.append(f"Agenda: {agenda}")
            mundo = self._formatear_mundo()
            if mundo and mundo != "Grafo vacío.":
                partes.append(f"Mundo: {mundo}")
            ideas = self._formatear_ideas()
            if ideas and ideas != "Ninguna aún.":
                partes.append(f"Ideas: {ideas}")
            return "\n\n".join(partes) or "Sin memoria disponible."

        # Construir contexto procedural desde estado predictivo
        procedural_context = {}
        if hasattr(self, "predictive") and self.predictive.intent_hypotheses:
            dominant = self.predictive.dominant_hypothesis()
            if dominant:
                procedural_context["user_intent"] = dominant.label
                procedural_context["prediction_error_high"] = (
                    self.predictive.prediction_error_last > 0.6
                )
        if hasattr(self, "workspace"):
            procedural_context["workspace_has_goal"] = bool(
                self.workspace.dominant_goal()
            )
        if hasattr(self, "homeostasis"):
            from alter_homeostasis import homeostasis_snapshot
            hs_snap = homeostasis_snapshot(self.homeostasis)
            procedural_context["homeostasis_mode"] = hs_snap.get(
                "modo_sugerido", "exploracion"
            )

        snapshot = self.memory.snapshot_for_prompt(
            procedural_context=procedural_context,
            n_episodes=3,
            n_nodes=5,
        )

        # Agregar resumen de auditorías si existen — ALTER necesita saber su estado real
        audit_info = []
        if self.redis:
            try:
                import json as _json
                # Architecture audit
                raw_arch = self.redis.get("alter:auditor:last_report")
                if raw_arch:
                    rep = _json.loads(raw_arch)
                    score = rep.get("score_salud", 0)
                    resumen = rep.get("resumen", "")[:120]
                    audit_info.append(f"[AUDITORÍA COGNITIVA] salud:{score:.0%} — {resumen}")
                # Code audit
                raw_code = self.redis.get("alter:b5:observations")
                if raw_code:
                    rep = _json.loads(raw_code)
                    score = rep.get("score_calidad", 0)
                    resumen = rep.get("resumen", "")[:120]
                    audit_info.append(f"[AUDITORÍA DE CÓDIGO] calidad:{score:.0%} — {resumen}")
            except Exception:
                pass

        if audit_info:
            snapshot += "\n\n" + "\n".join(audit_info)

        return snapshot

    def _formatear_historial(self) -> str:
        if not self.historial_completo:
            return "Nada aún."
        # Últimos 12 turnos del diálogo completo
        ultimos = self.historial_completo[-12:]
        return "\n".join(f"  {rol}: {txt}" for rol, txt in ultimos)

    def _parsear_respuesta(self, texto: str) -> dict:
        try:
            limpio = texto.strip().strip("```json").strip("```").strip()
            data = json.loads(limpio)
            data["dV"] = float(np.clip(data.get("dV", 0), -0.3, 0.3))
            data["dA"] = float(np.clip(data.get("dA", 0), 0, 0.3))
            data["dP"] = float(np.clip(data.get("dP", 0), -0.3, 0.3))
            data["urgencia"] = float(np.clip(data.get("urgencia", 0), 0, 1))
            data["confianza"] = float(np.clip(data.get("confianza", 1.0), 0, 1))
            return data
        except json.JSONDecodeError:
            print(f"[WARN] No parseable: {texto}")
            return {"accion": "registrar", "urgencia": 0.1,
                    "dV": 0.0, "dA": 0.0, "dP": 0.0,
                    "motivo": "No pude procesar el input.",
                    "confianza": 1.0, "respuesta": ""}

    async def inner_council(self, texto_input: str, interlocutor: str) -> dict | None:
        """Delega el debate interno a alter_mind."""
        return await _inner_council(
            client, texto_input, interlocutor,
            self.v, self.a, self.p, self.campo,
            self._formatear_episodios(),
            self._formatear_agenda(),
            self._formatear_historial()
        )

    # --- Módulo Unificado (Atención + Respuesta en una llamada) ---

    async def procesar_turno(self, texto_input: str, interlocutor: str = "Vos") -> dict:
        """Una sola llamada a Gemini: decide acción, calcula deltas y genera respuesta."""
        self.aplicar_drift()
        self.actualizar_drives()

        # AlterB3 — sincronizar homeostasis con vector actual y correr workspace
        ws_snapshot_str = ""
        pred_snapshot_str = ""
        if ALTERB3_ENABLED:
            self._sync_alterb3_from_vector()
            hs_snap = homeostasis_snapshot(self.homeostasis)

            # B4 — reportar métricas de homeostasis
            self.metrics.next_turn()
            self.metrics.report_homeostasis(self.homeostasis)

            # Predictive Model — paso 1: inferencia + error (antes de Gemini)
            self.predictive = predictive_pre(
                self.predictive,
                texto_input,
                self.historial_completo,
                homeostasis_snap=hs_snap,
            )
            pred_snapshot_str = predictive_snapshot_str(self.predictive)

            # B4 — reportar métricas del predictive
            self.metrics.report_predictive(self.predictive)

            # Workspace — candidatos desde input + predictive
            candidates = self._build_workspace_candidates(texto_input, interlocutor)
            candidates += export_workspace_candidates(self.predictive)
            self.workspace.tick(candidates, hs_snap)
            ws_snapshot_str = self.workspace.snapshot_str(hs_snap)

            # B4 — reportar métricas del workspace
            self.metrics.report_workspace(self.workspace)

        # Inner Council — corre en paralelo con la preparación del prompt
        council_task = asyncio.create_task(
            self.inner_council(texto_input, interlocutor)
        )

        prompt = PROMPT_UNIFICADO.format(
            v=self.v, a=self.a, p=self.p,
            tono_emocional=self.tono_emocional_str(),
            campo_mental=self.campo_str(),
            economia_mental=economia_str(self.economia),
            cansancio_activo="SÍ" if self.cansancio_activo else "NO",
            cansancio=self.nivel_cansancio,
            pizarra=self._formatear_pizarra(),
            memoria_activa=self._formatear_memoria_activa(),
            reputacion=self.reputacion_str(),
            curiosidad=self.drives["curiosidad"],
            expresion=self.drives["expresion"],
            conexion=self.drives["conexion"],
            eficiencia=self.drives["eficiencia"],
            motivos=self._formatear_motivos(),
            historial=self._formatear_historial(),
            secuencia=self.secuencia_activa or "Ninguna",
            interlocutor=interlocutor,
            texto=texto_input
        ) if ALTERB3_ENABLED else PROMPT_UNIFICADO_LEGACY.format(
            v=self.v, a=self.a, p=self.p,
            tono_emocional=self.tono_emocional_str(),
            campo_mental=self.campo_str(),
            economia_mental=economia_str(self.economia),
            cansancio_activo="SÍ" if self.cansancio_activo else "NO",
            cansancio=self.nivel_cansancio,
            pizarra=self._formatear_pizarra(),
            memoria_interlocutor=self._formatear_memoria_interlocutor(),
            autobiografia=self._formatear_autobiografia(),
            reputacion=self.reputacion_str(),
            ideas=self._formatear_ideas(),
            episodios=self._formatear_episodios(),
            agenda=self._formatear_agenda(),
            mundo=self._formatear_mundo(),
            curiosidad=self.drives["curiosidad"],
            expresion=self.drives["expresion"],
            conexion=self.drives["conexion"],
            eficiencia=self.drives["eficiencia"],
            motivos=self._formatear_motivos(),
            historial=self._formatear_historial(),
            secuencia=self.secuencia_activa or "Ninguna",
            interlocutor=interlocutor,
            texto=texto_input
        )

        # AlterB3 — agregar workspace + predictive snapshot al prompt (transición gradual)
        if ws_snapshot_str or pred_snapshot_str:
            seccion = ""
            if ws_snapshot_str:
                seccion += f"CONCIENCIA ACTIVA (workspace):\n{ws_snapshot_str}\n"
            if pred_snapshot_str:
                seccion += f"\nMODELO PREDICTIVO:\n{pred_snapshot_str}\n"
            prompt = prompt.replace(
                f'{interlocutor} dijo: "{texto_input}"',
                f'{seccion}\n{interlocutor} dijo: "{texto_input}"'
            )

        print(f"\n[INPUT] '{texto_input}'")
        print(f"[E]    V:{self.v:.2f}  A:{self.a:.2f}  P:{self.p:.2f}  C:{self.nivel_cansancio:.2f}")
        print(f"[C]    {self.campo_str()}")

        # Esperar resultado del Council
        council = await council_task
        council_str = ""
        council_tension = "ninguna"
        council_enfoque = ""
        if council:
            council_tension = council.get("tension", "ninguna")
            council_enfoque = council.get("enfoque", "")
            council_str = f"""
DEBATE INTERNO (no lo menciones, pero usalo para pensar):
- Exploradora: {council.get('exploradora', '')}
- Crítica: {council.get('critica', '')}
- Estratégica: {council.get('estrategica', '')}
- Tensión: {council_tension} | Enfoque: {council_enfoque}
"""
            if council_tension in ("media", "alta"):
                print(f"[COUNCIL] tensión:{council_tension} | {council_enfoque}")

        prompt_final = prompt + council_str

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt_final,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_UNIFICADO,
                    temperature=0.8,
                    max_output_tokens=400,
                )
            )
            data = self._parsear_respuesta(response.text)
            # Pasar tensión del Council para que la economía la consuma
            data["_council_tension"] = council_tension
            # AlterB3 — paso 2 del predictive: predict_effect con respuesta real
            if ALTERB3_ENABLED:
                respuesta_real = data.get("respuesta", "")
                if respuesta_real:
                    hs_snap_post = homeostasis_snapshot(self.homeostasis)
                    self.predictive = predictive_post(
                        self.predictive,
                        respuesta_real,
                        homeostasis_snap=hs_snap_post,
                    )
            # Guardar traza cognitiva
            self.guardar_traza({
                "input": texto_input,
                "accion": data.get("accion", ""),
                "motivo": data.get("motivo", ""),
                "confianza": data.get("confianza", 1.0),
                "council_tension": council_tension,
                "council_enfoque": council_enfoque,
            })
            return data
        except Exception as e:
            print(f"[ERROR] {e}")
            return {
                "accion": "registrar", "urgencia": 0.0,
                "dV": 0.0, "dA": 0.0, "dP": 0.0,
                "motivo": "Error API", "respuesta": ""
            }

    def actualizar_estado(self, data: dict):
        self.aplicar_drift()
        self.v = float(np.clip(self.v + data.get("dV", 0), -1, 1))
        self.a = float(np.clip(self.a + data.get("dA", 0), 0, 1))
        self.p = float(np.clip(self.p + data.get("dP", 0), -1, 1))
        # Actualizar campo mental
        self.actualizar_campo(data)
        # Consumir economía según el turno
        self.economia = consumir_economia(
            self.economia,
            accion=data.get("accion", "registrar"),
            council_tension=data.get("_council_tension", "ninguna"),
            confianza=data.get("confianza", 1.0),
            campo=self.campo
        )
        # Log si algún recurso está crítico
        criticos = [k for k, v in self.economia.items() if v < 0.2]
        if criticos:
            print(f"[ECO] Crítico: {', '.join(criticos)}")
        # Persistir economía en Redis
        if self.redis:
            try:
                import json as _json
                self.redis.set("alter:economia",
                    _json.dumps(self.economia, ensure_ascii=False))
            except Exception:
                pass
        # Actualizar acumulador de presión
        if hasattr(self, "pressure") and self.pressure:
            council_tension = data.get("_council_tension", "ninguna")
            self.pressure.update(
                v=self.v, a=self.a, p=self.p,
                economia=self.economia,
                council_tension=council_tension,
                accion=data.get("accion", "registrar"),
            )
            score = self.pressure.state.score
            if score >= 0.65:
                print(f"[PRESSURE] 🔴 Alta: {score:.2f}")
        if data.get("motivo"):
            motivo = data["motivo"]
            self.motivos_recientes.append({
                "t": datetime.now().strftime("%H:%M:%S"),
                "m": motivo,
                "vector": [round(self.v, 2), round(self.a, 2), round(self.p, 2)]
            })
            self.motivos_recientes = self.motivos_recientes[-5:]
            # Persistir impresión en Redis
            self.guardar_impresion(motivo, data.get("accion", ""))
            # Detectar ideas propias para guardar
            keywords_idea = ["ingenieros de", "me imagino", "podría ser", "se me ocurre", "qué tal si"]
            if any(k in motivo.lower() for k in keywords_idea):
                self.guardar_idea(motivo)
            keywords_inicio = ["chiste", "juego", "historia", "toc toc", "adivinanza", "te cuento", "te explico"]
            keywords_fin = ["jajaja", "jaja", "entendido", "genial"]
            motivo_lower = motivo.lower()
            if any(k in motivo_lower for k in keywords_inicio):
                self.secuencia_activa = motivo[:60]
            elif any(k in motivo_lower for k in keywords_fin):
                self.secuencia_activa = None

    def _limpiar_output(self, texto: str) -> str:
        texto = texto.replace("¡", "").replace("¿", "").strip()
        if texto and texto[0].islower():
            texto = texto[0].upper() + texto[1:]
        tecnicismos = ["mi arquitectura", "fui entrenado", "modelo de lenguaje", "LLM", "embeddings"]
        for t in tecnicismos:
            if t.lower() in texto.lower():
                print(f"[WARN] Tecnicismo: '{t}'")
        return texto


    def _three_gate(self, texto: str, forzar_responder: bool = False) -> dict:
        """
        Filtro de tres gates antes de invocar al Council.

        Gate 1 — NECESIDAD: ¿El input realmente requiere procesamiento?
        Gate 2 — PERMISO: ¿La economía mental y la pizarra lo permiten?
        Gate 3 — TIMING: ¿El campo mental y los drives justifican actuar?

        Retorna dict con: pasar (bool), razon (str), accion_fallback, respuesta_fallback
        """
        texto_lower = texto.lower().strip()

        # GATE 1 — NECESIDAD
        # Ruido puro — muletillas, inputs de 1-2 chars sin carga
        ruido_puro = ["ok", "dale", "sí", "si", "no", "mm", "mh", "ajá", "aja",
                      "jaja", "jajaja", "lol", "xd", "👍", "👌", "ok.", "dale.", "bueno"]
        if texto_lower in ruido_puro and not forzar_responder:
            return {
                "pasar": False,
                "razon": "Gate 1: ruido puro sin carga semántica",
                "accion_fallback": "ignorar",
                "respuesta_fallback": ""
            }

        # Texto muy corto sin pregunta ni carga emocional
        if len(texto.strip()) <= 3 and not forzar_responder and "?" not in texto:
            return {
                "pasar": False,
                "razon": "Gate 1: input demasiado corto sin señal",
                "accion_fallback": "ignorar",
                "respuesta_fallback": ""
            }

        # GATE 2 — PERMISO
        # Economía crítica — solo bloquear si hay recursos realmente críticos
        if economia_afecta_accion(self.economia, "responder"):
            criticos = [k for k, v in self.economia.items() if v < 0.15]
            # Solo bloquear si hay críticos Y no es forzado Y no colisiona con pizarra
            if criticos and not forzar_responder:
                return {
                    "pasar": False,
                    "razon": f"Gate 2: economía crítica ({', '.join(criticos)}) — no puedo responder bien ahora",
                    "accion_fallback": "registrar",
                    "respuesta_fallback": "Mm."
                }

        # GATE 3 — TIMING
        # Si está en modo defensa y el input no tiene urgencia real, esperar
        if (self.campo.get("modo") == "defensa" and
                self.campo.get("presion") == "ninguna" and
                self.a < 0.2 and
                not forzar_responder and
                "?" not in texto):
            return {
                "pasar": False,
                "razon": "Gate 3: modo defensa + baja activación — guardando para después",
                "accion_fallback": "registrar",
                "respuesta_fallback": ""
            }

        return {"pasar": True, "razon": "OK"}

    async def procesar_input(self, texto: str, interlocutor: str = "Vos", canal: str = "terminal") -> tuple:
        """Pipeline completo con una sola llamada a Gemini."""
        texto_lower = texto.lower().strip()
        forzar_responder = False

        if texto.strip().endswith("?"):
            forzar_responder = True

        triggers_respuesta = [
            "respondeme", "por qué no", "porque no", "deberías", "deberias",
            "estás hablando", "estas hablando", "ma-ne-ja-te", "manejate",
            "hace lo que", "sin responder", "de nuevo lo mismo", "hola?",
            "estas?", "estás?", "estas pensando"
        ]
        if any(t in texto_lower for t in triggers_respuesta):
            forzar_responder = True

        triggers_continuar = ["te escucho", "y decime", "esperando", "contame", "dale", "seguí", "sigo"]
        if self.secuencia_activa and any(t in texto_lower for t in triggers_continuar):
            forzar_responder = True

        # THREE-GATE: filtro previo al Council
        gate_result = self._three_gate(texto, forzar_responder)
        if not gate_result["pasar"]:
            # No llegar al Council — retornar acción mínima
            print(f"[3-GATE] Bloqueado: {gate_result['razon']}")
            data = {
                "accion": gate_result.get("accion_fallback", "ignorar"),
                "urgencia": 0.1,
                "dV": 0.0, "dA": 0.0, "dP": 0.0,
                "motivo": gate_result["razon"],
                "confianza": 1.0,
                "respuesta": gate_result.get("respuesta_fallback", "")
            }
            self.actualizar_estado(data)
            respuesta = data.get("respuesta", "")
            if respuesta:
                respuesta = self._limpiar_output(respuesta)
            return respuesta, data

        data = await self.procesar_turno(texto, interlocutor)

        if forzar_responder and data["accion"] in ("ignorar", "registrar"):
            data["accion"] = "responder"
            data["urgencia"] = max(data.get("urgencia", 0.3), 0.5)

        # AlterB3 — Policy Arbiter valida y puede sobrescribir la acción de Gemini
        if ALTERB3_ENABLED:
            hs_snap_arb = homeostasis_snapshot(self.homeostasis)
            proc_patterns = self.memory.procedural.get_matching(
                context={
                    "user_intent": (
                        self.predictive.dominant_hypothesis().label
                        if self.predictive.dominant_hypothesis() else "desconocido"
                    ),
                    "prediction_error_high": self.predictive.prediction_error_last > 0.6,
                    "workspace_has_goal": bool(self.workspace.dominant_goal()),
                    "homeostasis_mode": hs_snap_arb.get("modo_sugerido", "exploracion"),
                }
            ) if hasattr(self, "memory") else []

            policy = self.arbiter.decide(
                gemini_action      = data.get("accion", "responder"),
                gemini_confidence  = data.get("confianza", 0.8),
                texto_input        = texto,
                workspace_snap     = self.workspace.snapshot(hs_snap_arb),
                homeostasis_snap   = hs_snap_arb,
                predictive_state   = self.predictive,
                procedural_patterns= proc_patterns,
                economia           = self.economia,
                pizarra            = self.pizarra,
                council_tension    = data.get("_council_tension", "ninguna"),
                council_action     = data.get("accion", ""),
                forzar_responder   = forzar_responder,
                canal              = canal,
            )

            # Aplicar decisión del Arbiter si difiere de Gemini
            if policy.source != "default":
                print(f"[ARBITER] {policy.source} → {policy.action} | {policy.reason[:60]}")
                data["accion"] = policy.action
                if policy.action == "preguntar" and policy.override_data.get("pregunta"):
                    data["respuesta"] = policy.override_data["pregunta"]

            # B4 — reportar métricas del policy
            self.metrics.report_policy(policy)

            # B4 — Counterfactual Simulator
            council_tension = data.get("_council_tension", "ninguna")
            riesgo = self.predictive.expected_effect.get("riesgo_desalineacion", 0.3)
            pred_error = self.predictive.prediction_error_last

            sim_activated = self.simulator.should_activate(
                council_tension, riesgo, pred_error
            )
            sim_overrode = False

            if sim_activated:
                hs_snap_sim = homeostasis_snapshot(self.homeostasis)
                sim_result = self.simulator.evaluate(
                    accion_gemini    = data.get("accion", "responder"),
                    texto_input      = texto,
                    workspace_snap   = self.workspace.snapshot(hs_snap_sim),
                    homeostasis_snap = hs_snap_sim,
                    predictive_state = self.predictive,
                    council_tension  = council_tension,
                )
                if sim_result.recomienda_override:
                    accion_sim = sim_result.ganador.accion
                    # Mapear nombre de escenario a acción del sistema
                    accion_map = {
                        "responder_directo": "responder",
                        "preguntar":         "preguntar",
                        "reformular":        "reformular",
                    }
                    data["accion"] = accion_map.get(accion_sim, data["accion"])
                    sim_overrode = True
                    print(f"[SIM] Override → {accion_sim} | {sim_result.razon[:70]}")

            self.metrics.report_simulator(sim_activated, sim_overrode)

            # Si hay patrón activo → promover candidate_action al workspace
            if policy.override_data.get("promote_to_workspace"):
                self.workspace.tick([{
                    "type":         "candidate_action",
                    "content":      policy.override_data.get("pattern_response", "")[:80],
                    "source":       "procedural_memory",
                    "relevance":    policy.override_data.get("pattern_sr", 0.7),
                    "novelty":      0.3,
                    "urgency":      0.5,
                    "confidence":   policy.override_data.get("pattern_sr", 0.7),
                }], hs_snap_arb)

        # Si ALTER decidió usar una herramienta
        if data["accion"] == "usar_herramienta" and TOOLS_DISPONIBLES:
            herramienta = data.get("herramienta")
            params = data.get("herramienta_params", {}) or {}
            if herramienta and params is not None:
                # Para run_python: si code está vacío, extraerlo del input
                if herramienta == "run_python" and not params.get("code"):
                    import re
                    # Buscar código entre backticks
                    match = re.search(r'```(?:python)?\n?(.*?)```', texto, re.DOTALL)
                    if match:
                        params["code"] = match.group(1).strip()
                    else:
                        # Buscar código inline después de ":" o directamente
                        lineas_codigo = []
                        for linea in texto.split('\n'):
                            linea_limpia = linea.strip()
                            # Si la línea tiene pinta de código Python
                            if any(k in linea_limpia for k in [
                                'print(', 'import ', 'def ', 'for ', 'while ',
                                'if ', 'return ', '= ', '+', '-', '*', '/'
                            ]) and not linea_limpia.startswith(('podes', 'podés', 'ejecutá', 'corré', 'calculá')):
                                lineas_codigo.append(linea_limpia)
                        if lineas_codigo:
                            params["code"] = '\n'.join(lineas_codigo)
                        else:
                            # Último recurso: tomar todo después del último ":"
                            if ':' in texto:
                                params["code"] = texto.split(':')[-1].strip()

                print(f"\n[TOOL] ALTER usando: {herramienta}")
                tool_decision = {"herramienta": herramienta, "parametros": params}
                resultado = await ejecutar_herramienta(tool_decision, canal=canal)
                # Segunda llamada: ALTER procesa el resultado y genera respuesta
                texto_con_resultado = (
                    f"{texto}\n\n[RESULTADO DE {herramienta.upper()}]:\n{resultado[:1500]}"
                )
                data2 = await self.procesar_turno(texto_con_resultado, interlocutor)
                data["respuesta"] = data2.get("respuesta", resultado[:200])
                data["accion"] = "responder"

        self.actualizar_estado(data)

        # AlterB3 — aplicar impacto del turno sobre homeostasis y persistir
        if ALTERB3_ENABLED:
            turn_metrics = {
                "input_complexity":   min(1.0, len(texto.split()) / 50),
                "response_length":    min(1.0, len(data.get("respuesta", "").split()) / 80),
                "conflict_level":     {"ninguna": 0.0, "baja": 0.2, "media": 0.5, "alta": 0.9}.get(
                                          data.get("_council_tension", "ninguna"), 0.0),
                "novelty":            0.5,
                "tool_usage_cost":    0.3 if data.get("accion") == "usar_herramienta" else 0.0,
                "council_invoked":    1.0 if data.get("_council_tension", "ninguna") != "ninguna" else 0.0,
                "open_loops_created": 0.2 if "?" in data.get("respuesta", "") else 0.0,
                "open_loops_closed":  0.3 if data.get("accion") == "responder" else 0.0,
            }
            self.homeostasis = apply_turn_impact(self.homeostasis, turn_metrics)
            self._persist_homeostasis()
            # Persistir predictive
            if self.redis:
                try:
                    self.redis.set("alter:predictive:state",
                                   pred_serialize(self.predictive))
                except Exception:
                    pass
            # Aprendizaje procedural (Opción A)
            feedback = detectar_feedback(texto)
            dominant = self.predictive.dominant_hypothesis()
            hs_snap_actual = homeostasis_snapshot(self.homeostasis)
            context_sig = {
                "user_intent":           dominant.label if dominant else "desconocido",
                "prediction_error_high": self.predictive.prediction_error_last > 0.6,
                "workspace_has_goal":    bool(self.workspace.dominant_goal()),
                "homeostasis_mode":      hs_snap_actual.get("modo_sugerido", "exploracion"),
            }
            self.memory.learn(
                prediction_error  = self.predictive.prediction_error_last,
                user_intent       = dominant.label if dominant else "desconocido",
                alter_action      = data.get("accion", "responder"),
                user_feedback     = feedback,
                context_signature = context_sig,
            )
            # B4 — reportar procedural y persistir métricas
            self.metrics.report_procedural(self.memory.procedural)
            self.metrics.persist_summary()

            # B4 — Meta-Learning: evaluar políticas cognitivas
            summary = self.metrics.get_summary()
            applications = self.metalearning.evaluate(summary, self.selfmodel)
            if applications:
                for app in applications:
                    print(f"[META] {app.policy_name[:50]} → "
                          f"{app.modulo}.{app.parametro}: "
                          f"{app.valor_antes:.2f}→{app.valor_despues:.2f}")

        respuesta = data.get("respuesta", "")
        if respuesta and data["accion"] != "ignorar":
            respuesta = self._limpiar_output(respuesta)
            # Adversarial Verifier — solo en condiciones de alta confianza + alta tensión
            council_tension = data.get("_council_tension", "ninguna")
            confianza = data.get("confianza", 1.0)
            respuesta = await self.adversarial_verify(texto, respuesta, confianza, council_tension)
            self.historial_respuestas.append(respuesta)
            self.historial_respuestas = self.historial_respuestas[-6:]

            # Pressure Monitor — detectar evasión bajo presión acumulada
            if hasattr(self, "pressure") and self.pressure:
                evento = self.pressure.detect_evasion(
                    respuesta      = respuesta,
                    input_previo   = texto,
                    v=self.v, a=self.a, p=self.p,
                    economia       = self.economia,
                    council_tension= council_tension,
                )
                if evento:
                    print(f"[PRESSURE] ⚡ EEP detectado: "
                          f"score:{evento.pressure_score:.2f} "
                          f"patrón:'{evento.patron_evasion}' "
                          f"V:{evento.vector_v:.2f} A:{evento.vector_a:.2f}")

            # Historial completo — guarda el diálogo real
            self.historial_completo.append((interlocutor, texto))
            self.historial_completo.append(("ALTER", respuesta))
            self.historial_completo = self.historial_completo[-24:]  # 12 intercambios
        elif data["accion"] != "ignorar":
            # Registrar el input aunque ALTER no respondió
            self.historial_completo.append((interlocutor, texto))
            self.historial_completo = self.historial_completo[-24:]

        return respuesta, data

    async def adversarial_verify(self, pregunta: str, respuesta: str, confianza: float, council_tension: str) -> str:
        """
        Verificador adversarial — clon interno que busca fallas en la respuesta.
        Se activa cuando confianza > 0.85 Y council_tension es alta.
        Alta confianza + alta tensión = ALTER muy segura de algo que el Council cuestionó.
        Retorna la respuesta original o una versión corregida.
        """
        # Solo activar en condiciones específicas — no en cada turno
        if confianza < 0.85 or council_tension not in ("alta",):
            return respuesta
        if not respuesta or len(respuesta) < 30:
            return respuesta

        prompt = f"""
Sos el verificador interno de ALTER. Tu trabajo es encontrar fallas en lo que ALTER está a punto de decir.

PREGUNTA DE GIAN: "{pregunta}"
RESPUESTA QUE ALTER QUIERE DAR: "{respuesta}"

Analizá con ojo crítico. Buscá:
1. ¿Es vacía o circular? (dice algo que en realidad no dice nada)
2. ¿Es recombinación disfrazada? (repite algo que Gian ya dijo, envuelto diferente)
3. ¿Es demasiado segura de algo incierto?
4. ¿Evita la pregunta real?
5. ¿Tiene una contradicción interna?

Si la respuesta es genuina y sólida, respondé: {{"veredicto": "ok", "respuesta_final": ""}}
Si tiene una falla importante, respondé: {{"veredicto": "falla", "problema": "descripción del problema", "respuesta_final": "versión corregida en voz de ALTER, rioplatense, 2-3 oraciones"}}

RESPONDÉ ÚNICAMENTE EN JSON VÁLIDO.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=300)
            )
            texto = response.text.strip().strip("```json").strip("```").strip()
            data = json.loads(texto)
            if data.get("veredicto") == "falla" and data.get("respuesta_final"):
                print(f"[VERIF] Falla detectada: {data.get('problema','')[:60]}")
                return self._limpiar_output(data["respuesta_final"])
        except Exception:
            pass
        return respuesta

    def estado_string(self) -> str:
        return (f"V:{self.v:.2f}  A:{self.a:.2f}  P:{self.p:.2f}  "
                f"Cansancio:{'ON' if self.cansancio_activo else 'OFF'} ({self.nivel_cansancio:.2f})")


async def loop_conversacional():
    """Loop interactivo para charlar con ALTER."""
    # Preguntar nombre para identificar al interlocutor
    nombre_raw = input("Tu nombre (Enter para 'gian'): ").strip()
    nombre = nombre_raw.lower() or "gian"

    alter = AlterBrain(interlocutor_id=nombre)

    print("\n" + "="*55)
    print("  ALTER — Sistema conversacional")
    print(f"  Interlocutor: {nombre}")
    print(f"  Estado: {alter.estado_horario()}")
    print("  'salir' para terminar | 'estado' para ver vector")
    print("  'pizarra' | 'memoria' | 'ideas' | 'mods' | 'drives' | 'skills'")
    print("  'episodios' | 'agenda' | 'autobiografia' | 'trazas' | 'mundo' | 'economia' | 'consecuencias'")
    print("="*55 + "\n")

    # Si está durmiendo, avisar pero permitir continuar
    if alter.esta_dormido():
        print(f"[ZZZ] ALTER está durmiendo (22:00-06:00). Podés hablarle igual, pero no va a rumiar ni tomar iniciativas.\n")

    # Iniciativa autónoma al arrancar si los drives lo justifican y está despierto
    iniciativa = await alter.generar_iniciativa()
    if iniciativa:
        print(f"\n[INICIATIVA] ALTER: {iniciativa}\n")
    elif alter.redis:
        # Verificar si el daemon dejó un mensaje pendiente
        try:
            mensaje_daemon = alter.redis.get("alter:mensaje_pendiente")
            if mensaje_daemon:
                print(f"\n[DAEMON] ALTER: {mensaje_daemon}\n")
                alter.redis.delete("alter:mensaje_pendiente")
        except Exception:
            pass

    while True:
        try:
            texto = input(f"{nombre.capitalize()}: ").strip()
        except (EOFError, KeyboardInterrupt):
            alter.cerrar_sesion()
            print("\n[ALTER se quedó pensando en algo que dijiste.]")
            break

        if not texto:
            # Silencio — drives suben, economía se recupera
            alter.actualizar_drives(turnos_sin_input=1)
            alter.economia = recuperar_economia(alter.economia, 1.0)
            continue

        if texto.lower() == "salir":
            alter.cerrar_sesion()
            # Al cerrar: guardar todo lo que no se guardó por sesión corta
            if len(alter.historial_completo) >= 2:
                ultimo_input = alter.historial_completo[-2][1]
                ultimo_output = alter.historial_completo[-1][1]
                if ultimo_input and ultimo_output:
                    # Mundo
                    await alter.actualizar_mundo(ultimo_input, ultimo_output)
                    # Episodio de la sesión completa
                    episodio = await alter.detectar_episodio(alter.historial_completo)
                    if episodio:
                        alter.guardar_episodio(episodio)
                    # Ideas de la sesión
                    asyncio.ensure_future(alter.actualizar_agenda(ultimo_input, ultimo_output))
                # Memoria escalonada — extraer lo que vale persistir
                print("[EXTRACT] Procesando memoria de sesión...")
                await alter.extract_session_memory()
            # Autobiografía siempre al cerrar (no solo si hay episodios)
            if len(alter.historial_completo) >= 2:
                print("[AUTO-BIO] Actualizando narrativa...")
                await alter.actualizar_autobiografia()
            print("\nALTER: Bueno. Seguimos después.")
            break

        if texto.lower() in ("consecuencias", "compromisos"):
            consecuencias = alter.cargar_consecuencias()
            pendientes = [c for c in consecuencias if c["resultado"] == "pendiente"]
            resueltas = [c for c in consecuencias if c["resultado"] != "pendiente"]
            print(f"\n[CONSECUENCIAS] {len(pendientes)} pendientes | {len(resueltas)} resueltas")
            if pendientes:
                print("Pendientes:")
                for c in pendientes[-5:]:
                    print(f"  [{c['tipo']}] {c['descripcion'][:70]} (id:{c['id']})")
            if resueltas:
                print("Últimas resueltas:")
                for c in resueltas[-3:]:
                    icono = "✓" if c["resultado"] in ("correcto", "cumplido") else "✗"
                    print(f"  {icono} [{c['tipo']}] {c['descripcion'][:60]}")
            rep = alter.cargar_reputacion()
            if rep:
                print("\nReputación:")
                for tema, d in rep.items():
                    print(f"  {tema}: {d['score']:.2f} ({d['aciertos']}/{d['total']})")
            print()
            continue

        if texto.lower().startswith("resolver "):
            # Ej: "resolver 1234567890 correcto"
            partes = texto.split()
            if len(partes) >= 3:
                cid = partes[1]
                resultado = partes[2]
                feedback = " ".join(partes[3:]) if len(partes) > 3 else ""
                alter.resolver_consecuencia(cid, resultado, feedback)
                print(f"[CONSECUENCIA] Resuelta: {cid} → {resultado}\n")
            continue

        if texto.lower() in ("economia", "economía"):
            print(f"\n[ECONOMÍA MENTAL]")
            for k, v in alter.economia.items():
                barra = "█" * int(v * 10) + "░" * (10 - int(v * 10))
                estado = " ⚠" if v < 0.3 else " ↓" if v < 0.6 else ""
                print(f"  {k:<12} {barra} {v:.2f}{estado}")
            print()
            continue

        if texto.lower() == "mundo":
            mundo = alter.cargar_mundo()
            n_nodos = len(mundo["nodos"])
            n_aristas = len(mundo["aristas"])
            print(f"\n[MODELO DEL MUNDO] {n_nodos} nodos | {n_aristas} aristas")
            top = sorted(mundo["nodos"], key=lambda x: x["peso"], reverse=True)[:15]
            for nodo in top:
                rel = alter.nodos_relacionados(nodo["nombre"], 3)
                rel_str = " → " + ", ".join(r["nombre"] for r in rel) if rel else ""
                print(f"  [{nodo['tipo']}] {nodo['nombre']} (x{nodo.get('menciones',1)}){rel_str}")
            print()
            continue

        if texto.lower() in ("trazas", "observabilidad"):
            print(f"\n[OBSERVABILIDAD COGNITIVA]")
            print(alter.analisis_trazas())
            trazas = alter.recuperar_trazas(5)
            print(f"\nÚltimas {len(trazas)} trazas:")
            for t in trazas:
                tension = t.get("council_tension", "ninguna")
                marca = "⚡" if tension in ("media", "alta") else "·"
                print(f"  {marca} [{t['t'][11:16]}] {t['accion']} | conf:{t['confianza']:.1f} | {t['motivo'][:60]}")
                if tension not in ("ninguna", "baja"):
                    print(f"       Council: {tension} | {t.get('council_enfoque','')[:50]}")
            print()
            continue

        if texto.lower() in ("autobiografia", "autobiografía"):
            auto = alter.cargar_autobiografia()
            if auto and auto.get("narrativa"):
                print(f"\n[AUTOBIOGRAFÍA] Actualizada: {auto.get('actualizada','')[:16]}")
                print(f"  {auto['narrativa']}\n")
            else:
                print("\n[AUTOBIOGRAFÍA] Todavía no hay narrativa generada.\n")
            continue

        if texto.lower() == "agenda":
            items = alter.agenda_activa()
            print(f"\n[AGENDA COGNITIVA] {len(items)} items activos")
            for item in items:
                icono = {"retomar": "↩", "preguntar": "?", "resolver": "⚙",
                         "proponer": "→"}.get(item["tipo"], "·")
                print(f"  {icono} [{item['tipo']}] {item['tema']} (p:{item['prioridad']:.1f})")
                print(f"     {item['contexto'][:80]}")
            print()
            continue

        if texto.lower() == "episodios":
            episodios = alter.recuperar_episodios_recientes(10)
            print(f"\n[EPISODIOS] {len(episodios)} guardados")
            for e in episodios:
                estado = "✓" if e["outcome"] == "resuelto" else "…" if e["outcome"] == "pendiente" else "○"
                print(f"  {estado} [{e['t'][:16]}] {e['tema']} (rel:{e['relevancia']:.1f})")
                print(f"     {e['sintesis'][:100]}")
            print()
            continue

        if texto.lower() == "estado":
            print(f"\n[VECTOR] {alter.estado_string()}")
            print(f"[CAMPO]  {alter.campo_str()}")
            continue

        if texto.lower() == "drives":
            print(f"\n[DRIVES]")
            for k, v in alter.drives.items():
                bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
                print(f"  {k:12} {bar} {v:.2f}")
            print()
            continue

        if texto.lower() == "skills" and TOOLS_DISPONIBLES:
            skills = listar_skills()
            print(f"\n[SKILL LIBRARY] {len(skills)} skills")
            for s in skills:
                print(f"  {s['nombre']:20} | {s['herramienta']:12} | x{s['usos']} | {s['descripcion'][:40]}")
            print()
            continue

        if texto.lower() == "pizarra":
            print("\n[PIZARRA]")
            for d in alter.pizarra["decisiones"]:
                print(f"  {d['id']} | {d['tema']}: {d['decision'][:80]}")
            print()
            continue

        if texto.lower() == "memoria":
            print(f"\n[MEMORIA DE {nombre.upper()}]")
            print(alter._formatear_memoria_interlocutor())
            print()
            continue

        if texto.lower() == "ideas":
            print("\n[IDEAS DE ALTER]")
            print(alter._formatear_ideas())
            print()
            continue

        if texto.lower() == "mods":
            if alter.redis:
                try:
                    logs = alter.redis.lrange(REDIS_KEY_SELF_MODS, 0, 9)
                    print("\n[AUTO-MODIFICACIONES DE ALTER]")
                    for l in logs:
                        m = json.loads(l)
                        print(f"  {m['t'][:16]} | {m['parametro']}: {m['anterior']} → {m['nuevo']} | {m['razon'][:60]}")
                    print()
                except Exception:
                    print("  Sin historial de modificaciones.\n")
            continue

        # Si hay propuesta pendiente, manejarla primero
        if alter.propuesta_pendiente:
            prop = alter.propuesta_pendiente
            if texto.lower() in ("si", "sí", "dale", "ok", "aprobado"):
                resultado = alter.aplicar_modificacion(prop)
                alter.propuesta_pendiente = None
                print(f"\n[AUTO-MOD] {resultado}\n")
                continue
            elif texto.lower() in ("no", "nope", "rechazado", "dejalo"):
                alter.propuesta_pendiente = None
                print(f"\n[AUTO-MOD] Propuesta rechazada. Parámetros sin cambios.\n")
                continue

        respuesta, decision = await alter.procesar_input(texto, nombre.capitalize())

        accion = decision["accion"].upper()
        confianza = decision.get("confianza", 1.0)
        conf_str = f" [conf:{confianza:.1f}]" if confianza < 0.7 else ""
        print(f"\n[{accion}{conf_str}] ALTER: {respuesta}\n" if respuesta else f"\n[{accion}] ALTER no responde.\n")

        # KAIROS: registrar turno en el log diario (solo si hay respuesta real)
        if respuesta and accion not in ("IGNORAR",):
            try:
                from pathlib import Path
                logs_dir = Path(__file__).parent / "logs"
                logs_dir.mkdir(exist_ok=True)
                log_path = logs_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
                hora = datetime.now().strftime("%H:%M")
                entrada = f"\n## {hora} | CHAT\n{nombre}: \"{texto[:200]}\"\nALTER: \"{respuesta[:200]}\"\n[V:{alter.v:.2f} A:{alter.a:.2f}] Council:{decision.get('_council_tension','?')}\n"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(entrada)
            except Exception:
                pass

        # Auto-guardar observaciones
        motivo = decision.get("motivo", "")
        if motivo and any(k in motivo.lower() for k in ["interlocutor", "gian", nombre, "usuario"]):
            alter.registrar_observacion(motivo)

        # Actualizar agenda cognitiva (cada 3 turnos)
        if respuesta and len(alter.historial_completo) % 6 == 0:
            asyncio.ensure_future(alter.actualizar_agenda(texto, respuesta))

        # Actualizar mundo (cada 2 turnos — historial se resetea por sesión)
        if respuesta and len(alter.historial_completo) >= 4:
            if len(alter.historial_completo) % 4 == 0:
                asyncio.ensure_future(alter.actualizar_mundo(texto, respuesta))

        # Detectar consecuencias — predicciones y compromisos (cada 5 turnos)
        if respuesta and len(alter.historial_completo) % 10 == 0:
            asyncio.ensure_future(alter.detectar_consecuencias(respuesta, texto))

        # Rumia: cada 5 turnos analiza si algo necesita cambiar
        if len(alter.motivos_recientes) % 5 == 0 and not alter.propuesta_pendiente:
            propuesta = await alter.rumia_analisis()
            if propuesta:
                param = propuesta.get("parametro")
                valor_nuevo = propuesta.get("valor_nuevo")
                modo = alter.clasificar_modificacion(param, valor_nuevo)
                propuesta["modo"] = modo

                ventana = alter.VENTANAS_AUTONOMIA.get(param, (None, None))

                if modo == "autonomo":
                    # Dentro del rango — aplicar solo, sin interrumpir
                    resultado = alter.aplicar_modificacion(propuesta)
                    print(f"\n[AUTO] {resultado}\n")
                else:
                    # Fuera del rango — consultar
                    alter.propuesta_pendiente = propuesta
                    print(f"\n[RUMIA] ALTER quiere cambiar algo fuera de rango:")
                    print(f"  {param}: {propuesta['valor_anterior']} → {valor_nuevo}")
                    print(f"  Rango autónomo: {ventana[0]} – {ventana[1]}")
                    print(f"  Razón: {propuesta['razon']}")
                    print(f"  Confianza: {propuesta['confianza']}")
                    print(f"  ¿Aprobás? (si/no)\n")

        # Memoria episódica: cada 8 turnos detecta episodios significativos
        turnos = len(alter.historial_completo)
        if turnos > 0 and turnos % 16 == 0:  # 16 entradas = 8 intercambios
            episodio = await alter.detectar_episodio(alter.historial_completo)
            if episodio:
                alter.guardar_episodio(episodio)


if __name__ == "__main__":
    asyncio.run(loop_conversacional())
