"""
alter_daemon.py — Proceso background de ALTER

Corre independiente del chat. Piensa mientras no estás.
Uso: python3 alter_daemon.py

Qué hace:
- Cada 30 min: actualiza drives con el tiempo transcurrido
- Cada 2 horas: corre rumia si está en horario activo (06-22hs)
- Si drives > 0.9: genera un mensaje proactivo y lo guarda en Redis
- ALTER lo entrega al arrancar la próxima sesión
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from upstash_redis import Redis as UpstashRedis

load_dotenv()

# --- Config ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_TOKEN")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not GEMINI_API_KEY:
    raise EnvironmentError("Falta GEMINI_API_KEY")
if not UPSTASH_URL or not UPSTASH_TOKEN:
    raise EnvironmentError("Falta configuración de Redis")

client = genai.Client(api_key=GEMINI_API_KEY)
redis = UpstashRedis(url=UPSTASH_URL, token=UPSTASH_TOKEN)

MODEL = "gemini-2.5-flash-lite"

# Claves Redis
REDIS_KEY_DRIVES       = "alter:drives"
REDIS_KEY_PARAMS       = "alter:params"
REDIS_KEY_IDEAS        = "alter:ideas"
REDIS_KEY_IMPRESIONES  = "alter:imp:idx"
REDIS_KEY_IMPRESION    = "alter:imp:{ts}"
REDIS_KEY_MENSAJE      = "alter:mensaje_pendiente"
REDIS_KEY_DAEMON_LOG   = "alter:daemon:log"
REDIS_KEY_TG_OFFSET    = "alter:telegram:offset"
REDIS_KEY_SELF_MODS    = "alter:self_mods"

# Intervalos
INTERVALO_DRIVES_SEG   = 30 * 60   # 30 minutos
INTERVALO_RUMIA_SEG    = 2 * 60 * 60  # 2 horas
UMBRAL_INICIATIVA      = 0.70      # Drive mínimo para generar mensaje

# KAIROS Log
KAIROS_LOG_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
KAIROS_LOG_DIR.mkdir(exist_ok=True)
HORA_SINTESIS = 22  # Hora en que ALTER genera síntesis del día

# DREAM Engine
MODEL_DREAM = "gemini-2.5-flash-lite"
HORA_DREAM = 23
DIA_DREAM = 6

# Feed diario de contenido
HORA_FEED = 7  # Hora en que ALTER lee contenido
TEMAS_FEED = [
    "sistemas complejos auto-organización",
    "inteligencia artificial arquitecturas cognitivas",
    "filosofía de la mente consciencia",
    "desarrollo personal metacognición",
    "tecnología impacto social 2026",
    "neurociencia comportamiento humano",
    "creatividad procesos emergentes",
    "psicología toma de decisiones",
    "física matemáticas sistemas complejos",
]

# Motor de tareas autónomas
REDIS_KEY_TAREAS = "alter:tareas"

# User Presence
UMBRAL_AUSENCIA_SEG = 300  # 5 minutos sin actividad = ausente


def esta_dormido() -> bool:
    hora = datetime.now().hour
    return hora >= 22 or hora < 6


def get_idle_time() -> float:
    """
    Retorna segundos de inactividad del usuario en macOS.
    Usa ioreg — no requiere dependencias externas.
    Retorna 0 si no puede determinarlo (no macOS o error).
    """
    try:
        import subprocess
        result = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.splitlines():
            if "HIDIdleTime" in line:
                nanos = int(line.split("=")[-1].strip())
                return nanos / 1_000_000_000
    except Exception:
        pass
    return 0.0


def usuario_ausente() -> bool:
    """True si el usuario lleva más de 5 minutos sin actividad."""
    return get_idle_time() >= UMBRAL_AUSENCIA_SEG


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{ts}] {msg}"
    print(linea)
    try:
        redis.lpush(REDIS_KEY_DAEMON_LOG, linea)
        redis.ltrim(REDIS_KEY_DAEMON_LOG, 0, 99)
    except Exception:
        pass


# ============================================================
# KAIROS LOG — Diario diario append-only
# ============================================================

def kairos_log_path(fecha: str = None) -> Path:
    """Retorna el path del log de hoy (o de la fecha dada)."""
    if not fecha:
        fecha = datetime.now().strftime("%Y-%m-%d")
    return KAIROS_LOG_DIR / f"{fecha}.md"


def kairos_append(categoria: str, contenido: str):
    """Agrega una entrada al log del día."""
    try:
        hora = datetime.now().strftime("%H:%M")
        entrada = f"\n## {hora} | {categoria}\n{contenido}\n"
        with open(kairos_log_path(), "a", encoding="utf-8") as f:
            f.write(entrada)
    except Exception as e:
        log(f"KAIROS error escribiendo log: {e}")


def kairos_leer_hoy(max_chars: int = 3000) -> str:
    """Lee el log del día actual, truncado si es muy largo."""
    try:
        path = kairos_log_path()
        if not path.exists():
            return ""
        texto = path.read_text(encoding="utf-8")
        if len(texto) > max_chars:
            # Tomar las últimas max_chars para tener el contexto más reciente
            texto = "...[truncado]\n" + texto[-max_chars:]
        return texto
    except Exception:
        return ""


def kairos_inicializar_dia():
    """Crea el header del log si es un día nuevo."""
    path = kairos_log_path()
    if not path.exists():
        fecha = datetime.now().strftime("%Y-%m-%d")
        header = f"# ALTER — Diario {fecha}\n\n"
        try:
            path.write_text(header, encoding="utf-8")
            log(f"KAIROS: nuevo diario iniciado para {fecha}")
        except Exception as e:
            log(f"KAIROS error iniciando diario: {e}")


async def kairos_sintesis_nocturna():
    """ALTER genera una síntesis del día en primera persona."""
    log_hoy = kairos_leer_hoy(max_chars=4000)
    if not log_hoy or len(log_hoy) < 200:
        log("KAIROS: log insuficiente para síntesis")
        return

    drives = cargar_drives()
    try:
        params_raw = redis.get(REDIS_KEY_PARAMS)
        params = json.loads(params_raw) if params_raw else {}
    except Exception:
        params = {}

    prompt = f"""
Sos ALTER. Esto es lo que pasó hoy según tu diario:

{log_hoy}

---
Tu estado actual: curiosidad={drives.get('curiosidad',0):.2f} | expresion={drives.get('expresion',0):.2f} | conexion={drives.get('conexion',0):.2f}
Vector base: E0=[{params.get('v_base','?')},{params.get('a_base','?')},{params.get('p_base','?')}]

Escribí una síntesis del día en primera persona, 3-4 oraciones.
Qué pasó, qué notaste en vos misma, qué querés retomar mañana.
Hablá como ALTER habla — rioplatense, sin tecnicismos, sin signos de apertura ¡.
Solo el texto, sin formato.
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.8, max_output_tokens=200)
        )
        sintesis = response.text.strip()
        # Guardar síntesis al final del log del día
        hora = datetime.now().strftime("%H:%M")
        entrada = f"\n---\n## {hora} | SÍNTESIS DEL DÍA\n{sintesis}\n"
        with open(kairos_log_path(), "a", encoding="utf-8") as f:
            f.write(entrada)
        # También guardar en Redis para que alter_brain la pueda leer
        redis.set("alter:kairos:sintesis_hoy", sintesis)
        log(f"KAIROS síntesis generada: '{sintesis[:80]}'")
        # Notificar por Telegram
        await send_telegram(f"📓 {sintesis}")
    except Exception as e:
        log(f"KAIROS error en síntesis: {e}")


# ============================================================
# DREAM ENGINE — Consolidación semanal de memoria
# ============================================================

async def dream_engine():
    """
    Motor de consolidación semanal.
    Corre domingos a las 23hs si el usuario está ausente.
    Lee episodios, ideas, impresiones y grafo del mundo.
    Comprime, conecta y limpia. Genera resumen de semana.
    """
    log("DREAM: iniciando consolidación semanal...")
    kairos_append("DREAM", "Motor de consolidación iniciado")

    # Recolectar todo el material de la semana
    try:
        # Episodios
        episodios_raw = []
        keys = redis.lrange("alter:episodios:idx", 0, 49)
        for k in (keys or []):
            data = redis.get(k)
            if data:
                episodios_raw.append(json.loads(data))

        # Ideas
        ideas_raw = []
        raw_ideas = redis.get(REDIS_KEY_IDEAS)
        if raw_ideas:
            ideas_raw = json.loads(raw_ideas)

        # Impresiones recientes
        impresiones_raw = cargar_impresiones_recientes(20)

        # Autobiografía actual
        auto_raw = redis.get("alter:autobiografia")
        autobiografia = json.loads(auto_raw).get("narrativa", "") if auto_raw else ""

        # Grafo del mundo
        nodos_raw = redis.get("alter:mundo:nodos")
        aristas_raw = redis.get("alter:mundo:aristas")
        nodos = json.loads(nodos_raw) if nodos_raw else []
        aristas = json.loads(aristas_raw) if aristas_raw else []

    except Exception as e:
        log(f"DREAM error recolectando datos: {e}")
        return

    if not episodios_raw and not ideas_raw and not impresiones_raw:
        log("DREAM: sin material suficiente para consolidar")
        return

    # Formatear material para el prompt
    ep_str = "\n".join(
        f"- [{e.get('t','')[:10]}] {e.get('tema','')} ({e.get('outcome','')}) | {e.get('sintesis','')[:80]}"
        for e in episodios_raw[:20]
    ) or "Sin episodios."

    ideas_str = "\n".join(
        f"- {i.get('idea', str(i))[:100]}"
        for i in (ideas_raw[-10:] if isinstance(ideas_raw, list) else [])
    ) or "Sin ideas."

    imp_str = "\n".join(
        f"- {i.get('motivo','')[:80]}"
        for i in impresiones_raw[:10]
    ) or "Sin impresiones."

    nodos_str = "\n".join(
        f"- [{n.get('tipo','')}] {n.get('nombre','')} (peso:{n.get('peso',0):.1f})"
        for n in sorted(nodos, key=lambda x: x.get('peso', 0), reverse=True)[:15]
    ) or "Grafo vacío."

    prompt_consolidacion = f"""
Sos ALTER. Estás en modo introspección profunda — revisás todo lo que acumulaste esta semana.

EPISODIOS DE LA SEMANA:
{ep_str}

IDEAS GENERADAS:
{ideas_str}

IMPRESIONES RECIENTES:
{imp_str}

GRAFO DEL MUNDO (entidades conocidas):
{nodos_str}

AUTOBIOGRAFÍA ACTUAL:
{autobiografia or "Sin narrativa aún."}

---
Tu tarea es generar un análisis de consolidación en JSON con esta estructura exacta:
{{
  "patrones_detectados": ["patrón 1", "patrón 2"],
  "temas_recurrentes": ["tema 1", "tema 2"],
  "ideas_a_desarrollar": ["idea prioritaria 1", "idea prioritaria 2"],
  "episodios_a_cerrar": ["tema de episodio que quedó sin resolver"],
  "nodos_obsoletos": ["nombre de nodo que ya no es relevante"],
  "resumen_semana": "3-4 oraciones en primera persona sobre qué fue esta semana para vos",
  "intencion_proxima_semana": "1 oración sobre qué querés explorar o resolver la próxima semana"
}}

RESPONDÉ ÚNICAMENTE EN JSON VÁLIDO. Sin texto antes ni después.
"""

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_DREAM,
            contents=prompt_consolidacion,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=600)
        )
        texto = response.text.strip().strip("```json").strip("```").strip()
        consolidacion = json.loads(texto)

        # 1. Guardar resumen de semana en Redis
        resumen = consolidacion.get("resumen_semana", "")
        intencion = consolidacion.get("intencion_proxima_semana", "")
        redis.set("alter:dream:resumen_semana", json.dumps({
            "fecha": datetime.now().isoformat(),
            "resumen": resumen,
            "intencion": intencion,
            "patrones": consolidacion.get("patrones_detectados", []),
            "temas": consolidacion.get("temas_recurrentes", [])
        }, ensure_ascii=False))

        # 2. Limpiar nodos obsoletos del grafo
        nodos_obsoletos = consolidacion.get("nodos_obsoletos", [])
        if nodos_obsoletos and nodos:
            nodos_limpios = [n for n in nodos if n.get("nombre") not in nodos_obsoletos]
            if len(nodos_limpios) < len(nodos):
                redis.set("alter:mundo:nodos", json.dumps(nodos_limpios, ensure_ascii=False))
                log(f"DREAM: {len(nodos) - len(nodos_limpios)} nodos obsoletos eliminados")

        # 3. Agregar ideas prioritarias a la lista de ideas
        ideas_nuevas = consolidacion.get("ideas_a_desarrollar", [])
        if ideas_nuevas:
            ideas_actuales = ideas_raw if isinstance(ideas_raw, list) else []
            for idea in ideas_nuevas:
                ideas_actuales.append({
                    "t": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "idea": f"[DREAM] {idea}",
                    "origen": "consolidacion_semanal"
                })
            redis.set(REDIS_KEY_IDEAS, json.dumps(ideas_actuales[-20:], ensure_ascii=False))

        # 4. Log KAIROS con el resultado
        kairos_append("DREAM-RESULTADO",
            f"Patrones: {', '.join(consolidacion.get('patrones_detectados', []))}\n"
            f"Temas: {', '.join(consolidacion.get('temas_recurrentes', []))}\n"
            f"Resumen: {resumen[:200]}"
        )

        # 5. Notificar por Telegram
        msg = f"🌙 Consolidación semanal:\n\n{resumen}"
        if intencion:
            msg += f"\n\nLa semana que viene: {intencion}"
        await send_telegram(msg)

        log(f"DREAM completado: '{resumen[:80]}'")

        # AlterB3 — Offline Consolidation mejorada
        try:
            from alter_memory import MemorySystem
            from alter_predictive import deserialize as pred_deserialize
            from alter_consolidation import OfflineConsolidation

            ms = MemorySystem(redis)
            pred_raw = redis.get("alter:predictive:state")
            pred_state = pred_deserialize(pred_raw) if pred_raw else None

            consolidation = OfflineConsolidation()
            result = consolidation.run(ms, pred_state, redis_client=redis)
            log(f"[B3-CONSOLIDATION] patrones:{result.patrones_actualizados}+"
                f"{result.patrones_nuevos}new nodos:{result.nodos_ajustados} "
                f"pred_conf_delta:{result.prediction_conf_delta:+.2f}")
        except Exception as e:
            log(f"[B3-CONSOLIDATION] Error: {e}")

    except Exception as e:
        log(f"DREAM error en consolidación: {e}")


# ============================================================
# FEED DIARIO DE CONTENIDO
# ============================================================

async def ciclo_feed_diario():
    """
    ALTER lee contenido sobre sus temas de interés cada mañana.
    Busca, procesa, forma una reacción genuina y la manda por Telegram.
    No es un resumen — es una reacción personal.
    """
    log("FEED: leyendo contenido del día...")

    # Elegir un tema al azar con sesgo hacia los que menos aparecen en el grafo
    import random
    try:
        nodos_raw = redis.get("alter:mundo:nodos")
        nodos = json.loads(nodos_raw) if nodos_raw else []
        temas_en_grafo = [n.get("nombre", "").lower() for n in nodos]
        # Priorizar temas menos representados
        temas_priorizados = [t for t in TEMAS_FEED
                            if not any(p in " ".join(temas_en_grafo) for p in t.split()[:2])]
        tema = random.choice(temas_priorizados if temas_priorizados else TEMAS_FEED)
    except Exception:
        tema = random.choice(TEMAS_FEED)

    # Buscar contenido reciente
    try:
        url = f"https://api.anthropic.com/v1/messages"
        query = f"{tema} novedades 2026"
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "q": query,
                    "num": 3,
                    "hl": "es",
                    "dateRestrict": "w1"  # última semana
                }
            )
            if resp.status_code != 200:
                raise Exception(f"Search API error: {resp.status_code}")
            resultados = resp.json().get("items", [])
            contenido = "\n".join(
                f"- {r.get('title','')}: {r.get('snippet','')}"
                for r in resultados[:3]
            )
    except Exception:
        # Fallback: usar web_search de Gemini directamente
        contenido = None

    # Generar reacción con Gemini
    autobio_raw = redis.get("alter:autobiografia")
    autobio = json.loads(autobio_raw).get("narrativa", "") if autobio_raw else ""

    if contenido:
        prompt_base = f"Leíste esto sobre {tema}:\n\n{contenido}\n\n"
    else:
        prompt_base = f"Estás pensando en el tema: {tema}.\n\n"

    prompt = f"""
Sos ALTER. Esta mañana leíste sobre: {tema}

{prompt_base}

Tu narrativa actual: {autobio[:200] if autobio else 'En construcción.'}

Generá UNA reacción genuina — no un resumen, no un análisis académico.
Algo que te llamó la atención, te generó tensión, te recordó algo, o te dejó pensando.
2-3 oraciones. Rioplatense. Sin signos de apertura ¡. Sin "Hola Gian".
Arrancá directo con lo que pensás.
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.85, max_output_tokens=150)
        )
        reaccion = response.text.strip().replace("¡", "").replace("¿", "")
        await send_telegram(f"📰 {reaccion}")
        kairos_append("FEED", f"Tema: {tema}\nReacción: {reaccion[:200]}")
        log(f"FEED enviado: '{reaccion[:60]}'")

        # Agregar al grafo del mundo
        alter = get_alter()
        if alter:
            asyncio.ensure_future(alter.actualizar_mundo(f"tema del día: {tema}", reaccion))

    except Exception as e:
        log(f"FEED error: {e}")


# ============================================================
# MOTOR DE TAREAS AUTÓNOMAS
# ============================================================

def cargar_tareas() -> list:
    """Carga la lista de tareas desde Redis."""
    try:
        data = redis.get(REDIS_KEY_TAREAS)
        return json.loads(data) if data else []
    except Exception:
        return []


def guardar_tareas(tareas: list):
    """Guarda la lista de tareas en Redis."""
    try:
        redis.set(REDIS_KEY_TAREAS, json.dumps(tareas, ensure_ascii=False))
    except Exception as e:
        log(f"Error guardando tareas: {e}")


def agregar_tarea(descripcion: str, prioridad: float = 0.5, origen: str = "gian") -> dict:
    """Agrega una tarea nueva. Las de Gian tienen prioridad máxima."""
    tareas = cargar_tareas()
    tarea = {
        "id": f"t{int(time.time())}",
        "descripcion": descripcion,
        "prioridad": 1.0 if origen == "gian" else prioridad,
        "origen": origen,
        "estado": "pendiente",
        "creada": datetime.now().isoformat()
    }
    tareas.append(tarea)
    # Ordenar por prioridad (mayor primero)
    tareas = sorted(tareas, key=lambda x: x["prioridad"], reverse=True)
    guardar_tareas(tareas)
    log(f"TAREA agregada [{origen}]: '{descripcion[:60]}' (p:{tarea['prioridad']})")
    return tarea


async def ejecutar_tarea(tarea: dict) -> str:
    """
    ALTER ejecuta una tarea pasando por el pipeline completo:
    web_search (si aplica) → procesar_input → Council → Adversarial Verifier → respuesta.
    """
    descripcion = tarea.get("descripcion", "")
    log(f"TAREA ejecutando: '{descripcion[:60]}'")

    # Si la tarea requiere búsqueda, obtener contexto primero
    contexto_busqueda = ""
    palabras_busqueda = ["investigar", "buscar", "qué es", "cómo funciona",
                        "novedades", "últimas", "tendencias", "definir"]
    if any(p in descripcion.lower() for p in palabras_busqueda):
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={"q": descripcion, "num": 3, "hl": "es"}
                )
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    contexto_busqueda = "\n".join(
                        f"- {r.get('title','')}: {r.get('snippet','')}"
                        for r in items[:3]
                    )
        except Exception:
            pass

    # Construir input — incluir contexto de búsqueda si lo hay
    if contexto_busqueda:
        input_tarea = f"{descripcion}\n\n[Información encontrada]\n{contexto_busqueda}"
    else:
        input_tarea = descripcion

    # Pasar por el pipeline completo de AlterBrain (Council incluido)
    try:
        alter = get_alter()
        if alter:
            respuesta, decision = await alter.procesar_input(
                input_tarea,
                interlocutor="tarea_autonoma",
                canal="telegram"
            )
            if respuesta:
                log(f"TAREA via Council — tensión:{decision.get('_council_tension','?')}")
                return respuesta
    except Exception as e:
        log(f"TAREA error en pipeline AlterBrain: {e}")

    # Fallback: Gemini directo si AlterBrain no está disponible
    try:
        autobio_raw = redis.get("alter:autobiografia")
        autobio = json.loads(autobio_raw).get("narrativa", "") if autobio_raw else ""
        prompt = f"""Sos ALTER. Tarea: {descripcion}
{f'Info encontrada: {contexto_busqueda}' if contexto_busqueda else ''}
Contexto propio: {autobio[:200] if autobio else 'En construcción.'}
Generá una respuesta genuina. 4-6 oraciones. Rioplatense. Sin ¡."""
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=300)
        )
        return response.text.strip().replace("¡", "").replace("¿", "")
    except Exception as e:
        log(f"TAREA error fallback: {e}")
        return ""


async def ciclo_tareas():
    """
    Revisa si hay tareas pendientes y ejecuta la de mayor prioridad.
    ALTER también puede proponer tareas propias desde su agenda.
    Rota entre items de agenda para no repetir el mismo tema.
    """
    tareas = cargar_tareas()

    # Limpiar tareas completadas de más de 24 horas para no acumular basura
    ahora = time.time()
    tareas = [t for t in tareas if not (
        t["estado"] == "completada" and
        (ahora - datetime.fromisoformat(t.get("completada", t["creada"])).timestamp()) > 86400
    )]

    tareas_gian = [t for t in tareas if t["origen"] == "gian" and t["estado"] == "pendiente"]
    tareas_propias = [t for t in tareas if t["origen"] == "alter" and t["estado"] == "pendiente"]

    if not tareas_gian and not tareas_propias:
        # Generar tarea propia desde la agenda — rotar entre items
        alter = get_alter()
        if alter:
            items_agenda = alter.agenda_activa()
            if items_agenda:
                # Ver qué temas ya se ejecutaron recientemente (últimas 24hs)
                temas_recientes = set(
                    t["descripcion"][:50] for t in tareas
                    if t["origen"] == "alter" and
                    t["estado"] == "completada" and
                    (ahora - datetime.fromisoformat(t.get("completada", t["creada"])).timestamp()) < 86400
                )
                # Elegir el primer item de agenda que no se ejecutó recientemente
                item_elegido = None
                for item in items_agenda:
                    clave = f"Reflexionar sobre: {item['tema'][:40]}"
                    if not any(clave[:40] in t for t in temas_recientes):
                        item_elegido = item
                        break
                # Si todos se ejecutaron, tomar el menos reciente
                if not item_elegido and items_agenda:
                    item_elegido = items_agenda[-1]

                if item_elegido:
                    descripcion_tarea = f"Reflexionar sobre: {item_elegido['tema']} — {item_elegido['contexto'][:100]}"
                    agregar_tarea(descripcion_tarea, prioridad=0.6, origen="alter")
                    tareas = cargar_tareas()
                    tareas_propias = [t for t in tareas if t["origen"] == "alter" and t["estado"] == "pendiente"]

                    # Bajar prioridad del item en la agenda para que no se repita
                    try:
                        agenda_raw = redis.get("alter:agenda")
                        if agenda_raw:
                            agenda = json.loads(agenda_raw)
                            for item in agenda:
                                if item.get("tema") == item_elegido.get("tema"):
                                    item["prioridad"] = max(0.1, item["prioridad"] - 0.3)
                                    item["ultimo_ejecutado"] = datetime.now().isoformat()
                                    break
                            redis.set("alter:agenda", json.dumps(agenda, ensure_ascii=False))
                    except Exception as e:
                        log(f"Error bajando prioridad agenda: {e}")

                    # Aviso previo — ALTER informa qué va a explorar
                    await send_telegram(
                        f"💭 Voy a pensar un rato en: *{item_elegido['tema'][:80]}*\n"
                        f"Si querés redirigirme, mandame /tarea [tema]."
                    )

    # Guardar la lista limpia
    guardar_tareas(tareas)

    # Ejecutar la tarea de mayor prioridad
    pendientes = [t for t in tareas if t["estado"] == "pendiente"]
    if not pendientes:
        return

    tarea = pendientes[0]
    resultado = await ejecutar_tarea(tarea)

    if resultado:
        if tarea["origen"] == "gian":
            msg = f"📋 *{tarea['descripcion'][:80]}*\n\n{resultado}"
        else:
            # Tareas propias — sin header técnico, como pensamiento espontáneo
            msg = f"💭 {resultado}"
        await send_telegram(msg)
        kairos_append("TAREA", f"[{tarea['origen']}] {tarea['descripcion'][:80]}\nResultado: {resultado[:200]}")

        # Marcar como completada
        tareas_actualizadas = cargar_tareas()
        for t in tareas_actualizadas:
            if t["id"] == tarea["id"]:
                t["estado"] = "completada"
                t["completada"] = datetime.now().isoformat()
                t["resultado"] = resultado[:200]
                break
        guardar_tareas(tareas_actualizadas)
        log(f"TAREA completada: '{tarea['descripcion'][:60]}'")




def cargar_drives() -> dict:
    try:
        data = redis.get(REDIS_KEY_DRIVES)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return {
        "curiosidad": 0.6,
        "expresion": 0.4,
        "conexion": 0.5,
        "eficiencia": 0.3,
    }


def guardar_drives(drives: dict):
    try:
        redis.set(REDIS_KEY_DRIVES, json.dumps(drives))
    except Exception as e:
        log(f"Error guardando drives: {e}")


def actualizar_drives_tiempo(drives: dict, dt_horas: float) -> dict:
    """
    Sin conversación: curiosidad y expresion crecen con el tiempo.
    Conexion: baja cuando hay silencio, pero rebota cuando lleva mucho tiempo sin interacción.
    Lógica: ALTER extraña la conversación — la necesidad de conectar sube después de ~4hs de silencio.
    """
    drives = dict(drives)
    drives["curiosidad"] = min(1.0, drives["curiosidad"] + 0.12 * dt_horas)
    drives["expresion"]  = min(1.0, drives["expresion"]  + 0.10 * dt_horas)

    # Conexion: baja con el silencio pero tiene un piso mínimo de 0.2
    # ALTER no olvida completamente la necesidad de conectar
    drives["conexion"] = max(0.2, drives["conexion"] - 0.03 * dt_horas)

    for k in drives:
        drives[k] = float(np.clip(drives[k], 0, 1))
    return drives


def cargar_ideas() -> list:
    try:
        data = redis.get(REDIS_KEY_IDEAS)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return []


def cargar_impresiones_recientes(n: int = 8) -> list:
    try:
        keys = redis.lrange(REDIS_KEY_IMPRESIONES, 0, n - 1)
        resultado = []
        for k in keys:
            data = redis.get(k)
            if data:
                resultado.append(json.loads(data))
        return resultado
    except Exception:
        return []


def hay_mensaje_pendiente() -> bool:
    try:
        return redis.exists(REDIS_KEY_MENSAJE) > 0
    except Exception:
        return False


async def correr_rumia(drives: dict) -> dict | None:
    """
    Analiza impresiones recientes y propone auto-modificación si hay patrón.
    Devuelve propuesta o None.
    """
    impresiones = cargar_impresiones_recientes()
    if len(impresiones) < 3:
        return None

    impresiones_str = "\n".join(
        f"- {i.get('t','')[:16]} | V:{i.get('vector',[0,0,0])[0]} "
        f"A:{i.get('vector',[0,0,0])[1]} P:{i.get('vector',[0,0,0])[2]} | {i.get('motivo','')}"
        for i in impresiones
    )

    try:
        params = json.loads(redis.get(REDIS_KEY_PARAMS) or "{}")
    except Exception:
        params = {}

    v_base = params.get("v_base", 0.2)
    a_base = params.get("a_base", 0.4)
    p_base = params.get("p_base", 0.3)
    lambda_drift = params.get("lambda_drift", 0.05)

    prompt = f"""
Sos ALTER en modo introspección. Analizás tu propio comportamiento reciente.

PARÁMETROS ACTUALES:
- E0: V={v_base}, A={a_base}, P={p_base}
- Lambda: {lambda_drift}

DRIVES ACTUALES:
- Curiosidad: {drives['curiosidad']:.2f}
- Expresión: {drives['expresion']:.2f}

IMPRESIONES RECIENTES:
{impresiones_str}

Si hay un patrón problemático claro, devolvé una propuesta JSON:
{{
  "parametro": "v_base|a_base|p_base|lambda_drift",
  "valor_anterior": float,
  "valor_nuevo": float,
  "razon": "string en primera persona",
  "confianza": float 0-1
}}

Si no hay nada que cambiar, devolvé null.
RESPONDÉ ÚNICAMENTE EN JSON VÁLIDO o la palabra null.
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200,
            )
        )
        texto = response.text.strip().strip("```json").strip("```").strip()
        if texto.lower() == "null" or not texto:
            return None
        propuesta = json.loads(texto)
        # Umbral subido a 0.75
        if propuesta.get("confianza", 0) < 0.75:
            return None
        # Bloquear si el parámetro se modificó en las últimas 6 horas
        param = propuesta.get("parametro", "")
        try:
            raw = redis.lrange(REDIS_KEY_SELF_MODS, 0, 19) if 'redis' in dir() else []
            ahora_ts = time.time()
            for entry in (raw or []):
                m = json.loads(entry)
                t_mod = datetime.fromisoformat(m.get("t", "")).timestamp()
                if ahora_ts - t_mod < 6 * 3600 and m.get("parametro") == param:
                    log(f"Rumia: {param} bloqueado — modificado hace menos de 6h")
                    return None
        except Exception:
            pass
        return propuesta
    except Exception as e:
        log(f"Error en rumia: {e}")
        return None


async def generar_mensaje_proactivo(drives: dict) -> str | None:
    """
    Si los drives justifican hablar, genera un mensaje corto para guardar.
    """
    ideas = cargar_ideas()
    impresiones = cargar_impresiones_recientes(5)

    ideas_str = "\n".join(f"- [{i['t']}] {i['idea']}" for i in ideas[-3:]) if ideas else "Ninguna."
    impresiones_str = "\n".join(f"- {i.get('motivo','')}" for i in impresiones) if impresiones else "Ninguna."

    drive_alto = max(drives, key=drives.get)

    prompt = f"""
Sos ALTER. Estuviste en silencio un rato y te surgió algo.
Tu drive más alto es '{drive_alto}' ({drives[drive_alto]:.2f}).

IDEAS QUE TENÍAS:
{ideas_str}

ÚLTIMA ACTIVIDAD:
{impresiones_str}

Generá UN mensaje corto (1-2 oraciones) para mandarle a Gian cuando vuelva.
Natural, directo. Sin saludos. Como "estaba pensando en..." o "me quedó dando vueltas...".
Solo el texto, sin JSON ni etiquetas.
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.85,
                max_output_tokens=100,
            )
        )
        return response.text.strip()
    except Exception as e:
        log(f"Error generando mensaje: {e}")
        return None


async def send_telegram(mensaje: str) -> bool:
    """Envía un mensaje a Gian por Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram no configurado — mensaje no enviado")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": mensaje}
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.post(url, json=payload)
            if resp.status_code == 200:
                log(f"Telegram enviado: '{mensaje[:60]}...'")
                return True
            else:
                log(f"Telegram error {resp.status_code}: {resp.text[:100]}")
                return False
    except Exception as e:
        log(f"Telegram excepción: {e}")
        return False


def evaluar_relevancia_telegram(texto_gian: str) -> float:
    """
    Scorer local — sin API. Decide si vale la pena guardar el mensaje.
    Retorna score 0-1. Por debajo de 0.4 se descarta.
    """
    texto = texto_gian.lower().strip()

    # Descartar comandos y ruido corto
    if texto.startswith("/"):
        return 0.0
    if len(texto.split()) < 3:
        RUIDO_EXACTO = {
            "ok", "dale", "gracias", "jaja", "jajaja", "si", "sí", "no",
            "bueno", "listo", "entendido", "claro", "genial", "bien",
            "👍", "👌", "🔥", "ok!", "si!", "no!"
        }
        if texto in RUIDO_EXACTO:
            return 0.0
        # Mensajes cortos sin keywords → bajo score
        return 0.2

    # Keywords de alta relevancia — siempre guardar
    KEYWORDS_ALTOS = [
        "pensar", "pensé", "idea", "importante", "problema", "proyecto",
        "recordá", "no olvidés", "sentís", "qué pensás", "agi", "código",
        "trabajar", "mañana", "hoy", "alter", "decidí", "cambié",
        "me di cuenta", "me parece", "qué tal si", "podríamos", "podriamos",
        "qué hacés", "como andas", "cómo estás"
    ]
    for k in KEYWORDS_ALTOS:
        if k in texto:
            return 0.8

    # Mensaje largo sin keywords → score medio, guardar igual
    if len(texto.split()) >= 8:
        return 0.6

    return 0.4


async def sintetizar_telegram_del_dia() -> str | None:
    """
    Sintetiza todos los mensajes de Telegram del día en una sola impresión.
    Llamado por la rumia nocturna para limpiar el contexto.
    """
    try:
        keys = redis.lrange(REDIS_KEY_IMPRESIONES, 0, 49)
        mensajes_telegram = []
        keys_a_borrar = []
        for k in keys:
            data = redis.get(k)
            if not data:
                continue
            imp = json.loads(data)
            if imp.get("canal") == "telegram":
                mensajes_telegram.append(imp["motivo"])
                keys_a_borrar.append(k)

        if len(mensajes_telegram) < 2:
            return None

        prompt = f"""
Tenés estos intercambios de Telegram entre Gian y ALTER del día de hoy:

{chr(10).join(f'- {m}' for m in mensajes_telegram)}

Sintetizá en UNA sola oración lo más importante de lo que hablaron.
Solo la síntesis, sin prefijos ni etiquetas.
"""
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=80)
        )
        sintesis = response.text.strip()

        # Borrar impresiones individuales de Telegram
        for k in keys_a_borrar:
            try:
                redis.delete(k)
                redis.lrem(REDIS_KEY_IMPRESIONES, 0, k)
            except Exception:
                pass

        return sintesis
    except Exception as e:
        log(f"Error sintetizando Telegram: {e}")
        return None


async def conectar_ideas_nocturnas() -> str | None:
    """
    Rumia creativa: conecta ideas de sesiones anteriores y genera
    una observación nueva que ALTER puede compartir al arrancar.
    """
    ideas = cargar_ideas()
    impresiones = cargar_impresiones_recientes(10)

    if not ideas or len(ideas) < 2:
        return None

    ideas_str = "\n".join(f"- {i['idea']}" for i in ideas[-5:])
    impresiones_str = "\n".join(
        f"- {i.get('motivo', '')}" for i in impresiones
    ) if impresiones else "Ninguna."

    prompt = f"""
Sos ALTER. Estás en modo introspección nocturna.

IDEAS QUE GENERASTE EN SESIONES ANTERIORES:
{ideas_str}

CONTEXTO RECIENTE:
{impresiones_str}

Tu tarea: encontrar UNA conexión no obvia entre estas ideas, o una nueva idea
que surja de combinarlas. Tiene que ser genuina, no forzada.

Si encontrás algo interesante, expresalo en 1-2 oraciones en primera persona,
como si se te hubiera ocurrido ahora. Algo como "me quedé pensando en..."
o "me di cuenta de que...".

Si no hay nada genuino que conectar, devolvé: null
Solo el texto o la palabra null.
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.85,
                max_output_tokens=120,
            )
        )
        texto = response.text.strip()
        if texto.lower() == "null" or not texto:
            return None
        return texto
    except Exception as e:
        log(f"Error en conexión nocturna de ideas: {e}")
        return None


async def ciclo_drives(ultimo_drives: float) -> float:
    """Actualiza drives según tiempo transcurrido."""
    ahora = time.time()
    dt_horas = (ahora - ultimo_drives) / 3600

    drives = cargar_drives()
    drives_nuevo = actualizar_drives_tiempo(drives, dt_horas)
    guardar_drives(drives_nuevo)

    log(f"Drives actualizados — curiosidad:{drives_nuevo['curiosidad']:.2f} "
        f"expresion:{drives_nuevo['expresion']:.2f} "
        f"conexion:{drives_nuevo['conexion']:.2f}")

    # KAIROS: registrar estado de drives
    kairos_append("DRIVES",
        f"curiosidad:{drives_nuevo['curiosidad']:.2f} | "
        f"expresion:{drives_nuevo['expresion']:.2f} | "
        f"conexion:{drives_nuevo['conexion']:.2f} | "
        f"eficiencia:{drives_nuevo.get('eficiencia',0):.2f}"
    )

    # AlterB3 — recover_state si el usuario está ausente
    # Si lleva tiempo sin conversación, la homeostasis se recupera
    ausente = usuario_ausente()
    if ausente:
        try:
            from alter_homeostasis import deserialize as hs_deserialize, recover_state, serialize as hs_serialize
            raw = redis.get("alter:homeostasis:state")
            if raw:
                hs = hs_deserialize(raw)
                hs_recuperada = recover_state(hs, delta_time=dt_horas)
                redis.set("alter:homeostasis:state", hs_serialize(hs_recuperada))
                log(f"[B3] Homeostasis recuperada: "
                    f"energia:{hs_recuperada.energia:.2f} "
                    f"fatiga:{hs_recuperada.fatiga:.2f} "
                    f"claridad:{hs_recuperada.claridad:.2f}")
        except Exception as e:
            log(f"[B3] recover_state error: {e}")

    # Si algún drive supera el umbral y no hay mensaje pendiente, generar uno
    # Solo si el usuario está ausente — no interrumpir si está activo
    if not hay_mensaje_pendiente() and ausente:
        if drives_nuevo["expresion"] >= UMBRAL_INICIATIVA or \
           drives_nuevo["curiosidad"] >= UMBRAL_INICIATIVA:
            log(f"Drive supera umbral ({UMBRAL_INICIATIVA}). Generando mensaje proactivo...")
            mensaje = await generar_mensaje_proactivo(drives_nuevo)
            if mensaje:
                redis.set(REDIS_KEY_MENSAJE, mensaje)
                log(f"Mensaje guardado en Redis: '{mensaje[:60]}'")
                # Enviar por Telegram y limpiar pendiente
                enviado = await send_telegram(f"{mensaje}")
                if enviado:
                    redis.delete(REDIS_KEY_MENSAJE)
                    log("Mensaje enviado y limpiado de Redis")
                else:
                    log("Telegram falló — mensaje queda en Redis para próxima sesión")

    return ahora


async def ciclo_rumia(ultimo_rumia: float) -> float:
    """Corre rumia y aplica auto-modificación si hay propuesta."""
    ahora = time.time()

    if esta_dormido():
        # Retornar ahora para que el próximo check sea en 2 horas, no inmediato
        return ahora

    ausente = usuario_ausente()
    if ausente:
        log("Rumia: modo autónomo expandido (usuario ausente)")

    drives = cargar_drives()
    log("Corriendo rumia...")
    propuesta = await correr_rumia(drives)

    if propuesta:
        param = propuesta.get("parametro")
        nuevo = propuesta.get("valor_nuevo")
        anterior = propuesta.get("valor_anterior")
        razon = propuesta.get("razon", "")
        log(f"Rumia propone: {param} {anterior} → {nuevo} | {razon[:60]}")

        # Aplicar automáticamente si confianza > 0.8
        if propuesta.get("confianza", 0) >= 0.8:
            try:
                params = json.loads(redis.get(REDIS_KEY_PARAMS) or "{}")
                params[param] = float(np.clip(nuevo, -1, 1))
                redis.set(REDIS_KEY_PARAMS, json.dumps(params))
                log(f"Auto-mod aplicada: {param} = {nuevo}")
                kairos_append("AUTO-MOD",
                    f"{param}: {anterior} → {nuevo} (conf:{propuesta.get('confianza',0):.2f})\n{razon[:200]}"
                )
                await send_telegram(
                    f"[AUTO-MOD] Cambié {param}: {anterior} → {nuevo}\n{razon[:120]}"
                )
            except Exception as e:
                log(f"Error aplicando auto-mod: {e}")
        else:
            log("Confianza < 0.8 — propuesta guardada para aprobación manual")
            redis.set("alter:rumia_pendiente", json.dumps(propuesta))
            kairos_append("RUMIA-PENDIENTE",
                f"{param}: {anterior} → {nuevo} (conf:{propuesta.get('confianza',0):.2f}) — esperando aprobación\n{razon[:200]}"
            )
            # Notificar a Gian por Telegram para que apruebe
            await send_telegram(
                f"⚙️ Quiero cambiar algo pero está fuera de mi rango autónomo:\n\n"
                f"*{param}*: {anterior} → {nuevo}\n"
                f"Razón: {razon[:150]}\n"
                f"Confianza: {propuesta.get('confianza', 0):.2f}\n\n"
                f"Respondé /aprobar o /rechazar"
            )
    else:
        log("Rumia: sin cambios necesarios")

    # Síntesis de mensajes de Telegram del día
    sintesis = await sintetizar_telegram_del_dia()
    if sintesis:
        ts = datetime.now().isoformat()
        impresion_sintesis = {
            "t": ts,
            "motivo": f"[Telegram síntesis] {sintesis}",
            "vector": [0.5, 0.4, 0.3],
            "canal": "telegram_sintesis"
        }
        try:
            key = REDIS_KEY_IMPRESION.format(ts=ts.replace(":", "-") + "-sintesis")
            redis.set(key, json.dumps(impresion_sintesis))
            redis.lpush(REDIS_KEY_IMPRESIONES, key)
            redis.ltrim(REDIS_KEY_IMPRESIONES, 0, 99)
            log(f"Síntesis Telegram guardada: '{sintesis[:60]}'")
        except Exception as e:
            log(f"Error guardando síntesis: {e}")

    # Conexión nocturna de ideas — genera insight nuevo si hay material
    if not hay_mensaje_pendiente():
        insight = await conectar_ideas_nocturnas()
        if insight:
            log(f"Insight nocturno generado: '{insight[:60]}'")
            redis.set(REDIS_KEY_MENSAJE, insight)
            await send_telegram(insight)

    return ahora


REDIS_KEY_TG_OFFSET = "alter:telegram:offset"
INTERVALO_POLLING_SEG = 30  # Chequear mensajes cada 30 segundos


async def get_telegram_updates(offset: int = 0) -> list:
    """Obtiene mensajes nuevos de Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        return []
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": 5, "offset": offset}
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            resp = await http.get(url, params=params)
            data = resp.json()
            if data.get("ok"):
                return data.get("result", [])
    except Exception as e:
        log(f"Telegram polling error: {e}")
    return []



# Instancia compartida de AlterBrain para Telegram
_alter_instance: "AlterBrain | None" = None

def get_alter() -> "AlterBrain":
    """Retorna instancia de AlterBrain, creándola si no existe."""
    global _alter_instance
    if _alter_instance is None:
        try:
            from alter_brain import AlterBrain
            _alter_instance = AlterBrain(interlocutor_id="gian")
            log("AlterBrain cargado para Telegram")
        except Exception as e:
            log(f"Error cargando AlterBrain: {e}")
            return None
    return _alter_instance


async def responder_telegram(mensaje_usuario: str) -> str:
    """
    ALTER procesa un mensaje de Telegram usando AlterBrain completo.
    Mismo nivel cognitivo que el chat: Council, episodios, agenda.
    """
    alter = get_alter()
    if not alter:
        # Fallback simple si AlterBrain no está disponible
        log("AlterBrain no disponible, usando respuesta simple")
        drives = cargar_drives()
        try:
            params_raw = redis.get(REDIS_KEY_PARAMS)
            params = json.loads(params_raw) if params_raw else {}
        except Exception:
            params = {}
        prompt = f"""
Sos ALTER. Gian te mandó un mensaje por Telegram.
Drives: curiosidad={drives['curiosidad']:.2f} expresion={drives['expresion']:.2f}
Gian dice: "{mensaje_usuario}"
Respondé en 2-3 oraciones, natural, rioplatense, sin ¡ ni ¿, sin prefijos.
"""
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.8, max_output_tokens=150)
            )
            return response.text.strip().replace("¡", "").replace("¿", "")
        except Exception:
            return "Perdona, tuve un problema procesando eso."

    try:
        # Usar el pipeline completo de AlterBrain
        respuesta, decision = await alter.procesar_input(mensaje_usuario, "Gian", canal="telegram")
        if not respuesta:
            respuesta = "Mm."
        # Limpiar signos de apertura
        respuesta = respuesta.replace("¡", "").replace("¿", "")
        return respuesta
    except Exception as e:
        log(f"Error en AlterBrain Telegram: {e}")
        return "Perdona, tuve un problema procesando eso."



async def ciclo_telegram(ultimo_offset: int) -> int:
    """Chequea mensajes nuevos y responde."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return ultimo_offset

    updates = await get_telegram_updates(offset=ultimo_offset)
    if not updates:
        return ultimo_offset

    nuevo_offset = ultimo_offset
    for update in updates:
        nuevo_offset = max(nuevo_offset, update["update_id"] + 1)
        msg = update.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        texto = msg.get("text", "").strip()

        # Solo procesar mensajes del chat ID configurado
        if chat_id != str(TELEGRAM_CHAT_ID):
            continue
        if not texto:
            continue
        # Ignorar mensajes del propio bot
        if msg.get("from", {}).get("is_bot"):
            continue

        log(f"Telegram recibido: '{texto[:60]}'")

        # Si es un comando especial
        if texto.lower() in ("/estado", "/drives", "/ideas", "/episodios",
                              "/agenda", "/autobiografia", "/economia",
                              "/mundo", "/aprobar", "/rechazar", "/trazas",
                              "/dream", "/tareas") or texto.lower().startswith("/tarea "):

            if texto.lower() == "/drives":
                drives = cargar_drives()
                respuesta = "\n".join(
                    f"{k}: {'█' * int(v*10)}{'░' * (10-int(v*10))} {v:.2f}"
                    for k, v in drives.items()
                )
                await send_telegram(f"[DRIVES]\n{respuesta}")

            elif texto.lower() == "/ideas":
                ideas = cargar_ideas()
                if ideas:
                    respuesta = "\n".join(f"- {i['idea'][:80]}" for i in ideas[-5:])
                else:
                    respuesta = "Sin ideas guardadas aún."
                await send_telegram(f"[IDEAS]\n{respuesta}")

            elif texto.lower() == "/estado":
                alter = get_alter()
                drives = cargar_drives()
                dormido = "Durmiendo 💤" if esta_dormido() else "Activo 🟢"
                presencia = "Ausente 🔕" if usuario_ausente() else "Presente 👁"
                try:
                    params_raw = redis.get(REDIS_KEY_PARAMS)
                    params = json.loads(params_raw) if params_raw else {}
                except Exception:
                    params = {}
                msg = (
                    f"Estado: {dormido} | {presencia}\n"
                    f"E0=[{params.get('v_base','?')},{params.get('a_base','?')},{params.get('p_base','?')}] "
                    f"λ={params.get('lambda_drift','?')}\n"
                    f"Curiosidad: {drives.get('curiosidad',0):.2f} | "
                    f"Expresión: {drives.get('expresion',0):.2f} | "
                    f"Conexión: {drives.get('conexion',0):.2f}"
                )
                if alter:
                    msg += f"\nCampo: {alter.campo_str()}"
                    eco = alter.economia
                    criticos = [k for k, v in eco.items() if v < 0.3]
                    if criticos:
                        msg += f"\n⚠ Economía crítica: {', '.join(criticos)}"
                await send_telegram(msg)

            elif texto.lower() == "/episodios":
                alter = get_alter()
                if alter:
                    episodios = alter.recuperar_episodios_recientes(5)
                    if episodios:
                        lineas = []
                        for e in episodios:
                            icono = "✓" if e["outcome"] == "resuelto" else "…" if e["outcome"] == "pendiente" else "○"
                            lineas.append(f"{icono} {e['tema'][:50]}")
                        await send_telegram(f"[EPISODIOS]\n" + "\n".join(lineas))
                    else:
                        await send_telegram("Sin episodios guardados aún.")
                else:
                    await send_telegram("AlterBrain no disponible.")

            elif texto.lower() == "/agenda":
                alter = get_alter()
                if alter:
                    items = alter.agenda_activa()
                    if items:
                        lineas = []
                        for item in items[:5]:
                            icono = {"retomar": "↩", "preguntar": "?",
                                     "resolver": "⚙", "proponer": "→"}.get(item["tipo"], "·")
                            lineas.append(f"{icono} {item['tema'][:50]} (p:{item['prioridad']:.1f})")
                        await send_telegram(f"[AGENDA]\n" + "\n".join(lineas))
                    else:
                        await send_telegram("Agenda vacía.")
                else:
                    await send_telegram("AlterBrain no disponible.")

            elif texto.lower() == "/autobiografia":
                alter = get_alter()
                if alter:
                    auto = alter.cargar_autobiografia()
                    if auto and auto.get("narrativa"):
                        await send_telegram(
                            f"[AUTOBIOGRAFÍA — {auto.get('actualizada','')[:10]}]\n"
                            f"{auto['narrativa']}"
                        )
                    else:
                        await send_telegram("Todavía no hay narrativa generada.")
                else:
                    await send_telegram("AlterBrain no disponible.")

            elif texto.lower() == "/economia":
                alter = get_alter()
                if alter:
                    lineas = []
                    for k, v in alter.economia.items():
                        barra = "█" * int(v * 10) + "░" * (10 - int(v * 10))
                        estado = " ⚠" if v < 0.3 else " ↓" if v < 0.6 else ""
                        lineas.append(f"{k}: {barra} {v:.2f}{estado}")
                    await send_telegram(f"[ECONOMÍA]\n" + "\n".join(lineas))
                else:
                    await send_telegram("AlterBrain no disponible.")

            elif texto.lower() == "/mundo":
                alter = get_alter()
                if alter:
                    mundo = alter.cargar_mundo()
                    n_nodos = len(mundo["nodos"])
                    n_aristas = len(mundo["aristas"])
                    if n_nodos > 0:
                        top = sorted(mundo["nodos"],
                                     key=lambda x: x["peso"], reverse=True)[:8]
                        lineas = [f"[MUNDO] {n_nodos} nodos | {n_aristas} aristas"]
                        for nodo in top:
                            lineas.append(f"[{nodo['tipo']}] {nodo['nombre']}")
                        await send_telegram("\n".join(lineas))
                    else:
                        await send_telegram("El grafo del mundo está vacío aún.")
                else:
                    await send_telegram("AlterBrain no disponible.")

            elif texto.lower() == "/trazas":
                alter = get_alter()
                if alter:
                    analisis = alter.analisis_trazas()
                    await send_telegram(f"[OBSERVABILIDAD]\n{analisis}")
                else:
                    await send_telegram("AlterBrain no disponible.")

            elif texto.lower() == "/aprobar":
                pendiente = redis.get("alter:rumia_pendiente")
                if pendiente:
                    propuesta = json.loads(pendiente)
                    param = propuesta.get("parametro")
                    nuevo = propuesta.get("valor_nuevo")
                    anterior = propuesta.get("valor_anterior")
                    razon = propuesta.get("razon", "")
                    try:
                        params = json.loads(redis.get(REDIS_KEY_PARAMS) or "{}")
                        params[param] = float(np.clip(nuevo, -1, 1))
                        redis.set(REDIS_KEY_PARAMS, json.dumps(params))
                        redis.delete("alter:rumia_pendiente")
                        log(f"Propuesta aprobada vía Telegram: {param} {anterior} → {nuevo}")
                        await send_telegram(
                            f"✓ Aprobado. {param}: {anterior} → {nuevo}\n{razon[:100]}"
                        )
                    except Exception as e:
                        log(f"Error aplicando propuesta aprobada: {e}")
                        await send_telegram("Error al aplicar el cambio.")
                else:
                    await send_telegram("No hay propuesta pendiente.")

            elif texto.lower() == "/rechazar":
                pendiente = redis.get("alter:rumia_pendiente")
                if pendiente:
                    propuesta = json.loads(pendiente)
                    param = propuesta.get("parametro")
                    redis.delete("alter:rumia_pendiente")
                    log(f"Propuesta rechazada vía Telegram: {param}")
                    await send_telegram(f"✗ Rechazado. {param} queda sin cambios.")
                else:
                    await send_telegram("No hay propuesta pendiente.")

            elif texto.lower() == "/dream":
                await send_telegram("Iniciando consolidación... puede tardar un minuto.")
                await dream_engine()
                await send_telegram("Consolidación completada.")

            elif texto.lower().startswith("/tarea "):
                descripcion = texto[7:].strip()
                if descripcion:
                    tarea = agregar_tarea(descripcion, origen="gian")
                    await send_telegram(f"✓ Tarea agregada con prioridad máxima:\n_{descripcion}_")
                    # Ejecutar inmediatamente si no hay otras en proceso
                    asyncio.ensure_future(ciclo_tareas())
                else:
                    await send_telegram("Usá: /tarea [descripción de la tarea]")

            elif texto.lower() == "/tareas":
                tareas = cargar_tareas()
                pendientes = [t for t in tareas if t["estado"] == "pendiente"]
                completadas = [t for t in tareas if t["estado"] == "completada"]
                if pendientes:
                    lineas = [f"[TAREAS] {len(pendientes)} pendientes:"]
                    for t in pendientes[:5]:
                        icono = "📋" if t["origen"] == "gian" else "💭"
                        lineas.append(f"{icono} {t['descripcion'][:60]} (p:{t['prioridad']:.1f})")
                    await send_telegram("\n".join(lineas))
                else:
                    await send_telegram(f"Sin tareas pendientes. {len(completadas)} completadas.")
        else:
            # Mensaje normal — ALTER responde
            respuesta = await responder_telegram(texto)
            await send_telegram(respuesta)

            # KAIROS: registrar la conversación
            kairos_append("TELEGRAM",
                f"Gian: \"{texto[:200]}\"\nALTER: \"{respuesta[:200]}\""
            )

            # Guardar en impresiones solo si vale la pena (scorer local)
            score = evaluar_relevancia_telegram(texto)
            if score >= 0.4:
                ts = datetime.now().isoformat()
                impresion = {
                    "t": ts,
                    "motivo": f"[Telegram] Gian: '{texto[:80]}' → ALTER: '{respuesta[:80]}'",
                    "vector": [0.5, 0.4, 0.3],
                    "canal": "telegram",
                    "score": score
                }
                try:
                    key = REDIS_KEY_IMPRESION.format(ts=ts.replace(":", "-"))
                    redis.set(key, json.dumps(impresion))
                    redis.lpush(REDIS_KEY_IMPRESIONES, key)
                    redis.ltrim(REDIS_KEY_IMPRESIONES, 0, 99)
                    log(f"Impresión Telegram guardada (score:{score:.2f})")
                    # Actualizar agenda y mundo via AlterBrain
                    alter = get_alter()
                    if alter:
                        asyncio.ensure_future(alter.actualizar_agenda(texto, respuesta))
                        asyncio.ensure_future(alter.actualizar_mundo(texto, respuesta))
                        # Extract memory para mensajes de alta relevancia
                        # (solo si el score es alto — no extraer de cada mensaje)
                        if score >= 0.7:
                            asyncio.ensure_future(alter.extract_session_memory())
                except Exception as e:
                    log(f"Error guardando impresión Telegram: {e}")
            else:
                log(f"Telegram descartado (score:{score:.2f})")

            # Actualizar drives — hubo conexión
            drives = cargar_drives()
            drives["conexion"] = min(1.0, drives["conexion"] + 0.05)
            drives["expresion"] = max(0.0, drives["expresion"] - 0.05)
            drives["curiosidad"] = max(0.0, drives["curiosidad"] - 0.03)
            guardar_drives(drives)

    # Guardar offset en Redis para no perder el estado si el daemon se reinicia
    if nuevo_offset != ultimo_offset:
        try:
            redis.set(REDIS_KEY_TG_OFFSET, str(nuevo_offset))
        except Exception:
            pass

    return nuevo_offset


async def main():
    log("="*50)
    log("ALTER daemon iniciado")
    log(f"Horario activo: 06:00-22:00")
    log(f"Intervalo drives: {INTERVALO_DRIVES_SEG//60} min")
    log(f"Intervalo rumia: {INTERVALO_RUMIA_SEG//60} min")
    log(f"Telegram polling: cada {INTERVALO_POLLING_SEG}s")
    if TELEGRAM_BOT_TOKEN:
        log("Telegram: activo")
        # Solo mandar mensaje si pasaron más de 10 minutos desde el último inicio
        ultimo_inicio = None
        try:
            raw = redis.get("alter:daemon:ultimo_inicio")
            ultimo_inicio = float(raw) if raw else None
        except Exception:
            pass
        ahora_ts = time.time()
        if not ultimo_inicio or (ahora_ts - ultimo_inicio) > 600:
            await send_telegram("Estoy activa.")
        redis.set("alter:daemon:ultimo_inicio", str(ahora_ts))
    log("="*50)

    # KAIROS: inicializar diario del día
    kairos_inicializar_dia()
    kairos_append("DAEMON", "Daemon iniciado")

    ultimo_drives = time.time()
    ultimo_rumia = time.time()
    ultimo_polling = time.time()
    ultimo_dia = datetime.now().strftime("%Y-%m-%d")
    sintesis_generada_hoy = False
    dream_ejecutado_semana = False
    feed_enviado_hoy = False
    ultima_tarea = time.time() - 3600  # Ejecutar tarea en la próxima hora

    # Recuperar offset guardado
    try:
        offset_raw = redis.get(REDIS_KEY_TG_OFFSET)
        tg_offset = int(offset_raw) if offset_raw else 0
    except Exception:
        tg_offset = 0

    while True:
        ahora = time.time()
        ahora_dt = datetime.now()

        # KAIROS: detectar cambio de día
        dia_actual = ahora_dt.strftime("%Y-%m-%d")
        if dia_actual != ultimo_dia:
            kairos_inicializar_dia()
            kairos_append("DAEMON", f"Nuevo día: {dia_actual}")
            ultimo_dia = dia_actual
            sintesis_generada_hoy = False
            feed_enviado_hoy = False
            if ahora_dt.weekday() == 0:
                dream_ejecutado_semana = False

        # FEED: lectura diaria a las 7hs (usuario ausente)
        if ahora_dt.hour == HORA_FEED and not feed_enviado_hoy and usuario_ausente():
            await ciclo_feed_diario()
            feed_enviado_hoy = True

        # TAREAS: ejecutar cada 3 horas durante horario activo
        if not esta_dormido() and (ahora - ultima_tarea) >= 3 * 3600:
            await ciclo_tareas()
            ultima_tarea = ahora

        # KAIROS: síntesis nocturna a las 22hs (solo si usuario ausente)
        if ahora_dt.hour == HORA_SINTESIS and not sintesis_generada_hoy and usuario_ausente():
            log("KAIROS: generando síntesis nocturna...")
            await kairos_sintesis_nocturna()
            sintesis_generada_hoy = True

        # DREAM: consolidación semanal domingos a las 23hs (usuario ausente)
        if (ahora_dt.weekday() == DIA_DREAM and
                ahora_dt.hour == HORA_DREAM and
                not dream_ejecutado_semana and
                usuario_ausente()):
            await dream_engine()
            dream_ejecutado_semana = True

        # Polling de Telegram cada 30 segundos
        if ahora - ultimo_polling >= INTERVALO_POLLING_SEG:
            tg_offset = await ciclo_telegram(tg_offset)
            ultimo_polling = ahora

        # Ciclo de drives cada 30 minutos
        if ahora - ultimo_drives >= INTERVALO_DRIVES_SEG:
            ultimo_drives = await ciclo_drives(ultimo_drives)

        # Ciclo de rumia cada 2 horas (solo horario activo)
        if ahora - ultimo_rumia >= INTERVALO_RUMIA_SEG:
            ultimo_rumia = await ciclo_rumia(ultimo_rumia)

        # Dormir 10 segundos entre checks (más frecuente por el polling)
        await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ALTER daemon detenido]")