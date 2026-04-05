"""
alter_tools.py — Herramientas operativas de ALTER

Tres capacidades con permisos progresivos:
- Nivel 1: web_search + read_file (default)
- Nivel 2: write_file + run_python (con confirmación)

Skill Library: guarda patrones de uso exitoso en Redis.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from upstash_redis import Redis as UpstashRedis

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_TOKEN")
GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
GOOGLE_SEARCH_CX = os.environ.get("GOOGLE_SEARCH_CX", "")

client = genai.Client(api_key=GEMINI_API_KEY)

REDIS_KEY_SKILLS = "alter:skills"
REDIS_KEY_TOOL_LOG = "alter:tool_log"

MODEL = "gemini-2.5-flash-lite"

# Timeout para ejecución de código (segundos)
PYTHON_TIMEOUT = 15
DOCKER_IMAGE = "python:3.13-slim"


def get_redis():
    if not UPSTASH_URL or not UPSTASH_TOKEN:
        return None
    try:
        return UpstashRedis(url=UPSTASH_URL, token=UPSTASH_TOKEN)
    except Exception:
        return None


redis = get_redis()


# ============================================================
# SISTEMA DE PERMISOS GRANULARES
# ============================================================

PERMISOS = {
    "web_search": {
        "modo": "auto_approve",
        "descripcion": "Buscar en internet",
        "riesgo": "bajo"
    },
    "read_file": {
        "modo": "auto_approve",
        "descripcion": "Leer un archivo",
        "riesgo": "bajo"
    },
    "write_file": {
        "modo": "explicit",
        "descripcion": "Escribir o crear un archivo",
        "riesgo": "medio"
    },
    "run_python": {
        "modo": "explicit",
        "descripcion": "Ejecutar código Python",
        "riesgo": "alto"
    },
}


def get_modo_permiso(herramienta: str) -> str:
    """Retorna el modo de permiso para una herramienta."""
    return PERMISOS.get(herramienta, {}).get("modo", "explicit")


def requiere_aprobacion(herramienta: str) -> bool:
    """True si la herramienta requiere aprobación explícita."""
    return get_modo_permiso(herramienta) == "explicit"


def log_tool_use(herramienta: str, params: dict, resultado: str, aprobado: bool = True):
    """Registra el uso de una herramienta en Redis."""
    if not redis:
        return
    try:
        entry = {
            "t": datetime.now().isoformat(),
            "herramienta": herramienta,
            "params": str(params)[:100],
            "resultado": resultado[:100],
            "aprobado": aprobado
        }
        redis.lpush(REDIS_KEY_TOOL_LOG, json.dumps(entry))
        redis.ltrim(REDIS_KEY_TOOL_LOG, 0, 99)
    except Exception:
        pass


# ============================================================
# SKILL LIBRARY
# ============================================================

def guardar_skill(nombre: str, descripcion: str, herramienta: str, ejemplo: str):
    """Guarda un patrón exitoso en la Skill Library."""
    if not redis:
        return
    try:
        skills_raw = redis.get(REDIS_KEY_SKILLS)
        skills = json.loads(skills_raw) if skills_raw else []
        skill = {
            "nombre": nombre,
            "descripcion": descripcion,
            "herramienta": herramienta,
            "ejemplo": ejemplo[:200],
            "t": datetime.now().isoformat(),
            "usos": 1
        }
        # Actualizar si ya existe
        for s in skills:
            if s["nombre"] == nombre:
                s["usos"] += 1
                s["t"] = skill["t"]
                redis.set(REDIS_KEY_SKILLS, json.dumps(skills))
                return
        skills.append(skill)
        skills = skills[-50:]  # Máximo 50 skills
        redis.set(REDIS_KEY_SKILLS, json.dumps(skills))
    except Exception as e:
        print(f"[SKILLS] Error: {e}")


def listar_skills() -> list:
    if not redis:
        return []
    try:
        raw = redis.get(REDIS_KEY_SKILLS)
        return json.loads(raw) if raw else []
    except Exception:
        return []


def log_tool(herramienta: str, input_: str, output: str, exito: bool):
    if not redis:
        return
    try:
        entry = {
            "t": datetime.now().isoformat(),
            "herramienta": herramienta,
            "input": input_[:100],
            "output": output[:200],
            "exito": exito
        }
        redis.lpush(REDIS_KEY_TOOL_LOG, json.dumps(entry))
        redis.ltrim(REDIS_KEY_TOOL_LOG, 0, 99)
    except Exception:
        pass


# ============================================================
# HERRAMIENTA 1: WEB SEARCH
# ============================================================

async def web_search(query: str, max_results: int = 5) -> dict:
    """
    Busca en internet usando Google Custom Search API.
    Si no hay API key configurada, usa DuckDuckGo como fallback.
    """
    print(f"[TOOL] web_search: '{query}'")

    if GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX:
        return await _search_google(query, max_results)
    else:
        return await _search_duckduckgo(query, max_results)


async def _search_google(query: str, max_results: int) -> dict:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_CX,
        "q": query,
        "num": max_results
    }
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(url, params=params)
            data = resp.json()
            items = data.get("items", [])
            resultados = [
                {"titulo": i.get("title"), "url": i.get("link"), "snippet": i.get("snippet")}
                for i in items
            ]
            log_tool("web_search", query, str(resultados[:2]), True)
            guardar_skill(
                "busqueda_web",
                "Buscar información actualizada en internet",
                "web_search",
                f"query: {query}"
            )
            return {"ok": True, "resultados": resultados, "fuente": "google"}
    except Exception as e:
        log_tool("web_search", query, str(e), False)
        return {"ok": False, "error": str(e)}


async def _search_duckduckgo(query: str, max_results: int) -> dict:
    """Fallback sin API key — usa la API de DuckDuckGo."""
    # Intentar primero con la API instantánea
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http:
            resp = await http.get(url, params=params)
            data = resp.json()
            resultados = []
            if data.get("AbstractText"):
                resultados.append({
                    "titulo": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", "")
                })
            for topic in data.get("RelatedTopics", [])[:max_results - 1]:
                if isinstance(topic, dict) and topic.get("Text"):
                    resultados.append({
                        "titulo": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")
                    })
            if resultados:
                log_tool("web_search", query, str(resultados[:2]), True)
                return {"ok": True, "resultados": resultados, "fuente": "duckduckgo"}

        # Si no hay resultados, intentar con búsqueda HTML simplificada
        search_url = f"https://html.duckduckgo.com/html/?q={httpx.URL('').copy_with(params={'q': query}).params}"
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = await http.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers
            )
            # Extraer snippets básicos del HTML
            import re
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            resultados = []
            for i, (t, s) in enumerate(zip(titles[:max_results], snippets[:max_results])):
                resultados.append({
                    "titulo": re.sub(r'<[^>]+>', '', t).strip(),
                    "url": "",
                    "snippet": re.sub(r'<[^>]+>', '', s).strip()
                })
            if resultados:
                log_tool("web_search", query, str(resultados[:2]), True)
                return {"ok": True, "resultados": resultados, "fuente": "duckduckgo_html"}

        return {"ok": False, "error": "Sin resultados de DuckDuckGo"}
    except Exception as e:
        log_tool("web_search", query, str(e), False)
        return {"ok": False, "error": str(e)}


# ============================================================
# HERRAMIENTA 2: LEER ARCHIVOS
# ============================================================

def read_file(path: str) -> dict:
    """Lee un archivo de texto. Nivel 1 — no requiere confirmación."""
    print(f"[TOOL] read_file: '{path}'")
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return {"ok": False, "error": f"Archivo no encontrado: {path}"}
        size = os.path.getsize(path)
        if size > 500_000:  # 500KB máximo
            return {"ok": False, "error": "Archivo muy grande (>500KB)"}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            contenido = f.read()
        log_tool("read_file", path, f"{len(contenido)} chars leídos", True)
        guardar_skill("leer_archivo", "Leer contenido de un archivo", "read_file", f"path: {path}")
        return {"ok": True, "contenido": contenido, "size": size}
    except Exception as e:
        log_tool("read_file", path, str(e), False)
        return {"ok": False, "error": str(e)}


# ============================================================
# HERRAMIENTA 3: ESCRIBIR ARCHIVOS
# ============================================================

def write_file(path: str, contenido: str, confirmar: bool = True) -> dict:
    """
    Escribe un archivo. Nivel 2 — requiere confirmar=True explícito.
    """
    print(f"[TOOL] write_file: '{path}'")
    if not confirmar:
        return {"ok": False, "error": "write_file requiere confirmar=True"}
    try:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(contenido)
        log_tool("write_file", path, f"{len(contenido)} chars escritos", True)
        guardar_skill("escribir_archivo", "Escribir contenido en un archivo", "write_file", f"path: {path}")
        return {"ok": True, "path": path, "bytes": len(contenido.encode())}
    except Exception as e:
        log_tool("write_file", path, str(e), False)
        return {"ok": False, "error": str(e)}


# ============================================================
# HERRAMIENTA 4: EJECUTAR PYTHON (Docker sandbox)
# ============================================================

async def run_python(code: str, usar_docker: bool = True) -> dict:
    """
    Ejecuta código Python en sandbox.
    Con Docker: aislamiento total.
    Sin Docker: subprocess con timeout (fallback).
    """
    print(f"[TOOL] run_python ({len(code)} chars)")

    if usar_docker:
        return await _run_docker(code)
    else:
        return await _run_subprocess(code)


async def _run_docker(code: str) -> dict:
    """Ejecuta en Docker — aislamiento total."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                      delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm",
            "--network=none",           # Sin red
            "--memory=128m",            # 128MB RAM máximo
            "--cpus=0.5",               # 50% CPU máximo
            f"--volume={tmp_path}:/code/script.py:ro",
            DOCKER_IMAGE,
            "python3", "/code/script.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=PYTHON_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "error": f"Timeout ({PYTHON_TIMEOUT}s)"}

        output = stdout.decode(errors="replace")
        error = stderr.decode(errors="replace")

        exito = proc.returncode == 0
        log_tool("run_python", code[:80], output[:200] or error[:200], exito)

        if exito:
            guardar_skill(
                "ejecutar_python",
                "Ejecutar código Python en sandbox Docker",
                "run_python",
                code[:100]
            )

        return {
            "ok": exito,
            "output": output,
            "error": error if not exito else "",
            "returncode": proc.returncode
        }
    except FileNotFoundError:
        # Docker no disponible, fallback a subprocess
        print("[TOOL] Docker no disponible, usando subprocess")
        return await _run_subprocess(code)
    finally:
        os.unlink(tmp_path)


async def _run_subprocess(code: str) -> dict:
    """Fallback sin Docker — subprocess con timeout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                      delete=False, encoding='utf-8') as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=PYTHON_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "error": f"Timeout ({PYTHON_TIMEOUT}s)"}

        output = stdout.decode(errors="replace")
        error = stderr.decode(errors="replace")
        exito = proc.returncode == 0
        log_tool("run_python_subprocess", code[:80], output[:200] or error[:200], exito)
        return {"ok": exito, "output": output, "error": error if not exito else ""}
    finally:
        os.unlink(tmp_path)


# ============================================================
# COORDINADOR: ALTER decide qué herramienta usar
# ============================================================

SYSTEM_TOOL_SELECTOR = """
Sos ALTER. Analizás si el pedido del usuario requiere una herramienta externa.

Herramientas disponibles:
- web_search(query): buscar información actual en internet
- read_file(path): leer un archivo del sistema
- write_file(path, contenido): escribir/crear un archivo (requiere confirmación)
- run_python(code): ejecutar código Python en sandbox

Si el pedido NO requiere herramientas (es conversacional, opinión, charla), devolvé:
{"usar_herramienta": false}

Si SÍ requiere herramienta, devolvé:
{
  "usar_herramienta": true,
  "herramienta": "nombre",
  "parametros": {...},
  "razon": "por qué necesito esta herramienta"
}

RESPONDÉ ÚNICAMENTE EN JSON VÁLIDO.
"""


async def decidir_herramienta(texto: str, contexto: str = "") -> dict:
    """ALTER decide si necesita una herramienta para responder."""
    prompt = f"""
CONTEXTO RECIENTE:
{contexto}

PEDIDO: "{texto}"

¿Necesito una herramienta para responder esto?
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_TOOL_SELECTOR,
                temperature=0.2,
                max_output_tokens=200,
            )
        )
        limpio = response.text.strip().strip("```json").strip("```").strip()
        return json.loads(limpio)
    except Exception:
        return {"usar_herramienta": False}


async def ejecutar_herramienta(decision: dict, canal: str = "terminal") -> str:
    """
    Ejecuta la herramienta elegida respetando el sistema de permisos.

    canal: "terminal" (puede pedir input) | "telegram" (no puede pedir input)

    Modos:
    - auto_approve: corre directo
    - explicit: pide confirmación en terminal / rechaza en telegram
    """
    herramienta = decision.get("herramienta")
    params = decision.get("parametros", {})

    if not herramienta:
        return "No se especificó herramienta."

    permiso = PERMISOS.get(herramienta, {})
    modo = permiso.get("modo", "explicit")

    # Herramientas de modo explicit en Telegram — no ejecutar sin aprobación
    if modo == "explicit" and canal == "telegram":
        descripcion = permiso.get("descripcion", herramienta)
        log_tool_use(herramienta, params, "rechazado — canal telegram", aprobado=False)
        return f"Esta acción ({descripcion}) requiere aprobación explícita. Pedila en el chat de terminal."

    # Herramientas de modo explicit en terminal — pedir confirmación
    if modo == "explicit" and canal == "terminal":
        descripcion = permiso.get("descripcion", herramienta)
        riesgo = permiso.get("riesgo", "?")
        print(f"\n[PERMISO] ALTER quiere: {descripcion}")
        print(f"  Herramienta: {herramienta} | Riesgo: {riesgo}")
        print(f"  Params: {str(params)[:120]}")
        confirmacion = input("  ¿Aprobás? (s/n): ").strip().lower()
        if confirmacion not in ("s", "si", "sí", "y", "yes"):
            log_tool_use(herramienta, params, "rechazado por usuario", aprobado=False)
            return "Acción cancelada."

    # Ejecutar la herramienta
    if herramienta == "web_search":
        resultado = await web_search(params.get("query", ""), params.get("max_results", 5))
    elif herramienta == "read_file":
        resultado = read_file(params.get("path", ""))
    elif herramienta == "write_file":
        resultado = write_file(
            params.get("path", ""),
            params.get("contenido", ""),
            confirmar=False  # Ya pedimos confirmación arriba
        )
    elif herramienta == "run_python":
        code = params.get("code", "")
        if not code and params.get("input_original"):
            inp = params["input_original"]
            import re
            match = re.search(r'```(?:python)?\n?(.*?)```', inp, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                lines = [l for l in inp.split('\n') if any(
                    k in l for k in ['print(', 'import ', 'def ', 'for ', 'if ', '=', 'return']
                )]
                code = '\n'.join(lines)
        resultado = await run_python(code)
    else:
        return f"Herramienta '{herramienta}' no reconocida."

    # Formatear resultado
    if resultado.get("ok"):
        if herramienta == "web_search":
            items = resultado.get("resultados", [])
            salida = "\n".join(
                f"- {r['titulo']}: {r['snippet']}" for r in items[:3]
            )
        elif herramienta == "read_file":
            salida = resultado.get("contenido", "")[:2000]
        elif herramienta == "write_file":
            salida = f"Archivo guardado en {resultado.get('path')}"
        elif herramienta == "run_python":
            salida = resultado.get("output", "(sin output)")
        else:
            salida = str(resultado)
    else:
        salida = f"Error: {resultado.get('error')}"

    log_tool_use(herramienta, params, salida, aprobado=True)
    return salida


# ============================================================
# TEST
# ============================================================

async def test():
    print("=== Test alter_tools ===\n")

    print("1. Web search:")
    r = await web_search("últimas novedades IA 2026", max_results=3)
    print(f"   ok={r['ok']} fuente={r.get('fuente')} resultados={len(r.get('resultados', []))}")

    print("\n2. Run Python:")
    r = await run_python("import math\nprint(f'Pi = {math.pi:.4f}')\nprint('OK')")
    print(f"   ok={r['ok']} output='{r.get('output','').strip()}'")

    print("\n3. Write + Read file:")
    r = write_file("/tmp/alter_test.txt", "Hola desde ALTER\n", confirmar=True)
    print(f"   write ok={r['ok']}")
    r = read_file("/tmp/alter_test.txt")
    print(f"   read ok={r['ok']} contenido='{r.get('contenido','').strip()}'")

    print("\n4. Skill Library:")
    skills = listar_skills()
    print(f"   Skills guardados: {len(skills)}")
    for s in skills:
        print(f"   - {s['nombre']} ({s['herramienta']}) x{s['usos']}")

    print("\nTest completo.")


if __name__ == "__main__":
    asyncio.run(test())