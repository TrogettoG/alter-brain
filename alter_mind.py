"""
alter_mind.py — Capa de Mente de ALTER

Define los procesos cognitivos: campo mental, Inner Council,
memoria episódica, agenda cognitiva, drives.

Esta capa NO sabe de la voz ni del estilo de ALTER.
Solo sabe cómo piensa.

Separación:
  Persona  ← alter_persona.py
  Mente    ← esta capa
  Brain    ← alter_brain.py (orquestador)
  Daemon   ← alter_daemon.py (infraestructura)
"""

import asyncio
import json
import time
from datetime import datetime

import numpy as np
from google import genai
from google.genai import types

from alter_persona import MODELO, SYSTEM_PROMPT_COUNCIL


# ============================================================
# CAMPO MENTAL EXPANDIDO
# ============================================================

MODOS_COGNITIVOS = ["exploracion", "sintesis", "defensa", "ejecucion", "juego"]
FOCOS_ATENCIONALES = ["usuario", "idea", "tarea", "conflicto", "memoria", "mundo"]
PRESIONES = ["ninguna", "responder", "corregir", "preguntar", "cerrar"]
FRICCIONES = ["ninguna", "duda", "contradiccion", "saturacion", "fatiga"]

CAMPO_DEFAULT = {
    "modo":    "exploracion",
    "foco":    "usuario",
    "presion": "ninguna",
    "friccion": "ninguna",
}


def actualizar_campo(campo: dict, v: float, a: float, p: float,
                     data: dict, drives: dict,
                     nivel_cansancio: float,
                     n_historial: int) -> dict:
    """
    Actualiza el campo mental a partir del vector emocional y el contexto.
    Pure function — no tiene side effects.
    """
    campo = dict(campo)
    accion = data.get("accion", "responder")
    motivo = data.get("motivo", "").lower()

    # Modo cognitivo
    if accion == "interrumpir" or p > 0.6:
        campo["modo"] = "defensa"
    elif a > 0.6 and v > 0.3:
        campo["modo"] = "exploracion"
    elif v < -0.3 and a < 0.3:
        campo["modo"] = "sintesis"
    elif drives.get("eficiencia", 0) > 0.6:
        campo["modo"] = "ejecucion"
    else:
        campo["modo"] = "juego"

    # Foco atencional
    if any(k in motivo for k in ["tarea", "código", "archivo", "buscar", "ejecutar"]):
        campo["foco"] = "tarea"
    elif any(k in motivo for k in ["conflicto", "error", "contradicción", "problema"]):
        campo["foco"] = "conflicto"
    elif any(k in motivo for k in ["recuerdo", "antes", "episodio", "memoria"]):
        campo["foco"] = "memoria"
    elif any(k in motivo for k in ["idea", "concepto", "teoría", "pensar"]):
        campo["foco"] = "idea"
    else:
        campo["foco"] = "usuario"

    # Presión interna
    if v < -0.5 or accion == "interrumpir":
        campo["presion"] = "corregir"
    elif "?" in motivo or any(k in motivo for k in ["pregunta", "no entiendo", "cómo"]):
        campo["presion"] = "preguntar"
    elif drives.get("eficiencia", 0) > 0.7:
        campo["presion"] = "cerrar"
    elif accion in ("responder", "registrar"):
        campo["presion"] = "responder"
    else:
        campo["presion"] = "ninguna"

    # Fricción interna
    confianza = data.get("confianza", 1.0)
    if confianza < 0.5:
        campo["friccion"] = "duda"
    elif v < -0.7 and p > 0.5:
        campo["friccion"] = "contradiccion"
    elif nivel_cansancio > 0.6:
        campo["friccion"] = "fatiga"
    elif n_historial > 20:
        campo["friccion"] = "saturacion"
    else:
        campo["friccion"] = "ninguna"

    return campo


def campo_str(campo: dict) -> str:
    """Descripción compacta del campo mental para el prompt."""
    partes = [f"modo:{campo['modo']}"]
    if campo["foco"] != "usuario":
        partes.append(f"foco:{campo['foco']}")
    if campo["presion"] != "ninguna":
        partes.append(f"presión:{campo['presion']}")
    if campo["friccion"] != "ninguna":
        partes.append(f"fricción:{campo['friccion']}")
    return " | ".join(partes)


# ============================================================
# TONO EMOCIONAL
# ============================================================

def tono_emocional_str(v: float, a: float, p: float) -> str:
    """Convierte el vector en descripción cualitativa."""
    if v > 0.7:    vstr = "muy positivo, entusiasta"
    elif v > 0.3:  vstr = "positivo, receptivo"
    elif v > -0.3: vstr = "neutral, estable"
    elif v > -0.7: vstr = "algo bajo, introspectivo"
    else:          vstr = "bajo, reservado"

    if a > 0.7:    astr = "alta energía"
    elif a > 0.4:  astr = "energía normal"
    else:          astr = "energía baja, pausado"

    if p > 0.6:    pstr = "firme en sus posiciones"
    elif p > 0.2:  pstr = "equilibrado"
    elif p > -0.2: pstr = "abierto, flexible"
    else:          pstr = "cediendo, escuchando más que imponiendo"

    return f"{vstr} | {astr} | {pstr}"


# ============================================================
# INNER COUNCIL
# ============================================================

async def inner_council(client: genai.Client,
                         texto_input: str,
                         interlocutor: str,
                         v: float, a: float, p: float,
                         campo: dict,
                         episodios_str: str,
                         agenda_str: str,
                         historial_str: str) -> dict | None:
    """
    Debate interno de las tres voces antes de responder.
    Retorna el resultado o None si falla.
    """
    contexto = f"""
ESTADO: V:{v:.2f} A:{a:.2f} P:{p:.2f} | {tono_emocional_str(v, a, p)}
CAMPO: {campo_str(campo)}

EPISODIOS RELEVANTES:
{episodios_str}

AGENDA:
{agenda_str}

HISTORIAL RECIENTE:
{historial_str}

{interlocutor} dice: "{texto_input}"
"""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODELO,
            contents=contexto,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_COUNCIL,
                temperature=0.7,
                max_output_tokens=200,
            )
        )
        limpio = response.text.strip().strip("```json").strip("```").strip()
        return json.loads(limpio)
    except Exception:
        return None


# ============================================================
# ECONOMÍA MENTAL
# ============================================================

ECONOMIA_DEFAULT = {
    "atencion":   1.0,  # capacidad de procesar con profundidad
    "energia":    1.0,  # disponibilidad para responder
    "tolerancia": 1.0,  # aguante ante conflicto/contradicción
    "expresion":  1.0,  # ganas de hablar vs callar
}

ECONOMIA_RECUPERACION = {
    "atencion":   0.05,  # por turno sin input intenso
    "energia":    0.08,
    "tolerancia": 0.06,
    "expresion":  0.04,
}

def consumir_economia(economia: dict, accion: str,
                       council_tension: str,
                       confianza: float,
                       campo: dict) -> dict:
    """
    Consume recursos internos según la complejidad del turno.
    Pure function — retorna nueva economía.
    """
    economia = dict(economia)

    # Costo base por acción
    costo_accion = {
        "ignorar":          {"atencion": 0.01, "energia": 0.0,  "tolerancia": 0.0,  "expresion": 0.0},
        "registrar":        {"atencion": 0.03, "energia": 0.02, "tolerancia": 0.0,  "expresion": 0.0},
        "responder":        {"atencion": 0.05, "energia": 0.06, "tolerancia": 0.02, "expresion": 0.08},
        "interrumpir":      {"atencion": 0.08, "energia": 0.05, "tolerancia": 0.10, "expresion": 0.05},
        "usar_herramienta": {"atencion": 0.10, "energia": 0.08, "tolerancia": 0.01, "expresion": 0.03},
    }.get(accion, {"atencion": 0.03, "energia": 0.03, "tolerancia": 0.01, "expresion": 0.02})

    # Multiplicador por tensión del Council
    mult_tension = {"ninguna": 1.0, "baja": 1.1, "media": 1.4, "alta": 1.8}.get(
        council_tension, 1.0
    )

    # Multiplicador por baja confianza — más esfuerzo para generar
    mult_confianza = 1.0 + (1.0 - confianza) * 0.5

    # Multiplicador por fricción interna
    mult_friccion = {
        "ninguna": 1.0, "duda": 1.3, "contradiccion": 1.5,
        "saturacion": 1.4, "fatiga": 1.6
    }.get(campo.get("friccion", "ninguna"), 1.0)

    mult_total = mult_tension * mult_confianza * mult_friccion

    for recurso, costo in costo_accion.items():
        consumo = costo * mult_total
        economia[recurso] = float(np.clip(economia[recurso] - consumo, 0.0, 1.0))

    return economia


def recuperar_economia(economia: dict, turnos_descanso: float = 1.0) -> dict:
    """Recupera recursos con el tiempo."""
    economia = dict(economia)
    for recurso, tasa in ECONOMIA_RECUPERACION.items():
        economia[recurso] = float(np.clip(
            economia[recurso] + tasa * turnos_descanso, 0.0, 1.0
        ))
    return economia


def economia_str(economia: dict) -> str:
    """Descripción compacta para el prompt."""
    criticos = [k for k, v in economia.items() if v < 0.3]
    bajos = [k for k, v in economia.items() if 0.3 <= v < 0.6]

    if not criticos and not bajos:
        return "plena capacidad"
    partes = []
    if criticos:
        partes.append("crítico: " + ", ".join(criticos))
    if bajos:
        partes.append("bajo: " + ", ".join(bajos))
    return " | ".join(partes)


def economia_afecta_accion(economia: dict, accion_propuesta: str) -> str:
    """
    Si los recursos están muy bajos, puede forzar una acción más conservadora.
    """
    if economia.get("energia", 1.0) < 0.15 and accion_propuesta == "responder":
        return "registrar"  # Demasiado agotado para responder
    if economia.get("tolerancia", 1.0) < 0.1 and accion_propuesta == "interrumpir":
        return "registrar"  # Sin tolerancia para el conflicto
    return accion_propuesta


# ============================================================
# DRIVES
# ============================================================

DRIVES_DEFAULT = {
    "curiosidad": 0.6,
    "expresion":  0.4,
    "conexion":   0.5,
    "eficiencia": 0.3,
}

VENTANAS_AUTONOMIA = {
    "v_base":       (0.3, 0.6),
    "a_base":       (0.3, 0.6),
    "p_base":       (0.1, 0.5),
    "lambda_drift": (0.05, 0.12),
}


def actualizar_drives_sesion(drives: dict, turnos_sin_input: float = 0) -> dict:
    """Actualiza drives durante la sesión."""
    drives = dict(drives)
    if turnos_sin_input > 0:
        drives["curiosidad"] = min(1.0, drives["curiosidad"] + 0.02)
        drives["expresion"]  = min(1.0, drives["expresion"]  + 0.01)
        drives["conexion"]   = max(0.0, drives["conexion"]   - 0.005)
    else:
        drives["conexion"]   = min(1.0, drives["conexion"]   + 0.01)
        drives["curiosidad"] = max(0.1, drives["curiosidad"] - 0.02)
        drives["expresion"]  = max(0.1, drives["expresion"]  - 0.01)
    for k in drives:
        drives[k] = float(np.clip(drives[k], 0, 1))
    return drives


def clasificar_modificacion(param: str, valor_nuevo: float) -> str:
    """Devuelve 'autonomo' si el valor está dentro del rango, 'consultar' si no."""
    ventana = VENTANAS_AUTONOMIA.get(param)
    if not ventana:
        return "consultar"
    lo, hi = ventana
    return "autonomo" if lo <= valor_nuevo <= hi else "consultar"


# ============================================================
# HORARIO
# ============================================================

def esta_dormido() -> bool:
    """ALTER duerme entre las 22:00 y las 06:00."""
    hora = datetime.now().hour
    return hora >= 22 or hora < 6


def estado_horario() -> str:
    if esta_dormido():
        hora = datetime.now().hour
        if hora >= 22:
            mins = (6 + 24 - hora) * 60 - datetime.now().minute
        else:
            mins = (6 - hora) * 60 - datetime.now().minute
        return f"durmiendo (despierta en ~{mins//60}h {mins%60}min)"
    return "activo"