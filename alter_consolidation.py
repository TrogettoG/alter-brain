"""
alter_consolidation.py — Offline Consolidation mejorada de AlterB3 (Fase 4)

Extiende dream_engine() del daemon con actualizaciones reales de parámetros.

Lo que hacía DREAM antes:
    - Resumir episodios, ideas, impresiones
    - Limpiar nodos obsoletos
    - Generar resumen semanal

Lo que hace ahora además:
    1. Actualiza success_rate de patrones procedurales desde historial semanal
    2. Ajusta pesos de nodos semánticos por frecuencia de uso en episodios
    3. Sube/baja prioridades de agenda según qué se resolvió
    4. Genera nuevos patrones procedurales desde episodios recurrentes
    5. Actualiza model_confidence del Predictive Model
    6. Revisa si la narrativa de identidad necesita actualización

Integración:
    alter_daemon.py llama a OfflineConsolidation.run() en lugar de (o además de)
    dream_engine(). Gradualmente dream_engine() se delega aquí.

No usa Gemini directamente — opera sobre datos estructurados.
Gemini solo se llama para la narrativa final (resumen semanal).
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


# ============================================================
# RESULTADO DE CONSOLIDACIÓN
# ============================================================

@dataclass
class ConsolidationResult:
    timestamp:              str
    patrones_actualizados:  int = 0
    patrones_nuevos:        int = 0
    nodos_ajustados:        int = 0
    nodos_eliminados:       int = 0
    agenda_items_bajados:   int = 0
    prediction_conf_delta:  float = 0.0
    resumen:                str = ""
    log:                    list = field(default_factory=list)

    def add_log(self, msg: str):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================================================
# OFFLINE CONSOLIDATION
# ============================================================

class OfflineConsolidation:
    """
    Motor de consolidación offline para AlterB3.

    Corre domingos a las 23hs (o con /dream manual).
    Opera sobre los módulos de memoria sin llamar a Gemini
    excepto para el resumen final narrativo.
    """

    # Umbrales
    UMBRAL_EPISODIO_RECURRENTE = 2        # N episodios del mismo tema = recurrente
    UMBRAL_NODO_OBSOLETO       = 0.15    # peso < umbral → candidato a eliminar
    UMBRAL_EXITO_PATRON        = 0.65    # success_rate > umbral = patrón exitoso
    DECAY_NODO_SEMANAL         = 0.05    # pesos bajan si no se usan
    BOOST_NODO_EPISODIO        = 0.10    # pesos suben si aparecen en episodios

    def run(
        self,
        memory_system,          # MemorySystem de alter_memory.py
        predictive_state=None,  # PredictiveState de alter_predictive.py
        redis_client=None,
    ) -> ConsolidationResult:
        """
        Ciclo completo de consolidación offline.
        No es async — opera sobre datos ya cargados.
        """
        result = ConsolidationResult(
            timestamp=datetime.now().isoformat()
        )

        # 1. Actualizar patrones procedurales
        self._consolidate_procedural(memory_system, result)

        # 2. Ajustar pesos semánticos
        self._consolidate_semantic(memory_system, result)

        # 3. Ajustar prioridades de agenda
        self._consolidate_agenda(memory_system, redis_client, result)

        # 4. Detectar episodios recurrentes → nuevos patrones
        self._detect_recurring_patterns(memory_system, result)

        # 5. Actualizar confianza del modelo predictivo
        if predictive_state:
            self._update_predictive_confidence(predictive_state, redis_client, result)

        result.add_log(f"Consolidación completada: "
                       f"{result.patrones_actualizados} patrones actualizados, "
                       f"{result.nodos_ajustados} nodos ajustados")

        return result

    # ----------------------------------------------------------
    # PASO 1 — Consolidación procedural
    # ----------------------------------------------------------

    def _consolidate_procedural(self, memory_system, result: ConsolidationResult):
        """
        Revisa el historial de episodios y refuerza o debilita
        patrones procedurales según evidencia acumulada.
        """
        episodes = memory_system.episodic.get_recent(30)
        if not episodes:
            return

        # Contar outcomes por tema
        tema_outcomes: dict[str, list[str]] = {}
        for ep in episodes:
            tema = ep.tema.lower()[:50]
            tema_outcomes.setdefault(tema, []).append(ep.outcome)

        patterns = memory_system.procedural._patterns
        for pattern in patterns:
            trigger_lower = pattern.trigger.lower()

            # Buscar episodios relacionados con este patrón
            exitos = 0
            fracasos = 0
            for tema, outcomes in tema_outcomes.items():
                # Overlap semántico simple
                trigger_words = set(trigger_lower.split())
                tema_words    = set(tema.split())
                if len(trigger_words & tema_words) >= 2:
                    for outcome in outcomes:
                        if outcome == "resuelto":
                            exitos += 1
                        elif outcome == "pendiente":
                            fracasos += 1

            if exitos > 0 or fracasos > 0:
                total = exitos + fracasos
                tasa_exito = exitos / total
                # Ajustar success_rate hacia la evidencia empírica
                nuevo_sr = 0.7 * pattern.success_rate + 0.3 * tasa_exito
                pattern.success_rate = float(np.clip(nuevo_sr, 0.10, 0.95))
                result.patrones_actualizados += 1
                result.add_log(
                    f"Patrón '{pattern.trigger[:40]}': "
                    f"sr {pattern.success_rate:.2f} "
                    f"({exitos}✓ {fracasos}✗)"
                )

        if result.patrones_actualizados > 0:
            memory_system.procedural._save()

    # ----------------------------------------------------------
    # PASO 2 — Consolidación semántica
    # ----------------------------------------------------------

    def _consolidate_semantic(self, memory_system, result: ConsolidationResult):
        """
        Ajusta pesos de nodos semánticos:
        - Sube peso si el nodo apareció en episodios recientes
        - Baja peso levemente si no fue mencionado (decay)
        - Elimina nodos con peso < umbral y sin episodios asociados
        """
        if not memory_system.semantic._redis:
            return

        episodes = memory_system.episodic.get_recent(20)
        episode_text = " ".join(
            f"{ep.tema} {ep.sintesis}" for ep in episodes
        ).lower()

        nodes = memory_system.semantic.get_all_nodes()
        if not nodes:
            return

        nodes_updated = []
        for node in nodes:
            nombre_lower = node.nombre.lower()

            # ¿Apareció en episodios recientes?
            if nombre_lower in episode_text:
                node.peso = float(np.clip(
                    node.peso + self.BOOST_NODO_EPISODIO, 0.0, 1.0
                ))
                result.nodos_ajustados += 1
            else:
                # Decay semanal
                node.peso = float(np.clip(
                    node.peso - self.DECAY_NODO_SEMANAL, 0.0, 1.0
                ))

            # Eliminar si peso muy bajo
            if node.peso < self.UMBRAL_NODO_OBSOLETO:
                result.nodos_eliminados += 1
                result.add_log(f"Nodo eliminado: [{node.tipo}] {node.nombre} (peso:{node.peso:.2f})")
                continue

            nodes_updated.append({
                "id":        node.id,
                "tipo":      node.tipo,
                "nombre":    node.nombre,
                "atributos": node.atributos,
                "peso":      node.peso,
                "updated":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            })

        try:
            memory_system.semantic._redis.set(
                "alter:mundo:nodos",
                json.dumps(nodes_updated, ensure_ascii=False)
            )
        except Exception:
            pass

    # ----------------------------------------------------------
    # PASO 3 — Consolidación de agenda
    # ----------------------------------------------------------

    def _consolidate_agenda(self, memory_system, redis_client, result: ConsolidationResult):
        """
        Baja prioridad de items de agenda que ya se abordaron en episodios.
        Items resueltos bajan a 0.1 (no se eliminan — pueden reactivarse).
        """
        if not redis_client:
            return

        episodes = memory_system.episodic.get_recent(20)
        temas_resueltos = {
            ep.tema.lower()[:40]
            for ep in episodes
            if ep.outcome == "resuelto"
        }

        if not temas_resueltos:
            return

        try:
            raw = redis_client.get("alter:agenda")
            if not raw:
                return
            agenda = json.loads(raw)

            for item in agenda:
                tema_lower = item.get("tema", "").lower()[:40]
                # Overlap semántico
                tema_words     = set(tema_lower.split())
                for resuelto in temas_resueltos:
                    resuelto_words = set(resuelto.split())
                    if len(tema_words & resuelto_words) >= 2:
                        if item.get("prioridad", 0) > 0.3:
                            item["prioridad"] = max(0.1, item["prioridad"] - 0.3)
                            result.agenda_items_bajados += 1
                            result.add_log(
                                f"Agenda bajada: '{item.get('tema','')[:40]}' "
                                f"(resuelto en episodio)"
                            )
                        break

            redis_client.set("alter:agenda", json.dumps(agenda, ensure_ascii=False))
        except Exception:
            pass

    # ----------------------------------------------------------
    # PASO 4 — Detección de patrones recurrentes
    # ----------------------------------------------------------

    def _detect_recurring_patterns(self, memory_system, result: ConsolidationResult):
        """
        Detecta temas que aparecen múltiples veces en episodios
        y genera nuevos patrones procedurales si no existen.
        """
        episodes = memory_system.episodic.get_recent(30)
        if len(episodes) < 3:
            return

        # Contar frecuencia de temas
        tema_count: dict[str, int] = {}
        for ep in episodes:
            tema = ep.tema.lower()[:50]
            tema_count[tema] = tema_count.get(tema, 0) + 1

        # Temas que aparecen N+ veces
        recurrentes = [
            (tema, count)
            for tema, count in tema_count.items()
            if count >= self.UMBRAL_EPISODIO_RECURRENTE
        ]

        for tema, count in recurrentes:
            # ¿Ya existe un patrón para este tema?
            ya_existe = any(
                tema[:20] in p.trigger.lower()
                for p in memory_system.procedural._patterns
            )
            if ya_existe:
                continue

            # Construir patrón desde episodios
            eps_del_tema = [
                ep for ep in episodes
                if tema[:20] in ep.tema.lower()
            ]
            outcomes = [ep.outcome for ep in eps_del_tema]
            tasa_exito = outcomes.count("resuelto") / len(outcomes) if outcomes else 0.5

            trigger  = f"tema recurrente detectado: {tema[:60]}"
            response = f"este tema aparece {count} veces — considerar respuesta específica"

            context_sig = {
                "tema_recurrente": True,
                "frecuencia":      count,
                "tasa_exito":      round(tasa_exito, 2),
            }

            memory_system.procedural.add_pattern(
                trigger, response, context_sig,
                initial_success=float(np.clip(tasa_exito, 0.3, 0.8))
            )
            result.patrones_nuevos += 1
            result.add_log(
                f"Nuevo patrón desde recurrencia: '{tema[:40]}' "
                f"({count}x, éxito:{tasa_exito:.2f})"
            )

    # ----------------------------------------------------------
    # PASO 5 — Actualización de confianza predictiva
    # ----------------------------------------------------------

    def _update_predictive_confidence(
        self,
        predictive_state,
        redis_client,
        result: ConsolidationResult
    ):
        """
        Actualiza model_confidence del Predictive Model
        basado en el historial de errores de la semana.
        """
        if not predictive_state.error_history:
            return

        # Media del error semanal
        error_medio = sum(predictive_state.error_history) / len(predictive_state.error_history)
        nueva_conf  = float(np.clip(1.0 - error_medio, 0.2, 0.95))
        delta       = nueva_conf - predictive_state.model_confidence

        predictive_state.model_confidence = nueva_conf
        result.prediction_conf_delta = delta

        result.add_log(
            f"Predictive confidence: {predictive_state.model_confidence:.2f} "
            f"(delta:{delta:+.2f}, error_medio:{error_medio:.2f})"
        )

        # Persistir
        if redis_client:
            try:
                from alter_predictive import serialize as pred_serialize
                redis_client.set(
                    "alter:predictive:state",
                    pred_serialize(predictive_state)
                )
            except Exception:
                pass


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    from alter_memory import MemorySystem, ProceduralMemory, EpisodicMemory, Episode
    from alter_predictive import PredictiveState

    # Crear sistema de memoria sin Redis
    ms = MemorySystem(None)

    # Agregar algunos patrones procedurales
    ms.procedural.add_pattern(
        "usuario quiere acción concreta",
        "dar código primero",
        {"user_intent": "quiere_accion"},
        initial_success=0.6
    )
    ms.procedural.add_pattern(
        "usuario quiere análisis",
        "explicar en profundidad",
        {"user_intent": "quiere_analisis"},
        initial_success=0.7
    )

    # Crear estado predictivo con historial de errores
    pred = PredictiveState(
        error_history=[0.3, 0.4, 0.2, 0.5, 0.3],
        model_confidence=0.5
    )

    consolidation = OfflineConsolidation()

    # Test 1: run() no explota sin Redis
    result = consolidation.run(ms, pred, redis_client=None)
    if result is None:
        errors.append("FAIL: run() retornó None")

    # Test 2: actualización de confianza predictiva
    if result.prediction_conf_delta == 0.0 and pred.error_history:
        # La confianza debería haber cambiado
        pass  # puede ser 0 si ya estaba calibrada

    expected_conf = 1.0 - (sum([0.3, 0.4, 0.2, 0.5, 0.3]) / 5)
    if abs(pred.model_confidence - expected_conf) > 0.01:
        errors.append(
            f"FAIL: model_confidence esperado {expected_conf:.2f}, "
            f"got {pred.model_confidence:.2f}"
        )

    # Test 3: ConsolidationResult tiene log
    if not isinstance(result.log, list):
        errors.append("FAIL: result.log debería ser lista")

    # Test 4: detect_recurring sin episodios no explota
    result2 = consolidation.run(ms, redis_client=None)
    if result2 is None:
        errors.append("FAIL: run() sin predictive_state retornó None")

    # Test 5: campos de ConsolidationResult son numéricos correctos
    for campo in ["patrones_actualizados", "patrones_nuevos", "nodos_ajustados",
                  "nodos_eliminados", "agenda_items_bajados"]:
        val = getattr(result, campo)
        if not isinstance(val, int) or val < 0:
            errors.append(f"FAIL: {campo} debería ser int >= 0, got {val}")

    return errors


if __name__ == "__main__":
    print("=== alter_consolidation.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo consolidación ===")
    from alter_memory import MemorySystem
    from alter_predictive import PredictiveState

    ms   = MemorySystem(None)
    pred = PredictiveState(
        error_history=[0.6, 0.7, 0.5, 0.8, 0.6, 0.4],
        model_confidence=0.4
    )

    ms.procedural.add_pattern(
        "usuario pide implementación pero ALTER da teoría",
        "priorizar código sobre explicación",
        {"user_intent": "quiere_accion", "prediction_error_high": True},
        initial_success=0.45
    )

    consolidation = OfflineConsolidation()
    result = consolidation.run(ms, pred)

    print(f"Patrones actualizados: {result.patrones_actualizados}")
    print(f"Patrones nuevos:       {result.patrones_nuevos}")
    print(f"Nodos ajustados:       {result.nodos_ajustados}")
    print(f"Confianza predictiva:  {pred.model_confidence:.2f} "
          f"(delta:{result.prediction_conf_delta:+.2f})")
    print(f"\nLog de consolidación:")
    for entry in result.log:
        print(f"  {entry}")
