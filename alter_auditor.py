"""
alter_auditor.py — Architecture Auditor de AlterB4 (Fase 4)

Lee métricas, Self-Model y Meta-Learning para detectar
cuellos de botella y proponer micro-ajustes estructurales.

Cuándo corre:
    Junto con DREAM — domingos, usuario ausente.
    También disponible con /auditar por Telegram.

Qué audita:
    - ¿Qué módulo tiene más warnings acumuladas?
    - ¿Hay drift en prediction_error a lo largo del tiempo?
    - ¿El workspace está siempre lleno o siempre vacío?
    - ¿Los patrones procedurales se degradan sistemáticamente?
    - ¿La homeostasis se recupera bien entre sesiones?
    - ¿Las políticas de meta-learning se están activando?

Output:
    - Reporte estructurado en KAIROS
    - Lista de propuestas priorizadas (para aprobación humana)
    - NO aplica cambios solo — reporta y espera

Redis keys:
    alter:auditor:last_report  — último reporte generado
    alter:auditor:proposals    — propuestas pendientes de aprobación
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class AuditFinding:
    """Un hallazgo del auditor."""
    severidad:   str    # "info" | "warning" | "critical"
    modulo:      str
    descripcion: str
    metrica:     str
    valor:       float
    umbral:      float
    recomendacion: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditProposal:
    """Una propuesta de ajuste estructural."""
    id:          str
    prioridad:   int    # 1=alta, 2=media, 3=baja
    titulo:      str
    descripcion: str
    modulo:      str
    ajuste:      dict
    evidencia:   list   # list[str] — hallazgos que la soportan
    estado:      str = "pendiente"  # "pendiente" | "aprobada" | "rechazada"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditReport:
    """Reporte completo de una auditoría."""
    timestamp:   str
    hallazgos:   list   # list[AuditFinding]
    propuestas:  list   # list[AuditProposal]
    resumen:     str
    score_salud: float  # 0..1 — salud general del sistema

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditReport":
        return cls(
            timestamp  = d.get("timestamp", ""),
            hallazgos  = [AuditFinding(**h) for h in d.get("hallazgos", [])],
            propuestas = [AuditProposal(**p) for p in d.get("propuestas", [])],
            resumen    = d.get("resumen", ""),
            score_salud= d.get("score_salud", 0.5),
        )


# ============================================================
# ARCHITECTURE AUDITOR
# ============================================================

class ArchitectureAuditor:
    """
    Audita el sistema completo y propone micro-ajustes estructurales.
    No aplica cambios — reporta y espera aprobación.
    """

    # Umbrales de auditoría
    UMBRAL_ERROR_CRITICO     = 0.70
    UMBRAL_ERROR_WARNING     = 0.55
    UMBRAL_OVERFLOW_WARNING  = 0.35
    UMBRAL_CLARIDAD_WARNING  = 0.40
    UMBRAL_FATIGA_WARNING    = 0.65
    UMBRAL_SR_WARNING        = 0.40
    UMBRAL_CONF_WARNING      = 0.40

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def run(
        self,
        metrics_summary,    # MetricsSummary
        self_model,         # SelfModel
        metalearning,       # MetaLearningEngine
    ) -> AuditReport:
        """
        Corre la auditoría completa.
        Lee estado de todos los módulos y genera hallazgos + propuestas.
        """
        hallazgos  = []
        propuestas = []

        # Auditar cada módulo
        hallazgos += self._audit_homeostasis(metrics_summary)
        hallazgos += self._audit_workspace(metrics_summary)
        hallazgos += self._audit_predictive(metrics_summary, self_model)
        hallazgos += self._audit_procedural(metrics_summary)
        hallazgos += self._audit_policy(metrics_summary)
        hallazgos += self._audit_simulator(metrics_summary)
        hallazgos += self._audit_metalearning(metalearning)

        # Generar propuestas desde hallazgos
        propuestas = self._generate_proposals(hallazgos, self_model)

        # Score de salud general
        criticos  = sum(1 for h in hallazgos if h.severidad == "critical")
        warnings  = sum(1 for h in hallazgos if h.severidad == "warning")
        score     = float(np.clip(
            1.0 - criticos * 0.15 - warnings * 0.05, 0.0, 1.0
        ))

        # Resumen
        resumen = self._generate_summary(hallazgos, propuestas, score)

        report = AuditReport(
            timestamp   = datetime.now().isoformat(),
            hallazgos   = hallazgos,
            propuestas  = propuestas,
            resumen     = resumen,
            score_salud = score,
        )

        self._save_report(report)
        return report

    # ----------------------------------------------------------
    # AUDITORÍAS POR MÓDULO
    # ----------------------------------------------------------

    def _audit_homeostasis(self, ms) -> list[AuditFinding]:
        findings = []
        if ms.hs_claridad_media < self.UMBRAL_CLARIDAD_WARNING:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "homeostasis",
                descripcion  = "Claridad media por debajo del umbral operativo",
                metrica      = "hs_claridad_media",
                valor        = ms.hs_claridad_media,
                umbral       = self.UMBRAL_CLARIDAD_WARNING,
                recomendacion= "Revisar recovery_state en ciclo_drives. "
                               "Posible drift de fatiga acumulada.",
            ))
        if ms.hs_fatiga_media > self.UMBRAL_FATIGA_WARNING:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "homeostasis",
                descripcion  = "Fatiga media alta — recovery insuficiente",
                metrica      = "hs_fatiga_media",
                valor        = ms.hs_fatiga_media,
                umbral       = self.UMBRAL_FATIGA_WARNING,
                recomendacion= "Aumentar tasa de recovery o reducir costo por turno.",
            ))
        if ms.hs_energia_media < 0.35:
            findings.append(AuditFinding(
                severidad    = "critical",
                modulo       = "homeostasis",
                descripcion  = "Energía media crítica — sistema en modo degradado",
                metrica      = "hs_energia_media",
                valor        = ms.hs_energia_media,
                umbral       = 0.35,
                recomendacion= "Reducir complejidad de entrada o aumentar recovery.",
            ))
        return findings

    def _audit_workspace(self, ms) -> list[AuditFinding]:
        findings = []
        if ms.ws_overflow_rate > self.UMBRAL_OVERFLOW_WARNING:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "workspace",
                descripcion  = f"Workspace en overflow {ms.ws_overflow_rate:.0%} de los turnos",
                metrica      = "ws_overflow_rate",
                valor        = ms.ws_overflow_rate,
                umbral       = self.UMBRAL_OVERFLOW_WARNING,
                recomendacion= "Revisar generación de candidatos. "
                               "Posible política mp002 activa.",
            ))
        if ms.ws_noise_ratio > 0.50:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "workspace",
                descripcion  = "Alto ratio ruido/señal en workspace",
                metrica      = "ws_noise_ratio",
                valor        = ms.ws_noise_ratio,
                umbral       = 0.50,
                recomendacion= "Revisar score_candidate — posible calibración incorrecta.",
            ))
        return findings

    def _audit_predictive(self, ms, self_model) -> list[AuditFinding]:
        findings = []
        if ms.pred_error_medio > self.UMBRAL_ERROR_CRITICO:
            findings.append(AuditFinding(
                severidad    = "critical",
                modulo       = "predictive",
                descripcion  = "Error de predicción sistémicamente alto",
                metrica      = "pred_error_medio",
                valor        = ms.pred_error_medio,
                umbral       = self.UMBRAL_ERROR_CRITICO,
                recomendacion= "Revisar infer_intent — posibles señales léxicas desactualizadas.",
            ))
        elif ms.pred_error_medio > self.UMBRAL_ERROR_WARNING:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "predictive",
                descripcion  = "Error de predicción elevado",
                metrica      = "pred_error_medio",
                valor        = ms.pred_error_medio,
                umbral       = self.UMBRAL_ERROR_WARNING,
                recomendacion= "Modelo calibrándose. Monitorear 2 semanas más.",
            ))

        # Intención con peor rendimiento
        if hasattr(self_model, "intent_performance") and self_model.intent_performance:
            worst = min(self_model.intent_performance, key=lambda i: i.success_rate)
            if worst.success_rate < 0.45:
                findings.append(AuditFinding(
                    severidad    = "warning",
                    modulo       = "predictive",
                    descripcion  = f"Baja precisión en intención '{worst.intent}'",
                    metrica      = f"intent_sr.{worst.intent}",
                    valor        = worst.success_rate,
                    umbral       = 0.45,
                    recomendacion= f"Agregar señales léxicas para '{worst.intent}' "
                                   f"en INTENT_SIGNALS.",
                ))
        return findings

    def _audit_procedural(self, ms) -> list[AuditFinding]:
        findings = []
        if ms.proc_sr_medio < self.UMBRAL_SR_WARNING and ms.proc_patrones_activos > 0:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "procedural",
                descripcion  = "Success rate medio de patrones bajo",
                metrica      = "proc_sr_medio",
                valor        = ms.proc_sr_medio,
                umbral       = self.UMBRAL_SR_WARNING,
                recomendacion= "Revisar patrones con sr < 0.3. "
                               "Posible política mp004 activa.",
            ))
        if ms.proc_patrones_degradados > ms.proc_patrones_activos:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "procedural",
                descripcion  = "Más patrones degradados que activos",
                metrica      = "proc_patrones_degradados",
                valor        = float(ms.proc_patrones_degradados),
                umbral       = float(ms.proc_patrones_activos),
                recomendacion= "Limpiar patrones con sr < 0.2 y sin usos recientes.",
            ))
        return findings

    def _audit_policy(self, ms) -> list[AuditFinding]:
        findings = []
        if ms.pol_override_rate > 0.60:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "policy",
                descripcion  = f"Arbiter sobrescribe Gemini {ms.pol_override_rate:.0%} del tiempo",
                metrica      = "pol_override_rate",
                valor        = ms.pol_override_rate,
                umbral       = 0.60,
                recomendacion= "Gemini y Arbiter no están alineados. "
                               "Revisar árbol de decisión.",
            ))
        return findings

    def _audit_simulator(self, ms) -> list[AuditFinding]:
        findings = []
        if ms.sim_activaciones == 0:
            findings.append(AuditFinding(
                severidad    = "info",
                modulo       = "simulator",
                descripcion  = "Simulator sin activaciones — condiciones nunca alcanzadas",
                metrica      = "sim_activaciones",
                valor        = 0.0,
                umbral       = 1.0,
                recomendacion= "Posiblemente los umbrales de activación son muy altos.",
            ))
        elif ms.sim_override_rate > 0.70:
            findings.append(AuditFinding(
                severidad    = "warning",
                modulo       = "simulator",
                descripcion  = f"Simulator override demasiado frecuente ({ms.sim_override_rate:.0%})",
                metrica      = "sim_override_rate",
                valor        = ms.sim_override_rate,
                umbral       = 0.70,
                recomendacion= "Subir OVERRIDE_THRESHOLD o revisar heurísticas de scoring.",
            ))
        return findings

    def _audit_metalearning(self, metalearning) -> list[AuditFinding]:
        findings = []
        activas = metalearning.get_active_policies()
        activadas = [p for p in activas if p.activaciones > 0]
        if not activadas and len(activas) > 0:
            findings.append(AuditFinding(
                severidad    = "info",
                modulo       = "metalearning",
                descripcion  = "Ninguna política cognitiva se activó aún",
                metrica      = "ml_activaciones",
                valor        = 0.0,
                umbral       = 1.0,
                recomendacion= "Normal en las primeras semanas. "
                               "Las condiciones deben alcanzarse.",
            ))
        return findings

    # ----------------------------------------------------------
    # PROPUESTAS
    # ----------------------------------------------------------

    def _generate_proposals(
        self,
        hallazgos: list[AuditFinding],
        self_model,
    ) -> list[AuditProposal]:
        proposals = []
        criticos  = [h for h in hallazgos if h.severidad == "critical"]
        warnings  = [h for h in hallazgos if h.severidad == "warning"]

        # Propuesta por cada hallazgo crítico
        for h in criticos:
            proposals.append(AuditProposal(
                id          = f"p_{h.modulo}_{int(time.time())}",
                prioridad   = 1,
                titulo      = f"CRÍTICO — {h.modulo}: {h.descripcion[:50]}",
                descripcion = h.recomendacion,
                modulo      = h.modulo,
                ajuste      = {"metrica": h.metrica, "valor": h.valor},
                evidencia   = [h.descripcion],
            ))

        # Agrupar warnings por módulo → una propuesta por módulo
        modulos_warnings: dict[str, list[AuditFinding]] = {}
        for h in warnings:
            modulos_warnings.setdefault(h.modulo, []).append(h)

        for modulo, hs in modulos_warnings.items():
            proposals.append(AuditProposal(
                id          = f"p_{modulo}_warn_{int(time.time())}",
                prioridad   = 2,
                titulo      = f"{modulo}: {len(hs)} advertencias detectadas",
                descripcion = " | ".join(h.recomendacion[:50] for h in hs),
                modulo      = modulo,
                ajuste      = {"warnings": [h.metrica for h in hs]},
                evidencia   = [h.descripcion for h in hs],
            ))

        proposals.sort(key=lambda p: p.prioridad)
        return proposals[:8]  # máx 8 propuestas por auditoría

    # ----------------------------------------------------------
    # RESUMEN Y PERSISTENCIA
    # ----------------------------------------------------------

    def _generate_summary(
        self,
        hallazgos: list[AuditFinding],
        propuestas: list[AuditProposal],
        score: float,
    ) -> str:
        criticos = sum(1 for h in hallazgos if h.severidad == "critical")
        warnings = sum(1 for h in hallazgos if h.severidad == "warning")
        infos    = sum(1 for h in hallazgos if h.severidad == "info")

        estado = "óptimo" if score > 0.85 else \
                 "bueno"  if score > 0.70 else \
                 "regular" if score > 0.50 else "degradado"

        lines = [
            f"Sistema en estado {estado} (score:{score:.2f}).",
            f"Hallazgos: {criticos} críticos | {warnings} advertencias | {infos} info.",
        ]
        if propuestas:
            top = propuestas[0]
            lines.append(f"Propuesta prioritaria: {top.titulo[:60]}")
        if criticos == 0 and warnings <= 1:
            lines.append("No se requieren ajustes urgentes.")

        return " ".join(lines)

    def _save_report(self, report: AuditReport):
        if not self._redis:
            return
        try:
            self._redis.set(
                "alter:auditor:last_report",
                json.dumps(report.to_dict(), ensure_ascii=False)
            )
            if report.propuestas:
                self._redis.set(
                    "alter:auditor:proposals",
                    json.dumps([p.to_dict() for p in report.propuestas],
                               ensure_ascii=False)
                )
        except Exception:
            pass

    def report_str(self, report: AuditReport) -> str:
        """Formato legible para Telegram y KAIROS."""
        lines = [
            f"🔍 AUDITORÍA — {report.timestamp[:16]}",
            f"Salud del sistema: {report.score_salud:.0%}",
            f"{report.resumen}",
        ]
        if report.hallazgos:
            criticos = [h for h in report.hallazgos if h.severidad == "critical"]
            warnings = [h for h in report.hallazgos if h.severidad == "warning"]
            for h in criticos:
                lines.append(f"🔴 [{h.modulo}] {h.descripcion[:60]}")
            for h in warnings[:3]:
                lines.append(f"🟡 [{h.modulo}] {h.descripcion[:60]}")
        if report.propuestas:
            lines.append(f"\nPropuestas ({len(report.propuestas)}):")
            for p in report.propuestas[:3]:
                prioridad_str = "❗" if p.prioridad == 1 else "⚡" if p.prioridad == 2 else "💡"
                lines.append(f"  {prioridad_str} {p.titulo[:60]}")
        return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []
    from alter_metrics import MetricsSummary
    from alter_selfmodel import SelfModel, IntentPerformance
    from alter_metalearning import MetaLearningEngine

    auditor = ArchitectureAuditor(redis_client=None)

    # Test 1: run con estado vacío no explota
    ms = MetricsSummary(timestamp=datetime.now().isoformat())
    sm = SelfModel()
    ml = MetaLearningEngine(redis_client=None)
    try:
        report = auditor.run(ms, sm, ml)
        if report is None:
            errors.append("FAIL: run retornó None")
    except Exception as e:
        errors.append(f"FAIL: run explotó: {e}")

    # Test 2: score_salud entre 0 y 1
    if not (0.0 <= report.score_salud <= 1.0):
        errors.append(f"FAIL: score_salud fuera de rango: {report.score_salud}")

    # Test 3: energía crítica genera hallazgo critical
    ms2 = MetricsSummary(
        timestamp=datetime.now().isoformat(),
        hs_energia_media=0.20,
    )
    report2 = auditor.run(ms2, sm, ml)
    criticos = [h for h in report2.hallazgos if h.severidad == "critical"]
    if not criticos:
        errors.append("FAIL: energía crítica no generó hallazgo critical")

    # Test 4: score baja con más hallazgos críticos
    criticos2 = [h for h in report2.hallazgos if h.severidad == "critical"]
    criticos1 = [h for h in report.hallazgos if h.severidad == "critical"]
    if len(criticos2) <= len(criticos1) and report2.score_salud > report.score_salud:
        errors.append("FAIL: más críticos debería dar score menor")

    # Test 5: propuestas se generan desde hallazgos
    if criticos and not report2.propuestas:
        errors.append("FAIL: hallazgos críticos deberían generar propuestas")

    # Test 6: report_str no explota
    snap = auditor.report_str(report2)
    if "AUDITORÍA" not in snap:
        errors.append("FAIL: report_str malformado")

    # Test 7: AuditReport round-trip
    d = report2.to_dict()
    report3 = AuditReport.from_dict(d)
    if report3.score_salud != report2.score_salud:
        errors.append("FAIL: AuditReport from_dict perdió score_salud")

    return errors


if __name__ == "__main__":
    print("=== alter_auditor.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo Auditoría ===")
    from alter_metrics import MetricsSummary
    from alter_selfmodel import SelfModel, IntentPerformance
    from alter_metalearning import MetaLearningEngine

    auditor = ArchitectureAuditor(redis_client=None)
    ms = MetricsSummary(
        timestamp=datetime.now().isoformat(),
        hs_energia_media=0.55,
        hs_fatiga_media=0.70,
        hs_claridad_media=0.42,
        ws_overflow_rate=0.38,
        pred_error_medio=0.62,
        pred_calibration=0.38,
        proc_sr_medio=0.45,
        proc_patrones_activos=3,
        proc_patrones_degradados=5,
        pol_override_rate=0.15,
        sim_activaciones=8,
        sim_override_rate=0.25,
    )
    sm = SelfModel()
    sm.intent_performance = [
        IntentPerformance("quiere_exploracion", 0.40, 10, 0.60),
        IntentPerformance("quiere_accion",      0.78, 12, 0.22),
    ]
    ml = MetaLearningEngine(redis_client=None)
    report = auditor.run(ms, sm, ml)
    print(auditor.report_str(report))
    print(f"\nHallazgos totales: {len(report.hallazgos)}")
    print(f"Propuestas totales: {len(report.propuestas)}")
