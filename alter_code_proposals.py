"""
alter_code_proposals.py — Proposals con pseudo-diffs de AlterB5 (Fase 4)

Convierte hipótesis estructurales (no_replayable_yet) en propuestas
concretas con evidencia, impacto estimado y pseudo-diff.

NO ejecuta nada. Solo propone con suficiente detalle para decidir.

Diferencia con AuditProposal (B4):
    AuditProposal — hallazgo de runtime, propuesta de ajuste paramétrico
    CodeProposal  — propuesta de cambio estructural en el código,
                    con pseudo-diff y plan de rollback

Redis keys:
    alter:b5:proposals  — propuestas pendientes de aprobación humana
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class PseudoDiff:
    """
    Representación legible de un cambio de código propuesto.
    No es un diff real — es una descripción estructurada del cambio.
    """
    archivo:      str
    tipo:         str    # "extract_function" | "split_class" | "move_logic" | "add_module"
    descripcion:  str
    antes:        str    # descripción del estado actual
    despues:      str    # descripción del estado propuesto
    lineas_afectadas: list  # list[int] — líneas relevantes

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        """Formato legible para Telegram y KAIROS."""
        lines = [
            f"📄 {self.archivo} — {self.tipo}",
            f"ANTES:  {self.antes[:100]}",
            f"DESPUÉS:{self.despues[:100]}",
        ]
        if self.lineas_afectadas:
            lines.append(f"Líneas: {self.lineas_afectadas[:5]}")
        return "\n".join(lines)


@dataclass
class CodeProposal:
    id:              str
    hypothesis_id:   str
    titulo:          str
    problema:        str     # qué problema resuelve
    evidencia:       list    # list[str]
    impacto:         str     # "bajo" | "medio" | "alto"
    riesgo:          str
    modulos_afectados: list  # list[str]
    pseudo_diff:     Optional[PseudoDiff]
    plan_rollback:   str
    estado:          str     # "pendiente" | "aprobada" | "rechazada" | "aplicada"
    created_at:      str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CodeProposal":
        pd = None
        if d.get("pseudo_diff"):
            pd = PseudoDiff(**d["pseudo_diff"])
        return cls(
            id               = d["id"],
            hypothesis_id    = d["hypothesis_id"],
            titulo           = d["titulo"],
            problema         = d["problema"],
            evidencia        = d.get("evidencia", []),
            impacto          = d.get("impacto", "medio"),
            riesgo           = d.get("riesgo", "medio"),
            modulos_afectados= d.get("modulos_afectados", []),
            pseudo_diff      = pd,
            plan_rollback    = d.get("plan_rollback", ""),
            estado           = d.get("estado", "pendiente"),
            created_at       = d.get("created_at", datetime.now().isoformat()),
        )

    def render(self) -> str:
        """Formato legible para Telegram."""
        lines = [
            f"📋 PROPUESTA [{self.id}]: {self.titulo}",
            f"Problema: {self.problema[:120]}",
            f"Impacto: {self.impacto} | Riesgo: {self.riesgo}",
            f"Módulos: {', '.join(self.modulos_afectados[:3])}",
        ]
        if self.evidencia:
            lines.append(f"Evidencia: {self.evidencia[0][:80]}")
        if self.pseudo_diff:
            lines.append(self.pseudo_diff.render())
        lines.append(f"Rollback: {self.plan_rollback[:80]}")
        return "\n".join(lines)


# ============================================================
# PROPOSAL ENGINE
# ============================================================

class ProposalEngine:
    """
    Convierte hipótesis estructurales en propuestas con pseudo-diff.
    Lee el Code Map para extraer contexto real del código.

    Solo genera propuestas para hipótesis con estado "no_replayable_yet"
    o "propuesta" y tipo en ("split", "refactor", "nuevo_modulo", "pipeline").

    No ejecuta nada. No modifica nada.
    """

    TIPOS_ESTRUCTURALES = {"split", "refactor", "nuevo_modulo", "pipeline"}

    def generate(
        self,
        hypotheses: list,   # list[ArchitectureHypothesis]
        repo_map,           # RepoMap de alter_code_map
        existing: list = None,  # list[CodeProposal] ya existentes
    ) -> list:
        """
        Genera propuestas para hipótesis estructurales.
        No duplica propuestas para la misma hipótesis.
        """
        existing = existing or []
        hyps_con_propuesta = {p.hypothesis_id for p in existing}

        nuevas = []
        for hyp in hypotheses:
            if hyp.id in hyps_con_propuesta:
                continue
            if hyp.tipo not in self.TIPOS_ESTRUCTURALES:
                continue
            if hyp.estado not in ("no_replayable_yet", "propuesta"):
                continue

            propuesta = self._build_proposal(hyp, repo_map)
            if propuesta:
                nuevas.append(propuesta)

        return nuevas

    def _build_proposal(self, hyp, repo_map) -> Optional[CodeProposal]:
        """Construye una propuesta concreta para una hipótesis."""
        import uuid

        if hyp.tipo == "split":
            return self._proposal_split(hyp, repo_map, str(uuid.uuid4())[:8])
        elif hyp.tipo == "refactor":
            return self._proposal_refactor(hyp, repo_map, str(uuid.uuid4())[:8])
        elif hyp.tipo == "nuevo_modulo":
            return self._proposal_nuevo_modulo(hyp, repo_map, str(uuid.uuid4())[:8])
        elif hyp.tipo == "pipeline":
            return self._proposal_pipeline(hyp, str(uuid.uuid4())[:8])
        return None

    def _proposal_split(self, hyp, repo_map, pid: str) -> CodeProposal:
        """Propuesta para dividir un módulo o clase grande."""
        modulo_map = repo_map.get_module(f"alter_{hyp.modulo}.py") if repo_map else None

        # Extraer funciones más largas del módulo
        funciones_largas = []
        if modulo_map:
            todas = modulo_map.get_all_functions()
            funciones_largas = sorted(
                [f for f in todas if f.lineas > 100],
                key=lambda f: f.lineas, reverse=True
            )[:3]

        # Construir pseudo-diff
        if funciones_largas:
            nombres = [f.nombre for f in funciones_largas]
            lineas  = [f.linea for f in funciones_largas]
            antes   = (f"Un solo archivo con {modulo_map.lineas if modulo_map else '?'} líneas. "
                      f"Funciones largas: {', '.join(nombres[:2])}")
            despues = (f"Separar en: alter_{hyp.modulo}_core.py (lógica principal), "
                      f"alter_{hyp.modulo}_pipeline.py (ciclo de procesamiento)")
        else:
            antes   = f"alter_{hyp.modulo}.py con toda la lógica concentrada"
            despues = f"Dividir en submódulos por responsabilidad"
            lineas  = []

        pd = PseudoDiff(
            archivo          = f"alter_{hyp.modulo}.py",
            tipo             = "split_class",
            descripcion      = hyp.titulo,
            antes            = antes,
            despues          = despues,
            lineas_afectadas = lineas,
        )

        plan = (f"Mantener alter_{hyp.modulo}.py como fachada que importa los nuevos módulos. "
                "Todo el código existente sigue funcionando sin cambios en los imports externos.")

        return CodeProposal(
            id               = pid,
            hypothesis_id    = hyp.id,
            titulo           = hyp.titulo,
            problema         = hyp.descripcion,
            evidencia        = hyp.evidencia,
            impacto          = hyp.impacto,
            riesgo           = hyp.riesgo,
            modulos_afectados= [f"alter_{hyp.modulo}.py"],
            pseudo_diff      = pd,
            plan_rollback    = plan,
            estado           = "pendiente",
        )

    def _proposal_refactor(self, hyp, repo_map, pid: str) -> CodeProposal:
        """Propuesta para refactorizar funciones largas."""
        modulo_map = repo_map.get_module(f"alter_{hyp.modulo}.py") if repo_map else None

        funciones_target = []
        if modulo_map:
            todas = modulo_map.get_all_functions()
            # Buscar funciones mencionadas en la evidencia
            palabras_ev = set(" ".join(hyp.evidencia).lower().split())
            funciones_target = [
                f for f in todas
                if any(p in f.nombre.lower() for p in palabras_ev)
                and f.lineas > 80
            ][:3]

        if funciones_target:
            nombres = [f.nombre for f in funciones_target]
            lineas  = [f.linea for f in funciones_target]
            antes   = f"Funciones de {funciones_target[0].lineas}+ líneas: {', '.join(nombres)}"
            despues = (f"Extraer subfunciones con responsabilidad única. "
                      f"Cada función < 60 líneas con docstring claro.")
        else:
            antes   = "Funciones con múltiples responsabilidades mezcladas"
            despues = "Extraer helpers privados por responsabilidad"
            lineas  = []

        pd = PseudoDiff(
            archivo          = f"alter_{hyp.modulo}.py",
            tipo             = "extract_function",
            descripcion      = hyp.titulo,
            antes            = antes,
            despues          = despues,
            lineas_afectadas = lineas,
        )

        plan = ("Los nombres públicos no cambian. "
                "Las subfunciones son privadas (_nombre). "
                "Tests de invariantes actuales siguen pasando.")

        return CodeProposal(
            id               = pid,
            hypothesis_id    = hyp.id,
            titulo           = hyp.titulo,
            problema         = hyp.descripcion,
            evidencia        = hyp.evidencia,
            impacto          = hyp.impacto,
            riesgo           = hyp.riesgo,
            modulos_afectados= [f"alter_{hyp.modulo}.py"],
            pseudo_diff      = pd,
            plan_rollback    = plan,
            estado           = "pendiente",
        )

    def _proposal_nuevo_modulo(self, hyp, repo_map, pid: str) -> CodeProposal:
        """Propuesta para crear un módulo nuevo."""
        pd = PseudoDiff(
            archivo          = hyp.evidencia[0][:50] if hyp.evidencia else f"alter_{hyp.modulo}.py",
            tipo             = "add_module",
            descripcion      = hyp.titulo,
            antes            = "Módulo referenciado en spec pero no implementado en repo",
            despues          = f"Crear archivo con estructura base: dataclasses, clase principal, tests",
            lineas_afectadas = [],
        )

        plan = ("Crear el archivo nuevo sin modificar los existentes. "
                "Integrar gradualmente actualizando los imports en alter_brain.py.")

        return CodeProposal(
            id               = pid,
            hypothesis_id    = hyp.id,
            titulo           = hyp.titulo,
            problema         = hyp.descripcion,
            evidencia        = hyp.evidencia,
            impacto          = hyp.impacto,
            riesgo           = hyp.riesgo,
            modulos_afectados= [f"alter_{hyp.modulo}.py"],
            pseudo_diff      = pd,
            plan_rollback    = plan,
            estado           = "pendiente",
        )

    def _proposal_pipeline(self, hyp, pid: str) -> CodeProposal:
        """Propuesta para cambiar el orden o estructura del pipeline."""
        pd = PseudoDiff(
            archivo          = "alter_brain.py",
            tipo             = "move_logic",
            descripcion      = hyp.titulo,
            antes            = "Pipeline actual en procesar_turno/procesar_input",
            despues          = hyp.descripcion[:120],
            lineas_afectadas = [],
        )

        plan = ("Mantener pipeline anterior como fallback con flag ALTERB3_ENABLED. "
                "Activar nuevo pipeline con feature flag específico.")

        return CodeProposal(
            id               = pid,
            hypothesis_id    = hyp.id,
            titulo           = hyp.titulo,
            problema         = hyp.descripcion,
            evidencia        = hyp.evidencia,
            impacto          = hyp.impacto,
            riesgo           = hyp.riesgo,
            modulos_afectados= ["alter_brain.py"],
            pseudo_diff      = pd,
            plan_rollback    = plan,
            estado           = "pendiente",
        )

    # ----------------------------------------------------------
    # PERSISTENCIA
    # ----------------------------------------------------------

    def save(self, proposals: list, redis_client) -> bool:
        if not redis_client:
            return False
        try:
            redis_client.set(
                "alter:b5:proposals",
                json.dumps([p.to_dict() for p in proposals], ensure_ascii=False)
            )
            return True
        except Exception:
            return False

    def load(self, redis_client) -> list:
        if not redis_client:
            return []
        try:
            raw = redis_client.get("alter:b5:proposals")
            if raw:
                return [CodeProposal.from_dict(d) for d in json.loads(raw)]
        except Exception:
            pass
        return []

    def snapshot_str(self, proposals: list) -> str:
        if not proposals:
            return "[PROPUESTAS] Sin propuestas estructurales."
        pendientes = [p for p in proposals if p.estado == "pendiente"]
        lines      = [
            f"[PROPUESTAS] {len(proposals)} total | "
            f"{len(pendientes)} pendientes de aprobación"
        ]
        for p in pendientes[:3]:
            lines.append(
                f"  📋 [{p.id}] {p.titulo[:55]} "
                f"(impacto:{p.impacto} riesgo:{p.riesgo})"
            )
        return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests() -> list[str]:
    errors = []

    from alter_architecture_hypotheses import ArchitectureHypothesis
    from alter_code_map import CodeMapper

    engine = ProposalEngine()
    mapper = CodeMapper()
    repo   = mapper.scan(".")

    # Test 1: generate sin hipótesis no explota
    try:
        props = engine.generate([], repo)
        if not isinstance(props, list):
            errors.append("FAIL: generate no retornó lista")
    except Exception as e:
        errors.append(f"FAIL: generate explotó: {e}")

    # Test 2: hipótesis no_replayable_yet genera propuesta
    hyp_split = ArchitectureHypothesis(
        id="h_split", titulo="Dividir alter_brain.py",
        tipo="split", descripcion="2894 líneas — demasiado grande",
        evidencia=["alter_brain.py tiene 2894 líneas"],
        modulo="brain", impacto="alto", riesgo="alto",
        score=0.70, estado="no_replayable_yet", replayable=False,
    )
    props = engine.generate([hyp_split], repo)
    if not props:
        errors.append("FAIL: hipótesis no_replayable_yet no generó propuesta")

    # Test 3: propuesta tiene pseudo_diff
    if props and props[0].pseudo_diff is None:
        errors.append("FAIL: propuesta de split debería tener pseudo_diff")

    # Test 4: no duplica propuestas existentes
    existing = props[:]
    props2 = engine.generate([hyp_split], repo, existing=existing)
    if props2:
        errors.append("FAIL: no debería generar propuesta duplicada")

    # Test 5: hipótesis de tipo parametro no genera propuesta
    hyp_param = ArchitectureHypothesis(
        id="h_param", titulo="Ajustar MAX_ITEMS",
        tipo="parametro", descripcion="overflow frecuente",
        evidencia=[], modulo="workspace", impacto="bajo", riesgo="bajo",
        score=0.55, estado="propuesta", replayable=True,
        parametro_target="workspace.MAX_ITEMS", rango_experimento=[5,7],
    )
    props3 = engine.generate([hyp_param], repo)
    if props3:
        errors.append("FAIL: hipótesis paramétrica no debería generar CodeProposal")

    # Test 6: round-trip CodeProposal
    if props:
        p = props[0]
        d = p.to_dict()
        p2 = CodeProposal.from_dict(d)
        if p2.id != p.id or p2.titulo != p.titulo:
            errors.append("FAIL: CodeProposal round-trip perdió datos")

    # Test 7: render no explota
    if props:
        rendered = props[0].render()
        if "PROPUESTA" not in rendered:
            errors.append("FAIL: render malformado")

    # Test 8: snapshot_str no explota
    snap = engine.snapshot_str(props)
    if "[PROPUESTAS]" not in snap:
        errors.append("FAIL: snapshot_str malformado")

    return errors


if __name__ == "__main__":
    print("=== alter_code_proposals.py — Tests de invariantes ===\n")
    errors = run_invariant_tests()
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print("\n=== Demo ProposalEngine ===")
    from alter_architecture_hypotheses import ArchitectureHypothesis
    from alter_code_map import CodeMapper

    engine = ProposalEngine()
    repo   = CodeMapper().scan(".")

    hypotheses = [
        ArchitectureHypothesis(
            id="h1", titulo="Dividir alter_brain.py en módulos específicos",
            tipo="split", descripcion="2894 líneas, 84 métodos en AlterBrain",
            evidencia=["alter_brain.py tiene 2894 líneas",
                       "AlterBrain tiene 84 métodos (máx: 40)"],
            modulo="brain", impacto="alto", riesgo="alto",
            score=0.75, estado="no_replayable_yet", replayable=False,
        ),
        ArchitectureHypothesis(
            id="h2", titulo="Extraer lógica de loop_conversacional",
            tipo="refactor", descripcion="loop_conversacional tiene 309 líneas",
            evidencia=["loop_conversacional tiene 309 líneas",
                       "procesar_input tiene 247 líneas"],
            modulo="brain", impacto="medio", riesgo="medio",
            score=0.60, estado="no_replayable_yet", replayable=False,
        ),
    ]

    proposals = engine.generate(hypotheses, repo)
    print(engine.snapshot_str(proposals))
    print()
    for p in proposals:
        print(p.render())
        print()
