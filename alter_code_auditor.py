"""
alter_code_auditor.py — Code Auditor de AlterB5 (Fase 1)

Compara Architecture State (spec) vs Code Map (realidad).
Detecta gaps, deuda técnica, acoplamiento y duplicación.

Qué detecta:
    gap_implementacion   — módulo en spec pero no en código
    funcion_larga        — función con demasiadas líneas
    sin_docstring        — clase o función importante sin docstring
    acoplamiento_alto    — módulo con demasiados imports
    modulo_grande        — archivo con demasiadas líneas
    parametro_drift      — parámetro en spec no encontrado en código
    clase_grande         — clase con demasiados métodos

Solo lectura. No modifica nada.

Redis keys:
    alter:b5:observations         — últimas observaciones
    alter:b5:observations:history — historial (últimas 20 auditorías)
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
class CodeObservation:
    tipo:        str    # tipo de observación
    severidad:   str    # "info" | "warning" | "critical"
    archivo:     str
    linea:       int
    descripcion: str
    sugerencia:  str
    modulo:      str = ""  # nombre del módulo en la spec

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CodeAuditReport:
    timestamp:     str
    observaciones: list   # list[CodeObservation]
    resumen:       str
    score_calidad: float  # 0..1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CodeAuditReport":
        obs = [CodeObservation(**o) for o in d.get("observaciones", [])]
        return cls(
            timestamp     = d["timestamp"],
            observaciones = obs,
            resumen       = d.get("resumen", ""),
            score_calidad = d.get("score_calidad", 0.5),
        )


# ============================================================
# CODE AUDITOR
# ============================================================

class CodeAuditor:
    """
    Compara Architecture State vs Code Map.
    Genera observaciones sobre gaps y deuda técnica.
    Solo lectura — no modifica nada.
    """

    MAX_LINEAS_FUNCION  = 150   # líneas máximas antes de warning
    MAX_LINEAS_MODULO   = 2000  # líneas máximas antes de warning
    MAX_IMPORTS_MODULO  = 20    # imports máximos antes de warning acoplamiento
    MAX_METODOS_CLASE   = 40    # métodos máximos antes de warning

    def audit(
        self,
        arch_state,    # ArchitectureState
        repo_map,      # RepoMap
    ) -> CodeAuditReport:
        """
        Corre la auditoría completa.
        Retorna CodeAuditReport con todas las observaciones.
        """
        obs = []

        obs += self._check_gaps(arch_state, repo_map)
        obs += self._check_debt(repo_map)
        obs += self._check_coupling(repo_map)
        obs += self._check_size(repo_map)
        obs += self._check_param_drift(arch_state, repo_map)

        # Score de calidad (inverso a densidad de warnings/críticos)
        criticos = sum(1 for o in obs if o.severidad == "critical")
        warnings = sum(1 for o in obs if o.severidad == "warning")
        score    = float(max(0.0, 1.0 - criticos * 0.10 - warnings * 0.03))

        resumen = self._generate_summary(obs, score)

        return CodeAuditReport(
            timestamp     = datetime.now().isoformat(),
            observaciones = obs,
            resumen       = resumen,
            score_calidad = score,
        )

    # ----------------------------------------------------------
    # CHECKS
    # ----------------------------------------------------------

    def _check_gaps(self, arch_state, repo_map) -> list:
        """Detecta módulos en spec que no tienen código real."""
        obs = []
        archivos_repo = {
            Path(m.archivo).name
            for m in repo_map.modulos
        }
        for mod_spec in arch_state.modules:
            archivo_nombre = Path(mod_spec.archivo).name
            if archivo_nombre not in archivos_repo:
                obs.append(CodeObservation(
                    tipo        = "gap_implementacion",
                    severidad   = "warning",
                    archivo     = mod_spec.archivo,
                    linea       = 0,
                    descripcion = f"Módulo '{mod_spec.nombre}' en spec pero sin archivo en repo",
                    sugerencia  = f"Crear {mod_spec.archivo} o actualizar la spec",
                    modulo      = mod_spec.nombre,
                ))
        return obs

    def _check_debt(self, repo_map) -> list:
        """Detecta funciones largas y falta de docstrings."""
        obs = []
        for modulo in repo_map.modulos:
            nombre_archivo = Path(modulo.archivo).name

            for func in modulo.get_all_functions():
                # Función muy larga
                if func.lineas > self.MAX_LINEAS_FUNCION:
                    severidad = "critical" if func.lineas > 150 else "warning"
                    obs.append(CodeObservation(
                        tipo        = "funcion_larga",
                        severidad   = severidad,
                        archivo     = nombre_archivo,
                        linea       = func.linea,
                        descripcion = (
                            f"'{func.nombre}' tiene {func.lineas} líneas "
                            f"(máx: {self.MAX_LINEAS_FUNCION})"
                        ),
                        sugerencia  = "Dividir en funciones más pequeñas",
                    ))

                # Sin docstring (solo para funciones públicas de más de 10 líneas)
                if (not func.docstring and
                        not func.nombre.startswith("_") and
                        func.lineas > 10):
                    obs.append(CodeObservation(
                        tipo        = "sin_docstring",
                        severidad   = "info",
                        archivo     = nombre_archivo,
                        linea       = func.linea,
                        descripcion = f"'{func.nombre}' sin docstring ({func.lineas} líneas)",
                        sugerencia  = "Agregar docstring con propósito y parámetros clave",
                    ))

            # Clases con demasiados métodos
            for clase in modulo.clases:
                if len(clase.metodos) > self.MAX_METODOS_CLASE:
                    obs.append(CodeObservation(
                        tipo        = "clase_grande",
                        severidad   = "warning",
                        archivo     = nombre_archivo,
                        linea       = clase.linea,
                        descripcion = (
                            f"Clase '{clase.nombre}' tiene {len(clase.metodos)} métodos "
                            f"(máx: {self.MAX_METODOS_CLASE})"
                        ),
                        sugerencia  = "Considerar separar en clases más específicas",
                    ))

        return obs

    def _check_coupling(self, repo_map) -> list:
        """Detecta módulos con acoplamiento alto (demasiados imports)."""
        obs = []
        for modulo in repo_map.modulos:
            # Filtrar imports internos de alter_*
            alter_imports = [
                i for i in modulo.imports
                if "alter_" in i.lower()
            ]
            if len(alter_imports) > self.MAX_IMPORTS_MODULO:
                obs.append(CodeObservation(
                    tipo        = "acoplamiento_alto",
                    severidad   = "warning",
                    archivo     = Path(modulo.archivo).name,
                    linea       = 1,
                    descripcion = (
                        f"'{Path(modulo.archivo).name}' importa "
                        f"{len(alter_imports)} módulos internos "
                        f"(máx: {self.MAX_IMPORTS_MODULO})"
                    ),
                    sugerencia  = "Revisar dependencias — posible God Object",
                ))
        return obs

    def _check_size(self, repo_map) -> list:
        """Detecta módulos demasiado grandes."""
        obs = []
        for modulo in repo_map.modulos:
            if modulo.lineas > self.MAX_LINEAS_MODULO:
                severidad = "critical" if modulo.lineas > 3000 else "warning"
                obs.append(CodeObservation(
                    tipo        = "modulo_grande",
                    severidad   = severidad,
                    archivo     = Path(modulo.archivo).name,
                    linea       = 0,
                    descripcion = (
                        f"'{Path(modulo.archivo).name}' tiene {modulo.lineas} líneas "
                        f"(máx recomendado: {self.MAX_LINEAS_MODULO})"
                    ),
                    sugerencia  = "Considerar dividir en módulos más específicos",
                ))
        return obs

    def _check_param_drift(self, arch_state, repo_map) -> list:
        """
        Detecta parámetros definidos en spec que no aparecen en el código.
        Búsqueda simple por nombre — no semántica.
        """
        obs = []
        for mod_spec in arch_state.modules:
            if not mod_spec.parametros:
                continue
            modulo_map = repo_map.get_module(mod_spec.archivo)
            if not modulo_map:
                continue

            # Texto completo de funciones del módulo para búsqueda
            funciones = modulo_map.get_all_functions()
            nombres_funcs = {f.nombre for f in funciones}

            for param in mod_spec.parametros:
                # Si el nombre del parámetro no aparece en ninguna función
                # del módulo — posible drift
                nombre_param = param.nombre
                encontrado = any(
                    nombre_param in f.nombre or nombre_param in f.docstring
                    for f in funciones
                )
                # También buscar en nombres de clases
                encontrado = encontrado or any(
                    nombre_param in c.nombre or nombre_param in c.docstring
                    for c in modulo_map.clases
                )
                if not encontrado and len(nombre_param) > 4:
                    obs.append(CodeObservation(
                        tipo        = "parametro_drift",
                        severidad   = "info",
                        archivo     = Path(mod_spec.archivo).name,
                        linea       = 0,
                        descripcion = (
                            f"Parámetro '{nombre_param}' en spec de "
                            f"'{mod_spec.nombre}' no encontrado en código"
                        ),
                        sugerencia  = "Verificar si el parámetro está implementado "
                                      "o si la spec está desactualizada",
                        modulo      = mod_spec.nombre,
                    ))
        return obs

    # ----------------------------------------------------------
    # RESUMEN Y PERSISTENCIA
    # ----------------------------------------------------------

    def _generate_summary(self, obs: list, score: float) -> str:
        criticos = [o for o in obs if o.severidad == "critical"]
        warnings = [o for o in obs if o.severidad == "warning"]
        infos    = [o for o in obs if o.severidad == "info"]

        estado = "limpio" if score > 0.90 else \
                 "bueno"  if score > 0.75 else \
                 "con deuda" if score > 0.50 else "degradado"

        partes = [
            f"Código en estado {estado} (calidad:{score:.2f}).",
            f"{len(criticos)} críticos | {len(warnings)} warnings | {len(infos)} info.",
        ]
        if criticos:
            top = criticos[0]
            partes.append(f"Crítico principal: {top.descripcion[:60]}")
        elif warnings:
            top = warnings[0]
            partes.append(f"Warning principal: {top.descripcion[:60]}")

        return " ".join(partes)

    def save(self, report: CodeAuditReport, redis_client) -> bool:
        if not redis_client:
            return False
        try:
            data = json.dumps(report.to_dict(), ensure_ascii=False)
            redis_client.set("alter:b5:observations", data)
            redis_client.lpush("alter:b5:observations:history", data)
            redis_client.ltrim("alter:b5:observations:history", 0, 19)
            return True
        except Exception:
            return False

    def report_str(self, report: CodeAuditReport) -> str:
        """Formato legible para Telegram y KAIROS."""
        criticos = [o for o in report.observaciones if o.severidad == "critical"]
        warnings = [o for o in report.observaciones if o.severidad == "warning"]

        lines = [
            f"🔬 AUDITORÍA DE CÓDIGO — {report.timestamp[:16]}",
            f"Calidad: {report.score_calidad:.0%}  {report.resumen}",
        ]
        for o in criticos[:3]:
            lines.append(f"🔴 [{o.tipo}] {o.archivo}: {o.descripcion[:60]}")
        for o in warnings[:3]:
            lines.append(f"🟡 [{o.tipo}] {o.archivo}: {o.descripcion[:60]}")
        if len(criticos) + len(warnings) > 6:
            resto = len(criticos) + len(warnings) - 6
            lines.append(f"... y {resto} observaciones más")
        return "\n".join(lines)


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests(directorio: str = ".") -> list[str]:
    errors = []

    from alter_architecture_state import build_current_spec
    from alter_code_map import CodeMapper

    spec   = build_current_spec()
    mapper = CodeMapper()
    repo   = mapper.scan(directorio)
    auditor = CodeAuditor()

    # Test 1: audit no explota
    try:
        report = auditor.audit(spec, repo)
    except Exception as e:
        errors.append(f"FAIL: audit explotó: {e}")
        return errors

    # Test 2: score entre 0 y 1
    if not (0.0 <= report.score_calidad <= 1.0):
        errors.append(f"FAIL: score_calidad fuera de rango: {report.score_calidad}")

    # Test 3: observaciones es lista
    if not isinstance(report.observaciones, list):
        errors.append("FAIL: observaciones no es lista")

    # Test 4: severidades válidas
    validas = {"info", "warning", "critical"}
    for o in report.observaciones:
        if o.severidad not in validas:
            errors.append(f"FAIL: severidad inválida: {o.severidad}")

    # Test 5: alter_brain.py detectado como módulo grande
    modulo_grande = [
        o for o in report.observaciones
        if o.tipo == "modulo_grande" and "alter_brain" in o.archivo
    ]
    if not modulo_grande:
        errors.append("FAIL: alter_brain.py debería detectarse como módulo grande")

    # Test 6: round-trip CodeAuditReport
    d = report.to_dict()
    report2 = CodeAuditReport.from_dict(d)
    if len(report2.observaciones) != len(report.observaciones):
        errors.append("FAIL: CodeAuditReport round-trip perdió observaciones")

    # Test 7: report_str no explota
    snap = auditor.report_str(report)
    if "AUDITORÍA DE CÓDIGO" not in snap:
        errors.append("FAIL: report_str malformado")

    return errors


if __name__ == "__main__":
    import sys
    directorio = sys.argv[1] if len(sys.argv) > 1 else "."

    print("=== alter_code_auditor.py — Tests de invariantes ===\n")
    errors = run_invariant_tests(directorio)
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print(f"\n=== Auditoría de código ({directorio}) ===")
    from alter_architecture_state import build_current_spec
    from alter_code_map import CodeMapper

    spec    = build_current_spec()
    mapper  = CodeMapper()
    repo    = mapper.scan(directorio)
    auditor = CodeAuditor()
    report  = auditor.audit(spec, repo)

    print(auditor.report_str(report))
    print(f"\nObservaciones totales: {len(report.observaciones)}")

    # Top funciones más largas
    funciones_largas = [
        o for o in report.observaciones if o.tipo == "funcion_larga"
    ]
    if funciones_largas:
        print(f"\nFunciones más largas ({len(funciones_largas)}):")
        for o in sorted(funciones_largas,
                        key=lambda x: int(x.descripcion.split()[2])
                        if len(x.descripcion.split()) > 2 else 0,
                        reverse=True)[:5]:
            print(f"  {o.archivo}:{o.linea} — {o.descripcion[:70]}")
