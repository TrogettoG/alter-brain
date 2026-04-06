"""
alter_code_map.py — Code Map de AlterB5 (Fase 1)

Mapa read-only del repo real de ALTER.
Usa ast de Python — sin dependencias externas.
Opción A: superficie — clases, funciones, imports, líneas.

Regla fundamental: SOLO LECTURA.
No modifica ningún archivo.

Redis keys:
    alter:b5:code_map  — mapa serializado (comprimido si es grande)
"""

import ast
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class FunctionInfo:
    nombre:    str
    linea:     int
    linea_fin: int
    args:      list
    docstring: str
    es_async:  bool
    lineas:    int   # total de líneas de la función

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassInfo:
    nombre:   str
    linea:    int
    metodos:  list   # list[FunctionInfo]
    atributos: list  # list[str] — detectados en __init__
    docstring: str
    lineas:   int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModuleMap:
    archivo:    str
    imports:    list   # list[str] — módulos importados
    clases:     list   # list[ClassInfo]
    funciones:  list   # list[FunctionInfo] — funciones top-level
    lineas:     int
    scanned_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModuleMap":
        clases = []
        for c in d.get("clases", []):
            metodos = [FunctionInfo(**m) for m in c.get("metodos", [])]
            clases.append(ClassInfo(
                nombre=c["nombre"], linea=c["linea"],
                metodos=metodos, atributos=c.get("atributos", []),
                docstring=c.get("docstring", ""), lineas=c.get("lineas", 0)
            ))
        funciones = [FunctionInfo(**f) for f in d.get("funciones", [])]
        return cls(
            archivo=d["archivo"], imports=d.get("imports", []),
            clases=clases, funciones=funciones,
            lineas=d.get("lineas", 0), scanned_at=d.get("scanned_at", "")
        )

    def get_all_functions(self) -> list:
        """Todas las funciones — top-level + métodos de clases."""
        all_funcs = list(self.funciones)
        for clase in self.clases:
            all_funcs.extend(clase.metodos)
        return all_funcs

    def get_function(self, nombre: str) -> Optional[FunctionInfo]:
        for f in self.get_all_functions():
            if f.nombre == nombre:
                return f
        return None


@dataclass
class RepoMap:
    directorio: str
    modulos:    list   # list[ModuleMap]
    scanned_at: str
    total_lineas: int
    total_funciones: int
    total_clases: int

    def to_dict(self) -> dict:
        return asdict(self)

    def get_module(self, archivo: str) -> Optional[ModuleMap]:
        nombre = Path(archivo).name
        return next(
            (m for m in self.modulos if Path(m.archivo).name == nombre),
            None
        )

    def find_function(self, nombre: str) -> list:
        """Busca una función en todo el repo. Retorna lista de (archivo, FunctionInfo)."""
        results = []
        for modulo in self.modulos:
            f = modulo.get_function(nombre)
            if f:
                results.append((modulo.archivo, f))
        return results

    def get_dependencies(self, archivo: str) -> dict:
        """
        Retorna dependencias de un archivo.
        {
          "imports": [...],         # lo que este módulo importa
          "imported_by": [...],     # quién importa este módulo
        }
        """
        nombre_base = Path(archivo).stem
        modulo = self.get_module(archivo)
        imports_propios = modulo.imports if modulo else []

        imported_by = []
        for m in self.modulos:
            if Path(m.archivo).stem == nombre_base:
                continue
            for imp in m.imports:
                if nombre_base in imp:
                    imported_by.append(m.archivo)
                    break

        return {
            "imports":     imports_propios,
            "imported_by": imported_by,
        }


# ============================================================
# CODE MAPPER
# ============================================================

class CodeMapper:
    """
    Scanner read-only del repo de ALTER.
    Usa ast de Python — sin dependencias externas.
    Solo archivos alter_*.py del directorio dado.
    """

    # Archivos a ignorar
    IGNORAR = {"__pycache__", ".git", "venv", "node_modules"}

    def scan(self, directorio: str) -> RepoMap:
        """
        Escanea todos los alter_*.py en el directorio.
        Retorna RepoMap con el mapa completo.
        """
        path = Path(directorio)
        archivos = sorted([
            f for f in path.glob("alter_*.py")
            if f.name not in self.IGNORAR
        ])

        modulos = []
        for archivo in archivos:
            try:
                mmap = self._scan_file(str(archivo))
                if mmap:
                    modulos.append(mmap)
            except Exception:
                pass

        total_lineas    = sum(m.lineas for m in modulos)
        total_funciones = sum(len(m.get_all_functions()) for m in modulos)
        total_clases    = sum(len(m.clases) for m in modulos)

        return RepoMap(
            directorio      = str(path),
            modulos         = modulos,
            scanned_at      = datetime.now().isoformat(),
            total_lineas    = total_lineas,
            total_funciones = total_funciones,
            total_clases    = total_clases,
        )

    def _scan_file(self, archivo: str) -> Optional[ModuleMap]:
        """Escanea un archivo .py y retorna su ModuleMap."""
        with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

        lineas = source.count("\n") + 1

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ModuleMap(
                archivo=archivo, imports=[], clases=[],
                funciones=[], lineas=lineas,
                scanned_at=datetime.now().isoformat()
            )

        imports   = self._extract_imports(tree)
        clases    = self._extract_classes(tree)
        funciones = self._extract_functions_top(tree)

        return ModuleMap(
            archivo    = archivo,
            imports    = imports,
            clases     = clases,
            funciones  = funciones,
            lineas     = lineas,
            scanned_at = datetime.now().isoformat(),
        )

    def _extract_imports(self, tree: ast.AST) -> list:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return list(set(imports))

    def _extract_classes(self, tree: ast.AST) -> list:
        clases = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metodos    = []
                atributos  = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fi = self._function_info(item)
                        metodos.append(fi)
                        # Extraer atributos de __init__
                        if item.name == "__init__":
                            for stmt in ast.walk(item):
                                if isinstance(stmt, ast.Assign):
                                    for t in stmt.targets:
                                        if isinstance(t, ast.Attribute) and \
                                           isinstance(t.value, ast.Name) and \
                                           t.value.id == "self":
                                            atributos.append(t.attr)

                docstring = ast.get_docstring(node) or ""
                linea_fin = getattr(node, "end_lineno", node.lineno)
                clases.append(ClassInfo(
                    nombre    = node.name,
                    linea     = node.lineno,
                    metodos   = metodos,
                    atributos = list(set(atributos)),
                    docstring = docstring[:150],
                    lineas    = linea_fin - node.lineno + 1,
                ))
        return clases

    def _extract_functions_top(self, tree: ast.AST) -> list:
        """Solo funciones top-level (no dentro de clases)."""
        funciones = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funciones.append(self._function_info(node))
        return funciones

    def _function_info(self, node) -> FunctionInfo:
        args = [
            arg.arg for arg in node.args.args
            if arg.arg != "self"
        ]
        docstring = ast.get_docstring(node) or ""
        linea_fin = getattr(node, "end_lineno", node.lineno)
        return FunctionInfo(
            nombre    = node.name,
            linea     = node.lineno,
            linea_fin = linea_fin,
            args      = args[:8],  # máx 8 args
            docstring = docstring[:120],
            es_async  = isinstance(node, ast.AsyncFunctionDef),
            lineas    = linea_fin - node.lineno + 1,
        )


# ============================================================
# PERSISTENCIA
# ============================================================

def save(repo_map: RepoMap, redis_client) -> bool:
    if not redis_client:
        return False
    try:
        data = json.dumps(repo_map.to_dict(), ensure_ascii=False)
        # Comprimir si es muy grande
        if len(data) > 50_000:
            import zlib, base64
            compressed = base64.b64encode(
                zlib.compress(data.encode())
            ).decode()
            redis_client.set("alter:b5:code_map", f"zlib:{compressed}")
        else:
            redis_client.set("alter:b5:code_map", data)
        return True
    except Exception:
        return False


def load(redis_client) -> Optional[RepoMap]:
    if not redis_client:
        return None
    try:
        raw = redis_client.get("alter:b5:code_map")
        if not raw:
            return None
        if raw.startswith("zlib:"):
            import zlib, base64
            raw = zlib.decompress(base64.b64decode(raw[5:])).decode()
        d = json.loads(raw)
        modulos = [ModuleMap.from_dict(m) for m in d.get("modulos", [])]
        return RepoMap(
            directorio      = d["directorio"],
            modulos         = modulos,
            scanned_at      = d["scanned_at"],
            total_lineas    = d["total_lineas"],
            total_funciones = d["total_funciones"],
            total_clases    = d["total_clases"],
        )
    except Exception:
        return None


# ============================================================
# TESTS DE INVARIANTES
# ============================================================

def run_invariant_tests(directorio: str = ".") -> list[str]:
    errors = []
    mapper = CodeMapper()

    # Test 1: scan no explota en el directorio actual
    try:
        repo = mapper.scan(directorio)
    except Exception as e:
        errors.append(f"FAIL: scan explotó: {e}")
        return errors

    # Test 2: al menos un módulo escaneado
    if not repo.modulos:
        errors.append("FAIL: no se escaneó ningún módulo")
        return errors

    # Test 3: total_lineas > 0
    if repo.total_lineas == 0:
        errors.append("FAIL: total_lineas es 0")

    # Test 4: get_module funciona
    if repo.modulos:
        primer = repo.modulos[0]
        found = repo.get_module(primer.archivo)
        if found is None:
            errors.append("FAIL: get_module no encontró módulo existente")

    # Test 5: get_dependencies no explota
    if repo.modulos:
        deps = repo.get_dependencies(repo.modulos[0].archivo)
        if "imports" not in deps or "imported_by" not in deps:
            errors.append("FAIL: get_dependencies estructura incorrecta")

    # Test 6: find_function retorna lista
    results = repo.find_function("run_invariant_tests")
    if not isinstance(results, list):
        errors.append("FAIL: find_function no retornó lista")

    # Test 7: FunctionInfo.lineas > 0 para funciones reales
    for m in repo.modulos:
        for f in m.get_all_functions():
            if f.lineas <= 0:
                errors.append(
                    f"FAIL: función '{f.nombre}' en {m.archivo} "
                    f"tiene lineas={f.lineas}"
                )
                break

    # Test 8: ModuleMap round-trip
    if repo.modulos:
        m = repo.modulos[0]
        d = m.to_dict()
        m2 = ModuleMap.from_dict(d)
        if m2.archivo != m.archivo:
            errors.append("FAIL: ModuleMap round-trip perdió archivo")
        if len(m2.clases) != len(m.clases):
            errors.append("FAIL: ModuleMap round-trip perdió clases")

    return errors


if __name__ == "__main__":
    import sys
    directorio = sys.argv[1] if len(sys.argv) > 1 else "."

    print("=== alter_code_map.py — Tests de invariantes ===\n")
    errors = run_invariant_tests(directorio)
    if errors:
        for e in errors:
            print(e)
    else:
        print("Todos los invariantes pasan.")

    print(f"\n=== Mapa del repo ({directorio}) ===")
    mapper = CodeMapper()
    repo   = mapper.scan(directorio)
    print(f"Módulos escaneados: {len(repo.modulos)}")
    print(f"Total líneas:       {repo.total_lineas:,}")
    print(f"Total funciones:    {repo.total_funciones}")
    print(f"Total clases:       {repo.total_clases}")
    print()
    for m in sorted(repo.modulos, key=lambda x: x.lineas, reverse=True)[:8]:
        nombre = Path(m.archivo).name
        print(f"  {nombre:<35} {m.lineas:>5} líneas  "
              f"{len(m.clases)} clases  "
              f"{len(m.get_all_functions())} funciones")
