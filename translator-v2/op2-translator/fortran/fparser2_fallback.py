from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Set

from fparser.common.readfortran import FortranStringReader

import fortran.parser
from store import Function, Program

if TYPE_CHECKING:
    from fortran import Fortran


def ensure_fparser2_ast(
    lang: "Fortran",
    program: Program,
    include_dirs: Set[Path],
    defines: List[str],
) -> bool:
    """
    Parse `program.source` with fparser2 if `program.ast` is missing.

    Returns True when an AST is available afterwards, False if fparser2
    could not parse the file.
    """
    del defines  # preprocessing already applied; kept for API symmetry

    if program.ast is not None:
        return True

    try:
        reader = FortranStringReader(program.source, include_dirs=list(include_dirs))
        ast = lang.parser(reader)
    except Exception as err:
        print(
            f"Warning: fparser2 fallback parse failed for {program.path}: {err}",
            file=sys.stderr,
        )
        return False

    program.ast = ast
    _attach_entity_asts(program, ast)
    return True


def _attach_entity_asts(program: Program, ast) -> None:
    """
    Walk an fparser2 AST and copy subprogram nodes onto Flang-built
    entities with the same name.
    """
    temp = fortran.parser.parseProgram(ast, program.source, program.path)
    ast_by_name = {
        entity.name.lower(): entity.ast
        for entity in temp.entities
        if isinstance(entity, Function)
    }

    for entity in program.entities:
        if not isinstance(entity, Function):
            continue
        node = ast_by_name.get(entity.name.lower())
        if node is not None:
            entity.ast = node

    # if Flang missed a subprogram that fparser2 found, add it
    # so downstream code can still resolve dependencies
    existing = {entity.name.lower() for entity in program.entities}
    for entity in temp.entities:
        if not isinstance(entity, Function):
            continue
        if entity.name.lower() in existing:
            continue
        program.entities.append(entity)
