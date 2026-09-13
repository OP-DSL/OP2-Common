"""
Flang/JSON-tree equivalents of the fparser2 AST mutations in
`fortran.translator.kernels` that `fortran/schemes.py` applies
to kernel entities before kernel-to-C++ translation.

fparser2 helper    flang_kernels equivalent
---------------    ------------------------
renameConsts       rename_consts
fixHydraIO         fix_hydra_io
insertAtomicIncs   insert_atomic_incs
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set

import op as OP
from fortran.flang_validator import is_ref, map_param
from op import OpError
from store import Function


# Generic JSON-tree walk helpers

def _resync_flat_views(body: Dict[str, Any]) -> None:
    """
    Rebuild `assignments` and `calls` from `stmts`.

    The scanner emits those flat lists in parallel with the statement tree.
    Kernel rewrites only mutate `stmts`, so analysis that still reads the
    flat views (`resolveParamAccesses`, the validator on a rewritten copy)
    must see the same tree.
    """
    assignments: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []

    def walk(stmts: List[Dict[str, Any]]) -> None:
        for stmt in stmts:
            kind = stmt.get("kind")
            if kind == "assign":
                assignments.append({
                    "line": stmt.get("line", 0),
                    "lhs": stmt["lhs"],
                    "rhs": stmt["rhs"],
                })
            elif kind == "call":
                calls.append({
                    "line": stmt.get("line", 0),
                    "name": stmt.get("name"),
                    "args": stmt.get("args", []),
                })
            elif kind == "if_stmt":
                walk([stmt["stmt"]])
            elif kind == "if_construct":
                for branch in stmt.get("branches", []):
                    walk(branch.get("body", []))
            elif kind == "do":
                walk(stmt.get("body", []))

    walk(body.get("stmts", []))
    body["assignments"] = assignments
    body["calls"] = calls


def _walk_stmt_bodies(
    stmts: List[Dict[str, Any]],
    visit_body: Callable[[List[Dict[str, Any]]], bool],
) -> bool:
    """
    Call `visit_body` on `stmts` and on every nested body reachable through
    if/do control flow. Returns True if `visit_body` returned True for any
    of them.
    """
    modified = visit_body(stmts)

    for stmt in stmts:
        kind = stmt.get("kind")
        if kind == "if_stmt":
            inner = [stmt["stmt"]]
            if _walk_stmt_bodies(inner, visit_body):
                stmt["stmt"] = inner[0]
                modified = True
        elif kind == "if_construct":
            for branch in stmt.get("branches", []):
                if _walk_stmt_bodies(branch.get("body", []), visit_body):
                    modified = True
        elif kind == "do":
            if _walk_stmt_bodies(stmt.get("body", []), visit_body):
                modified = True

    return modified


# rename_consts

def _rename_idents(node: Any, targets: Set[str], replacement: Callable[[str], str]) -> None:
    """
    Recursively rename every `name` / `part_ref` / `funcref` identifier
    occurrence (and `do` loop control variables) within a decls/stmts
    JSON subtree whose identifier is in `targets`.
    """
    if isinstance(node, list):
        for item in node:
            _rename_idents(item, targets, replacement)
        return

    if not isinstance(node, dict):
        return

    kind = node.get("kind")
    if kind == "name" and node.get("value") in targets:
        node["value"] = replacement(node["value"])
    elif kind in ("part_ref", "funcref") and node.get("name") in targets:
        node["name"] = replacement(node["name"])
    elif kind == "do" and node.get("var") in targets:
        node["var"] = replacement(node["var"])

    for value in node.values():
        _rename_idents(value, targets, replacement)


def rename_consts(entities: List[Function], const_ptrs: Set[str], replacement: Callable[[str], str]) -> None:
    """
    Flang equivalent of fortran.translator.kernels.renameConsts.
    """
    for entity in entities:
        body = getattr(entity, "flang_body", None)
        if body is None:
            continue

        targets = set(const_ptrs) - set(entity.parameters)
        if not targets:
            continue

        _rename_idents(body.get("decls", []), targets, replacement)
        _rename_idents(body.get("stmts", []), targets, replacement)
        for local in body.get("locals", []):
            dims = local.get("dims")
            if isinstance(dims, list):
                local["dims"] = [replacement(d) if d in targets else d for d in dims]
        _resync_flat_views(body)


# fix_hydra_io

_HYDRA_CLOBBER_CALLS = {
    "hyd_print", "hyd_dump", "hyd_kill", "hyd_error_print", "hyd_error_dump",
}
_HYDRA_CLOBBER_CALLS |= {f"{name}_hydrales" for name in list(_HYDRA_CLOBBER_CALLS)}


def _is_hydra_call(stmt: Dict[str, Any]) -> bool:
    return stmt.get("kind") == "call" and str(stmt.get("name", "")).lower() in _HYDRA_CLOBBER_CALLS


def _replace_hydra_calls(stmts: List[Dict[str, Any]]) -> bool:
    modified = False
    for i, stmt in enumerate(stmts):
        if _is_hydra_call(stmt):
            stmts[i] = {"kind": "stop"}
            modified = True
        elif stmt.get("kind") == "if_stmt" and _is_hydra_call(stmt["stmt"]):
            stmt["stmt"] = {"kind": "stop"}
            modified = True
    return modified


def fix_hydra_io(entities: List[Function]) -> None:
    """
    Flang equivalent of fortran.translator.kernels.fixHydraIO.

    WRITE statements need no rewrite here as flang_kernels_c.py's statement
    translator already turns every "write" node into a no-op comment.
    """
    for entity in entities:
        body = getattr(entity, "flang_body", None)
        if body is None:
            continue
        _walk_stmt_bodies(body.get("stmts", []), _replace_hydra_calls)
        _resync_flat_views(body)


# insert_atomic_incs

def _is_add_or_sub_expr(expr: Dict[str, Any]) -> bool:
    """
    True if `expr` is a `+`/`-` binary expression.

    Parentheses are unwrapped so `param = (param + x)` still counts.
    """
    expr = _unwrap_parens(expr)
    return expr.get("kind") == "binary" and expr.get("op") in ("+", "-")


def _replace_increments(stmts: List[Dict[str, Any]], param: str, typ: OP.Type) -> bool:
    modified = False
    for i, stmt in enumerate(stmts):
        if stmt.get("kind") != "assign" or not is_ref(stmt["lhs"], param):
            continue
        if not _is_add_or_sub_expr(stmt["rhs"]):
            raise OpError(f"Error: unexpected statement while inserting atomics")
        amount = _substitute_ref_with_zero(stmt["rhs"], param, typ)
        stmts[i] = {
            "kind": "call",
            "line": stmt.get("line", 0),
            "name": "atomicAdd",
            "args": [stmt["lhs"], amount],
        }
        modified = True
    return modified


def _substitute_ref_with_zero(expr: Dict[str, Any], ref_name: str, typ: OP.Type) -> Dict[str, Any]:
    """
    Replace every whole occurrence of `ref_name` inside `expr` with a zero
    literal of the appropriate type, preserving the rest of the tree shape.
    """
    if is_ref(expr, ref_name):
        return _zero_literal(typ)

    kind = expr.get("kind")
    if kind == "binary":
        return {
            **expr,
            "left": _substitute_ref_with_zero(expr["left"], ref_name, typ),
            "right": _substitute_ref_with_zero(expr["right"], ref_name, typ),
        }
    if kind in ("paren", "unary"):
        return {**expr, "expr": _substitute_ref_with_zero(expr["expr"], ref_name, typ)}

    return expr


def _unwrap_parens(expr: Dict[str, Any]) -> Dict[str, Any]:
    while expr.get("kind") == "paren":
        expr = expr["expr"]
    return expr


def _zero_literal(typ: OP.Type) -> Dict[str, Any]:
    if isinstance(typ, OP.Int):
        return {"kind": "int_lit", "text": "0", "kind_text": None}
    elif isinstance(typ, OP.Float) and typ.size == 32:
        return {"kind": "real_lit", "text": "0.0", "kind_text": None}
    elif isinstance(typ, OP.Float) and typ.size == 64:
        return {"kind": "real_lit", "text": "0.0", "kind_text": "8"}
    else:
        raise OpError(f"Error: unexpected arg type while inserting atomics: {typ}")


def insert_atomic_incs(entities: List[Function], loop: OP.Loop, match: Callable[[OP.Arg], bool]) -> None:
    """
    Rewrites `param = param +/- expr` into `call atomicAdd(param, expr)` for
    every dummy parameter that a matching OP_INC loop argument maps onto.
    """
    if len(entities) == 0:
        return

    modified: Dict[str, Set[int]] = {}

    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        if isinstance(loop.args[arg_idx], OP.ArgDat):
            typ = loop.dat(loop.args[arg_idx]).typ
        elif hasattr(loop.args[arg_idx], "typ"):
            typ = loop.args[arg_idx].typ
        else:
            raise OpError(f"Error: could not find type of arg while inserting atomics: {loop.args[arg_idx]}")

        def op(entity2: Function, param_idx2: int, typ2: OP.Type) -> bool:
            if param_idx2 in modified.get(entity2.name, set()):
                return False

            body = getattr(entity2, "flang_body", None)
            if body is None:
                return False

            param_name = entity2.parameters[param_idx2]
            if _walk_stmt_bodies(body.get("stmts", []), lambda stmts: _replace_increments(stmts, param_name, typ2)):
                modified.setdefault(entity2.name, set()).add(param_idx2)
                _resync_flat_views(body)

            return False

        map_param(entities[0], arg_idx, entities, op, typ)
