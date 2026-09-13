"""
Semantic validation for kernels parsed with LLVM Flang (--parser flang).
"""

from __future__ import annotations

from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import fortran.translator.kernels as ftk
import op as OP
from fortran.validator import printViolations
from op import OpError
from store import Application, Function, Program


def _flang_body(func: Function) -> Dict[str, Any]:
    """
    Return `func`'s `flang_body` dictionary, raise `OpError` if it's missing.
    """
    body = getattr(func, "flang_body", None)
    if body is None:
        raise OpError(f"missing Flang body data for {func.name}")
    return body


def is_ref(expr: Dict[str, Any], name: str) -> bool:
    """
    Checks if `expr` is a `name` / `part_ref` / `funcref` that refers to `name`.
    """
    kind = expr.get("kind")
    if kind == "name":
        return expr.get("value") == name
    if kind in ("part_ref", "funcref"):
        return expr.get("name") == name
    return False


def iter_param_occurrences(expr: Dict[str, Any], name: str) -> Iterator[Dict[str, Any]]:
    """
    Recursively yield every occurrence of `name` anywhere within `expr`.
    """
    kind = expr.get("kind")

    if kind == "name":
        if expr.get("value") == name:
            yield expr
        return

    if kind == "part_ref":
        if expr.get("name") == name:
            yield expr
        for sub in expr.get("subscripts", []):
            if sub.get("kind") == "triplet":
                for part in ("lower", "upper", "stride"):
                    if sub.get(part) is not None:
                        yield from iter_param_occurrences(sub[part], name)
            else:
                yield from iter_param_occurrences(sub, name)
        return

    if kind == "funcref":
        if expr.get("name") == name:
            yield expr
        for arg in expr.get("args", []):
            yield from iter_param_occurrences(arg, name)
        return

    if kind == "binary":
        yield from iter_param_occurrences(expr["left"], name)
        yield from iter_param_occurrences(expr["right"], name)
        return

    if kind in ("paren", "unary"):
        yield from iter_param_occurrences(expr["expr"], name)
        return


def iter_leaf_names(expr: Dict[str, Any]) -> Iterator[str]:
    """
    Yield every identifier referenced anywhere within `expr`.
    """
    kind = expr.get("kind")

    if kind == "name":
        yield expr["value"]
    elif kind == "part_ref":
        yield expr["name"]
        for sub in expr.get("subscripts", []):
            if sub.get("kind") == "triplet":
                for part in ("lower", "upper", "stride"):
                    if sub.get(part) is not None:
                        yield from iter_leaf_names(sub[part])
            else:
                yield from iter_leaf_names(sub)
    elif kind == "funcref":
        yield expr["name"]
        for arg in expr.get("args", []):
            yield from iter_leaf_names(arg)
    elif kind == "binary":
        yield from iter_leaf_names(expr["left"])
        yield from iter_leaf_names(expr["right"])
    elif kind in ("paren", "unary"):
        yield from iter_leaf_names(expr["expr"])


def iter_all_names(func: Function) -> Iterator[str]:
    """
    Yield every identifier referenced in `func`'s assignments, calls, and
    local array dimension expressions.
    """
    body = _flang_body(func)
    for assign in body["assignments"]:
        yield from iter_leaf_names(assign["lhs"])
        yield from iter_leaf_names(assign["rhs"])
    for call in body["calls"]:
        for arg in call.get("args", []):
            yield from iter_leaf_names(arg)
    for local in body.get("locals", []):
        yield from local.get("dims", [])


# Call-graph propagation

def _find_calls_for_param(
    func: Function,
    param: str,
    funcs_by_name: Dict[str, Function],
) -> List[Tuple[Function, int]]:
    """
    Find every callee reachable by `param` flowing into a call to a known function, at any
    nesting depth within `func`'s body.

    Returns a list of (callee, callee_param_idx) pairs.
    """
    results: List[Tuple[Function, int]] = []

    def record(
        container: Optional[Dict[str, Any]],
        is_actual_arg: bool,
        occurrence: Dict[str, Any],
    ) -> None:
        if container is None:
            return
        callee = funcs_by_name.get(container.get("name"))
        if callee is None:
            return
        items = container.get("args") if is_actual_arg else container.get("subscripts")
        if not items:
            return
        idx = next((i for i, item in enumerate(items) if item is occurrence), None)
        if idx is None or idx >= len(callee.parameters):
            return
        results.append((callee, idx))

    def visit(expr: Dict[str, Any]) -> None:
        kind = expr.get("kind")

        if kind == "name":
            return

        if kind == "part_ref":
            for sub in expr.get("subscripts", []):
                if sub.get("kind") == "triplet":
                    for part in ("lower", "upper", "stride"):
                        if sub.get(part) is not None:
                            visit(sub[part])
                    continue
                if is_ref(sub, param):
                    record(expr, False, sub)
                visit(sub)
            return

        if kind == "funcref":
            for arg in expr.get("args", []):
                if is_ref(arg, param):
                    record(expr, True, arg)
                visit(arg)
            return

        if kind == "binary":
            visit(expr["left"]); visit(expr["right"]); return
        if kind in ("paren", "unary"):
            visit(expr["expr"]); return

    body = _flang_body(func)
    for assign in body["assignments"]:
        visit(assign["lhs"])
        visit(assign["rhs"])
    for call in body["calls"]:
        call_container = {"name": call.get("name"), "args": call.get("args", [])}
        for arg in call.get("args", []):
            if is_ref(arg, param):
                record(call_container, True, arg)
            visit(arg)

    return results


def find_called(
    func: Function,
    param_idx: int,
    funcs_by_name: Dict[str, Function],
) -> List[Tuple[Function, int]]:
    """
    Equivalent to fortran.util's findCalled / findCalled2.
    """
    checked: Dict[str, Set[int]] = {}
    stack: List[Tuple[Function, int]] = [(func, param_idx)]

    while stack:
        cur_func, cur_idx = stack.pop()

        if cur_func.name not in checked:
            checked[cur_func.name] = set()
        elif cur_idx in checked[cur_func.name]:
            continue

        checked[cur_func.name].add(cur_idx)

        if cur_idx >= len(cur_func.parameters):
            continue

        param = cur_func.parameters[cur_idx]
        for callee, callee_idx in _find_calls_for_param(cur_func, param, funcs_by_name):
            if callee.name in checked and callee_idx in checked[callee.name]:
                continue
            stack.append((callee, callee_idx))

    called_list = []
    for name, indices in checked.items():
        f = funcs_by_name.get(name)
        if f is None:
            continue
        for idx in indices:
            called_list.append((f, idx))

    return called_list


def map_param(
    func: Function,
    param_idx: int,
    funcs: List[Function],
    op: Callable[..., Any],
    *args,
) -> None:
    """
    Equivalent to fortran.util's mapParam.
    """
    funcs_by_name = {f.name: f for f in funcs}
    for f2, idx2 in find_called(func, param_idx, funcs_by_name):
        stop = op(f2, idx2, *args)
        if stop:
            break


# Individual check helpers

def _is_safe_call_arg(
    occurrence: Dict[str, Any], body: Dict[str, Any], known_names: Set[str],
) -> bool:
    """
    Returns True if `occurrence` is a direct argument of a funcref/call, or a
    subscript of a `part_ref`, whose base name resolves to a known function.
    """

    def search(expr: Dict[str, Any]) -> bool:
        kind = expr.get("kind")

        if kind == "part_ref":
            for sub in expr.get("subscripts", []):
                if sub.get("kind") == "triplet":
                    for part in ("lower", "upper", "stride"):
                        if sub.get(part) is not None and search(sub[part]):
                            return True
                    continue
                if sub is occurrence:
                    return expr.get("name") in known_names
                if search(sub):
                    return True
            return False

        if kind == "funcref":
            for arg in expr.get("args", []):
                if arg is occurrence:
                    return expr.get("name") in known_names
                if search(arg):
                    return True
            return False

        if kind == "binary":
            return search(expr["left"]) or search(expr["right"])

        if kind in ("paren", "unary"):
            return search(expr["expr"])

        return False

    for assign in body["assignments"]:
        if search(assign["lhs"]) or search(assign["rhs"]):
            return True
    for call in body["calls"]:
        for arg in call.get("args", []):
            if arg is occurrence:
                return call.get("name") in known_names
            if search(arg):
                return True
    return False


def _linearize_increment(node: Dict[str, Any], ref_name: str, count: List[int]) -> str:

    def inc_sym() -> str:
        count[0] += 1
        return f"y{count[0]}"

    kind = node.get("kind")

    if kind == "paren":
        return f"({_linearize_increment(node['expr'], ref_name, count)})"

    if kind == "binary" and node.get("op") in ("+", "-"):
        left, right = node["left"], node["right"]
        if any(True for _ in iter_param_occurrences(left, ref_name)):
            return f"{_linearize_increment(left, ref_name, count)} {node['op']} {inc_sym()}"
        if any(True for _ in iter_param_occurrences(right, ref_name)):
            return f"{inc_sym()} {node['op']} {_linearize_increment(right, ref_name, count)}"
        raise OpError("")

    if kind == "name":
        return "x" if node.get("value") == ref_name else inc_sym()

    if kind in ("part_ref", "funcref"):
        return "x" if node.get("name") == ref_name else inc_sym()

    if any(True for _ in iter_param_occurrences(node, ref_name)):
        raise OpError("")

    return inc_sym()


# Individual checks (equivalent to fortran/validator.py)

def _param_has_array_shape(func: Function, param: str) -> bool:
    """
    True if `param` has an array spec in `flang_body["decls"]`.
    """
    for decl in _flang_body(func).get("decls", []):
        if decl.get("kind") != "type_decl":
            continue
        attr_dim = decl.get("dim")
        for ent in decl.get("entities", []):
            if ent.get("name") != param:
                continue
            return (ent.get("dim") or attr_dim) is not None
    return False


def checkConstRead(func: Function, const_ptrs: List[str], violations: List[str]) -> None:
    """
    Equivalent to fortran.validator's checkConstRead. Appends a violation
    for each assignment whose LHS is one of `const_ptrs`.
    """
    def msg(const_ptr: str, line: int) -> str:
        return f"In {func.name} (const {const_ptr}): {line}"

    for assign in _flang_body(func)["assignments"]:
        for const_ptr in const_ptrs:
            if is_ref(assign["lhs"], const_ptr):
                violations.append(msg(const_ptr, assign["line"]))
                break


def checkInc(func: Function, param_idx: int, funcs: List[Function], violations: List[str]) -> None:
    """
    Equivalent to fortran.validator's checkInc. Appends a violation if
    `func.parameters[param_idx]` is written but not formed as a proper
    increment (`x = x + delta`), or if it appears in a context that is
    not a known function argument or array subscript.
    """
    param = func.parameters[param_idx]
    funcs_by_name = {f.name: f for f in funcs}
    known_names = set(funcs_by_name.keys())
    body = _flang_body(func)

    def msg(s: str) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {param}): {s}"

    lhs_assignments = [a for a in body["assignments"] if is_ref(a["lhs"], param)]

    exempt_ids = set()
    for a in lhs_assignments:
        for occ in iter_param_occurrences(a["rhs"], param):
            exempt_ids.add(id(occ))

    other_occurrences: List[Dict[str, Any]] = []
    for a in body["assignments"]:
        if not is_ref(a["lhs"], param):
            other_occurrences.extend(iter_param_occurrences(a["lhs"], param))
        for occ in iter_param_occurrences(a["rhs"], param):
            if id(occ) not in exempt_ids:
                other_occurrences.append(occ)
    for c in body["calls"]:
        for arg in c.get("args", []):
            other_occurrences.extend(iter_param_occurrences(arg, param))

    for occ in other_occurrences:
        if not _is_safe_call_arg(occ, body, known_names):
            violations.append(msg("invalid context"))

    for a in lhs_assignments:
        rhs_refs = list(iter_param_occurrences(a["rhs"], param))

        if len(rhs_refs) > 1:
            violations.append(msg(f"multi-ref: {a['line']}"))
            continue
        if len(rhs_refs) == 0:
            violations.append(msg(f"no-ref: {a['line']}"))
            continue

        if rhs_refs[0] != a["lhs"]:
            violations.append(msg(f"index mismatch: {a['line']}"))
            continue

        try:
            count = [0]
            expr_str = _linearize_increment(a["rhs"], param, count)
        except OpError:
            violations.append(msg(f"invalid usage: {a['line']}"))
            continue

        if expr_str == "x":
            violations.append(msg(f"no-op: {a['line']}"))
        elif (expr_str.count("x") == 1 and expr_str.startswith("x +")) or expr_str.startswith("x -"):
            continue
        elif simplify(parse_expr(f"{expr_str.replace('x', '0')}", evaluate=False)) == 0:
            violations.append(msg(f"no-op: {a['line']}"))
        elif simplify(parse_expr(f"({expr_str}) - (x + {expr_str.replace('x', '0')})", evaluate=False)) != 0:
            violations.append(msg(f"non increment: {a['line']}"))


def checkRead(func: Function, param_idx: int, violations: List[str]) -> None:
    """
    Equivalent to fortran.validator's checkRead. Appends a violation for
    each assignment whose LHS is `func.parameters[param_idx]`.
    """
    param = func.parameters[param_idx]

    def msg(line: int) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {param}): {line}"

    for assign in _flang_body(func)["assignments"]:
        if is_ref(assign["lhs"], param):
            violations.append(msg(assign["line"]))


def checkRuntimeDimensionArrays(func: Function, consts: Set[str], violations: List[str]) -> None:
    """
    Equivalent to fortran.validator's checkRuntimeDimensionArrays. Appends
    a violation for each locally declared array whose shape bounds refer
    to a kernel parameter or an OP2 const.
    """
    blacklist = set(consts) | set(func.parameters)

    for local in _flang_body(func).get("locals", []):
        name = local["name"]
        if name in func.parameters:
            continue

        for dim_name in local.get("dims", []):
            if dim_name in blacklist:
                violations.append(f"In {func.name}: variable {name}, dimension {dim_name}")


def checkSlice(func: Function, param_idx: int, funcs: List[Function], violations: List[str]) -> None:
    """
    Equivalent to fortran.validator's checkSlice. Appends a violation if
    `func.parameters[param_idx]` is used as a full array reference or
    slice/section which is incompatible with stride insertion.
    """
    param = func.parameters[param_idx]
    if not _param_has_array_shape(func, param):
        return

    known_names = {f.name for f in funcs}

    def msg(line: int) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {param}): {line}"

    def visit(
        expr: Dict[str, Any],
        line: int,
        in_actual_arg: bool = False,
        in_subscript_of: Optional[bool] = None,
    ) -> None:
        kind = expr.get("kind")

        if kind == "name":
            if expr.get("value") != param:
                return
            if in_actual_arg or in_subscript_of is True:
                return
            violations.append(msg(line))
            return

        if kind in ("part_ref", "funcref"):
            if expr.get("name") == param:
                if kind == "part_ref":
                    for sub in expr.get("subscripts", []):
                        if sub.get("kind") == "triplet":
                            violations.append(msg(line))
                            break

            if kind == "part_ref":
                for sub in expr.get("subscripts", []):
                    if sub.get("kind") == "triplet":
                        for part in ("lower", "upper", "stride"):
                            if sub.get(part) is not None:
                                visit(sub[part], line)
                    else:
                        visit(sub, line, in_subscript_of=(expr.get("name") in known_names))
            else:
                for arg in expr.get("args", []):
                    visit(arg, line, in_actual_arg=True)
            return

        if kind == "binary":
            visit(expr["left"], line)
            visit(expr["right"], line)
            return

        if kind in ("paren", "unary"):
            visit(expr["expr"], line)
            return

    body = _flang_body(func)
    for assign in body["assignments"]:
        visit(assign["lhs"], assign["line"])
        visit(assign["rhs"], assign["line"])
    for call in body["calls"]:
        for arg in call.get("args", []):
            visit(arg, call["line"], in_actual_arg=True)


# Entry points

def can_validate_with_flang(loop: OP.Loop, program: Program, app: Application) -> bool:
    """
    Returns True if the kernel and every dependency needed to validate
    `loop` were parsed by Flang and carry `flang_body` data.
    """
    kernel_entities = app.findEntities(loop.kernel, program, [])
    if len(kernel_entities) != 1 or not isinstance(kernel_entities[0], Function):
        return False

    dependencies, _unknown = ftk.extractDependencies(kernel_entities, app, [])
    entities = kernel_entities + [e for e in dependencies if isinstance(e, Function)]

    return all(getattr(e, "flang_body", None) is not None for e in entities)


def validateLoop(loop: OP.Loop, program: Program, app: Application) -> None:
    """
    Equivalent to fortran.validator's validateLoop, walking the `flang_body`
    JSON instead of an fparser2 AST. Checks that `loop`'s kernel and its
    dependencies obey OP2 access-type and related restrictions, printing
    warnings and setting `loop.fallback` when an optimized kernel cannot
    be generated.
    """
    kernel_entities = app.findEntities(loop.kernel, program, [])

    if len(kernel_entities) == 0:
        raise OpError(f"unable to find kernel subroutine for {loop.kernel}")
    elif len(kernel_entities) > 1:
        raise OpError(f"ambiguous kernel subroutine for {loop.kernel}")

    dependencies, unknown_dependencies = ftk.extractDependencies(kernel_entities, app, [])
    entities = kernel_entities + list(filter(lambda e: isinstance(e, Function), dependencies))

    if len(unknown_dependencies) > 0:
        printViolations(loop, "unknown subroutine/function references", list(set(unknown_dependencies)))
        loop.fallback = True

    seen_entity_names = []
    for entity in entities:
        if entity.name in seen_entity_names:
            raise OpError(f"ambiguous subroutine/function {entity.name} used in kernel {loop.kernel}")
        seen_entity_names.append(entity.name)

    if len(loop.args) != len(kernel_entities[0].parameters):
        raise OpError(
            f"op_par_loop argument list length ({len(loop.args)}) mismatch "
            f"(expected: {len(kernel_entities[0].parameters)}, kernel subroutine: {loop.kernel})",
            loop.loc,
        )

    const_ptrs = app.constPtrs()

    violations: List[str] = []
    read_violations: List[str] = []

    for entity in entities:
        if not isinstance(entity, Function):
            continue

        const_param_aliases = set()
        for idx, param in enumerate(entity.parameters):
            if param not in const_ptrs:
                continue
            const_param_aliases.add(param)
            violations.append(f"In {entity.name}: parameter {idx + 1} ({param})")

        checkConstRead(entity, [c for c in const_ptrs if c not in const_param_aliases], read_violations)

    if len(violations) > 0:
        printViolations(loop, "subroutine/function parameter and const conflict", violations)

    if len(read_violations) > 0:
        printViolations(loop, "const written", read_violations)
        loop.fallback = True

    for entity in entities:
        if not isinstance(entity, Function):
            continue
        for name in iter_all_names(entity):
            if name in const_ptrs and name not in entity.parameters:
                loop.addConst(name)

    function_entities = [e for e in entities if isinstance(e, Function)]

    for idx, arg in enumerate(loop.args):
        if not (
            isinstance(arg, OP.ArgGbl)
            and arg.access_type in [OP.AccessType.MIN, OP.AccessType.MAX, OP.AccessType.INC, OP.AccessType.WORK]
        ) and not isinstance(arg, OP.ArgDat):
            continue

        if (isinstance(arg, OP.ArgGbl) and arg.dim == 1) or (isinstance(arg, OP.ArgDat) and loop.dat(arg).dim == 1):
            continue

        violations = []
        map_param(kernel_entities[0], idx, function_entities, checkSlice, function_entities, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]
            printViolations(
                loop, "element-wise access incompatible with stride insertion", violations, (idx, param_name)
            )
            loop.fallback = True

    for idx, arg in enumerate(loop.args):
        if isinstance(arg, OP.ArgInfo) or (hasattr(arg, "access_type") and arg.access_type != OP.AccessType.READ):
            continue

        violations = []
        map_param(kernel_entities[0], idx, function_entities, checkRead, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]
            msg = (
                "is an op_arg_idx but was written"
                if isinstance(arg, OP.ArgIdx)
                else "marked OP_READ but was written"
            )
            printViolations(loop, msg, violations, (idx, param_name))
            loop.fallback = True

    for idx, arg in enumerate(loop.args):
        if not isinstance(arg, OP.ArgDat) or arg.access_type != OP.AccessType.INC:
            continue

        violations = []
        map_param(kernel_entities[0], idx, function_entities, checkInc, function_entities, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]
            printViolations(loop, "marked OP_INC but not incremented", violations, (idx, param_name))
            loop.fallback = True

    violations = []
    for entity in function_entities:
        checkRuntimeDimensionArrays(entity, app.constPtrs(), violations)

    if len(violations) > 0:
        printViolations(loop, "runtime dimension local arrays", violations)
