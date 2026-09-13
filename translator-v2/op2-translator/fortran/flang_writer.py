"""
Text-level kernel rewriting for the --parser flang code path.
The helpers in this module are Flang-driven equivalents of the
fparser2-based rewriters in `fortran.translator.kernels`.

fparser2 helper                          flang_writer equivalent
--------------------------------------   -----------------------
ftk.writeSource(entities, prologue)      write_source
ftk.renameConsts(lang, entities, ...)    rename_consts
ftk.renameEntities(entities, replace)    rename_entities
ftk.fixHydraIO(entity)                   fix_hydra_io
ftk.removeExternals(entity)              remove_externals
ftk.insertStrides(...)                   insert_strides
ftk.insertAtomicIncs(..., c_api=False)   insert_atomic_incs
"""

from __future__ import annotations

import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import op as OP
from fortran.flang_validator import map_param
from language import Lang
from op import OpError
from store import Application, Entity, Function


# Private helpers

def _flang_functions(entities: Iterable[Entity]) -> List[Function]:
    """
    Return only the given entities that are Functions with Flang source text
    attached (flang_source).
    """
    out: List[Function] = []
    for e in entities:
        if isinstance(e, Function) and getattr(e, "flang_source", None):
            out.append(e)
    return out


def _substitute_outside_strings(source: str, replace_chunk: Callable[[str], str]) -> str:
    """
    Apply `replace_chunk` to every contiguous span of `source` that does not
    fall inside a Fortran character literal.
    """
    out: List[str] = []
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        if c == '"' or c == "'":
            quote = c
            j = i + 1
            while j < n:
                if source[j] == quote:
                    if j + 1 < n and source[j + 1] == quote:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(source[i:j])
            i = j
            continue

        j = i
        while j < n and source[j] not in '"\'':
            j += 1
        out.append(replace_chunk(source[i:j]))
        i = j

    return "".join(out)


# Mutations

def rename_consts(
    lang: Lang,  # kept for API symmetry
    entities: Sequence[Entity],
    app: Application,
    replacement: Callable[[str], str],
) -> None:
    """
    Rename references to OP2 declared consts inside each entity's flang_source.

    Mirrors fparser2's `renameConsts`; an identifier is renamed if it matches
    a known const pointer and is not one of the kernel parameters of the
    enclosing subprogram.
    """
    const_ptrs = app.constPtrs()

    for entity in _flang_functions(entities):
        targets = {c.lower() for c in const_ptrs} - {p.lower() for p in entity.parameters}
        if not targets:
            continue

        pattern = re.compile(
            r"\b(" + "|".join(re.escape(t) for t in sorted(targets)) + r")\b",
            flags=re.IGNORECASE,
        )

        def sub_chunk(chunk: str, _pattern=pattern) -> str:
            return _pattern.sub(lambda m: replacement(m.group(1).lower()), chunk)

        entity.flang_source = _substitute_outside_strings(entity.flang_source, sub_chunk)


def rename_entities(entities: Sequence[Entity], replacement: Callable[[str], str]) -> None:
    """
    Rename a group of subprograms in lockstep, updating both the definitions
    and any call sites among the same group.

    Mirrors fparser2's `renameEntities`.
    """
    funcs = _flang_functions(entities)
    rename_plan = [(f, f.name, replacement(f.name)) for f in funcs]

    for func, old_name, new_name in rename_plan:
        for other, _, _ in rename_plan:
            if other is func:
                continue
            rename_function_calls(other, old_name, new_name)

    for func, _, new_name in rename_plan:
        rename_function_definition(func, new_name)


def rename_function_calls(entity: Function, name: str, replacement: str) -> None:
    """
    Rewrite `call <name>(...)` and function-style references `<name>(...)`
    inside the entity body.
    """
    if not getattr(entity, "flang_source", None):
        return

    old = re.escape(name)

    def sub_chunk(chunk: str) -> str:
        chunk = re.sub(
            rf"(\bcall\s+){old}\b",
            lambda m: m.group(1) + replacement,
            chunk,
            flags=re.IGNORECASE,
        )
        chunk = re.sub(
            rf"\b{old}(\s*\()",
            lambda m: replacement + m.group(1),
            chunk,
            flags=re.IGNORECASE,
        )
        return chunk

    entity.flang_source = _substitute_outside_strings(entity.flang_source, sub_chunk)


def rename_function_definition(entity: Function, replacement: str) -> None:
    """
    Rename the subroutine/function definition within `entity.flang_source`.
    """
    if not getattr(entity, "flang_source", None):
        return

    old = re.escape(entity.name)

    def sub_chunk(chunk: str) -> str:
        chunk = re.sub(
            rf"(\b(?:subroutine|function)\s+){old}\b",
            lambda m: m.group(1) + replacement,
            chunk,
            flags=re.IGNORECASE,
        )
        chunk = re.sub(
            rf"(\bend\s+(?:subroutine|function)\s+){old}\b",
            lambda m: m.group(1) + replacement,
            chunk,
            flags=re.IGNORECASE,
        )
        return chunk

    entity.flang_source = _substitute_outside_strings(entity.flang_source, sub_chunk)
    entity.name = replacement


# Cleanup helpers

_EXTERNAL_LINE_RE = re.compile(r"^\s*external\s*(?:::)?\s*[^\n]*\n?", flags=re.IGNORECASE | re.MULTILINE)
_WRITE_STMT_RE = re.compile(r"^\s*write\s*\([^\n]*\)[^\n]*\n?", flags=re.IGNORECASE | re.MULTILINE)

_HYDRA_CLOBBER_CALLS = (
    "hyd_dump",
    "hyd_dump_hydrales",
    "hyd_error_dump",
    "hyd_error_dump_hydrales",
    "hyd_error_print",
    "hyd_error_print_hydrales",
    "hyd_kill",
    "hyd_kill_hydrales",
    "hyd_print",
    "hyd_print_hydrales",
)

_HYDRA_CALL_RE = re.compile(
    r"^(\s*)call\s+(" + "|".join(_HYDRA_CLOBBER_CALLS) + r")\b[^\n]*\n?",
    flags=re.IGNORECASE | re.MULTILINE,
)


def fix_hydra_io(entity: Function) -> None:
    """
    Replace `write(...)` statements with `continue` and certain hydra
    helper calls with `stop`.
    """
    if not getattr(entity, "flang_source", None):
        return

    def sub_chunk(chunk: str) -> str:
        chunk = _WRITE_STMT_RE.sub("continue\n", chunk)
        chunk = _HYDRA_CALL_RE.sub(lambda m: m.group(1) + "stop\n", chunk)
        return chunk

    entity.flang_source = _substitute_outside_strings(entity.flang_source, sub_chunk)


def remove_externals(entity: Function) -> None:
    """
    Strip `external` declarations from `entity.flang_source`.
    """
    if not getattr(entity, "flang_source", None):
        return

    def sub_chunk(chunk: str) -> str:
        return _EXTERNAL_LINE_RE.sub("", chunk)

    entity.flang_source = _substitute_outside_strings(entity.flang_source, sub_chunk)


# Private insert_strides / insert_atomic_incs helpers

def _ensure_op2_ret_decl(source: str) -> str:
    """
    Insert `integer(4) :: op2_ret` into the source.
    """
    if re.search(r"\binteger\s*\(?\s*4?\s*\)?\s*::\s*op2_ret\b", source, flags=re.IGNORECASE):
        return source
    if re.search(r"\binteger\s*\(kind\s*=\s*4\)\s*::\s*op2_ret\b", source, flags=re.IGNORECASE):
        return source

    lines = source.splitlines(keepends=True)
    insert_at = 0
    decl_start = re.compile(
        r"^\s*(integer|real|logical|character|complex|type|double|dimension|"
        r"parameter|data|common|save|external|intrinsic|implicit|use)\b",
        flags=re.IGNORECASE,
    )
    end_decl = re.compile(
        r"^\s*(contains|do|if|select|where|forall|call|stop|return|cycle|exit|"
        r"go\s*to|continue|assign|open|close|read|write|print|allocate|"
        r"deallocate|nullify|backspace|rewind|endfile|inquire|flush|wait|"
        r"sync|lock|unlock|event|form\s+team|change\s+team|end\s+team|"
        r"[a-zA-Z_]\w*\s*=)",
        flags=re.IGNORECASE,
    )

    seen_decl = False
    for i, line in enumerate(lines):
        if decl_start.match(line):
            seen_decl = True
            insert_at = i + 1
            continue
        if seen_decl and line.strip() and not decl_start.match(line):
            if end_decl.match(line) or (not line.strip().startswith("!") and "::" not in line and not decl_start.match(line)):
                insert_at = i
                break
            insert_at = i + 1

    lines.insert(insert_at, "integer(4) :: op2_ret\n")
    return "".join(lines)


def _erase_param_dimensions(source: str, param: str) -> str:
    """
    Collapse the parameter's explicit shape to assumed-size `*` in both
    `dimension(...)` attributes and entity-level `name(bounds)` forms.
    """
    lines = source.splitlines(keepends=True)
    out: List[str] = []
    entity_shape_re = re.compile(
        rf"\b{re.escape(param)}\s*\([^)]*\)",
        flags=re.IGNORECASE,
    )
    dim_attr_re = re.compile(r"\bdimension\s*\([^)]*\)", flags=re.IGNORECASE)

    for line in lines:
        if "::" in line and re.search(rf"\b{re.escape(param)}\b", line, flags=re.IGNORECASE):
            before, after = line.split("::", 1)
            if dim_attr_re.search(before):
                before = dim_attr_re.sub("dimension(*)", before)
            after = entity_shape_re.sub(f"{param}(*)", after)
            line = before + "::" + after
        out.append(line)

    return "".join(out)


def _expr_to_fortran(expr: Optional[Dict[str, Any]]) -> str:
    """
    Convert a JSON expression to Fortran text for bound expressions in decls.
    """
    if expr is None:
        return "1"

    kind = expr.get("kind")
    if kind == "int_lit":
        text = expr["text"]
        if expr.get("kind_text"):
            return f"{text}_{expr['kind_text']}"
        return text
    if kind == "real_lit":
        text = expr["text"]
        if expr.get("kind_text"):
            return f"{text}_{expr['kind_text']}"
        return text
    if kind == "name":
        return expr["value"]
    if kind == "paren":
        return f"({_expr_to_fortran(expr['expr'])})"
    if kind == "unary":
        return f"{expr['op']}{_expr_to_fortran(expr['expr'])}"
    if kind == "binary":
        return f"{_expr_to_fortran(expr['left'])} {expr['op']} {_expr_to_fortran(expr['right'])}"
    if kind in ("part_ref", "funcref"):
        items = expr.get("subscripts") if kind == "part_ref" else expr.get("args", [])
        return f"{expr['name']}({', '.join(_expr_to_fortran(i) for i in items)})"

    raise OpError(f"Unsupported bound expression while inserting strides: {expr}")


def _find_balanced_paren(source: str, open_idx: int) -> int:
    """
    Return the index of the closing parenthesis that matches `source[open_idx]`,
    or -1 if the parentheses are unbalanced.
    """
    depth = 0
    for i in range(open_idx, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _flatten_index(subscript_text: str, dims: List[Tuple[str, Optional[str]]]) -> str:
    """
    Collapse a multi-dimensional subscript into a 1D index expression.
    """
    subs = _split_top_level_args(subscript_text)
    if len(dims) != len(subs):
        raise OpError(f"Unexpected dimension mismatch ({subscript_text}, {dims})")

    sizes = []
    for lb, ub in dims:
        if lb == "1":
            sizes.append(f"({ub})")
        else:
            sizes.append(f"(1 + {ub} - ({lb}))")

    if dims[0][0] == "1":
        index = subs[0]
    else:
        index = f"({subs[0]} + 1 - ({dims[0][0]}))"

    for i, extra in enumerate(subs[1:], start=1):
        index += f" + ({extra} - ({dims[i][0]})) * {'*'.join(sizes[:i])}"

    return index


def _insert_stride_on_entity(entity: Function, param_idx: int, stride: str) -> bool:
    """
    Rewrite indexed references to the given dummy parameter as
    `param(op2_s(index, stride))`, flattening multi-dimensional
    subscripts and collapsing the declared shape to `*`.
    """
    if not getattr(entity, "flang_source", None):
        return False

    param = entity.parameters[param_idx]
    dims = _param_dims(entity, param)
    erase_dimensions = dims is not None
    if dims is None:
        dims = [("1", None)]

    hits = _iter_param_indexings(entity.flang_source, param)
    if not hits:
        return False

    # replace from the end, so earlier offsets remain valid
    source = entity.flang_source
    for start, end, subscript_text in reversed(hits):
        index = _flatten_index(subscript_text, dims)
        replacement = f"{param}(op2_s({index}, {stride}))"
        source = source[:start] + replacement + source[end:]

    if erase_dimensions:
        source = _erase_param_dimensions(source, param)

    entity.flang_source = source
    return True


def _iter_param_indexings(source: str, param: str) -> List[Tuple[int, int, str]]:
    """
    Return (start, end, subscript_text) for every `param(...)` occurrence
    outside string literals.
    """
    hits: List[Tuple[int, int, str]] = []
    pattern = re.compile(rf"\b{re.escape(param)}\s*\(", flags=re.IGNORECASE)

    in_string = [False] * len(source)
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        if c in "\"'":
            quote = c
            j = i + 1
            while j < n:
                if source[j] == quote:
                    if j + 1 < n and source[j + 1] == quote:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            for k in range(i, j):
                in_string[k] = True
            i = j
        else:
            i += 1

    for m in pattern.finditer(source):
        if in_string[m.start()]:
            continue
        open_idx = m.end() - 1
        close_idx = _find_balanced_paren(source, open_idx)
        if close_idx < 0:
            continue
        hits.append((m.start(), close_idx + 1, source[open_idx + 1:close_idx]))

    return hits


def _lhs_token_pattern(lhs: str) -> str:
    """
    Regex for a whole LHS reference.

    Bare names use a word boundary so, for example,
    `res` does not match inside `residual`.
    """
    escaped = re.escape(lhs)
    if "(" in lhs:
        return escaped
    return rf"\b{escaped}\b"


def _param_dims(entity: Function, param: str) -> Optional[List[Tuple[str, Optional[str]]]]:
    """
    Return explicit shape bounds for `param` from `flang_body["decls"]`,
    or None if the parameter has no array shape.
    """
    body = getattr(entity, "flang_body", None)
    if body is None:
        return None

    for decl in body.get("decls", []):
        if decl.get("kind") != "type_decl":
            continue

        attr_dim = decl.get("dim")
        for ent in decl.get("entities", []):
            if ent.get("name") != param:
                continue

            dim = ent.get("dim") or attr_dim
            if dim is None or dim.get("kind") != "explicit":
                return None

            out: List[Tuple[str, Optional[str]]] = []
            for d in dim.get("shape", []):
                lb = "1" if d.get("lb") is None else _expr_to_fortran(d["lb"])
                ub = _expr_to_fortran(d["ub"]) if d.get("ub") is not None else None
                out.append((lb, ub))
            return out

    return None


def _replace_fortran_increments(source: str, param: str, typ: OP.Type) -> Tuple[str, bool]:
    """
    Rewrite single-line increment assignments `param = param +/- expr` as
    `op2_ret = atomicAdd(param, amount)`.

    Returns the rewritten source and a bool indicating whether any
    replacement was made.
    """
    lines = source.splitlines(keepends=True)
    changed = False
    out: List[str] = []

    assign_re = re.compile(r"^(\s*)(.+?)\s*=\s*(.+?)(\s*)$")

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            out.append(line)
            continue

        m = assign_re.match(line.rstrip("\n"))
        if m is None:
            out.append(line)
            continue

        indent, lhs, rhs, trailing = m.group(1), m.group(2).strip(), m.group(3).strip(), m.group(4)
        lhs_l = lhs.lower()

        # LHS must be a reference to a parameter
        if not re.match(rf"^{re.escape(param)}(\s*\(.*\))?$", lhs_l, flags=re.IGNORECASE):
            out.append(line)
            continue

        if not _rhs_is_increment_of(lhs, rhs):
            out.append(line)
            continue

        zero = _zero_literal_fortran(typ)
        amount = _replace_lhs_with_zero(rhs, lhs, zero)
        newline = "\n" if line.endswith("\n") else ""
        out.append(f"{indent}op2_ret = atomicAdd({lhs}, {amount}){trailing}{newline}")
        changed = True

    return "".join(out), changed


def _replace_lhs_with_zero(rhs: str, lhs: str, zero: str) -> str:
    """
    Replace whole occurrences of `lhs` in `rhs` with `zero`.
    """
    return re.sub(_lhs_token_pattern(lhs), zero, rhs, flags=re.IGNORECASE)


def _rhs_is_increment_of(lhs: str, rhs: str) -> bool:
    """
    True if `rhs` looks like `lhs +/- expr` or `expr +/- lhs`.
    """
    token = _lhs_token_pattern(lhs)
    return re.search(
        rf"{token}\s*[+\-]|[+\-]\s*{token}",
        rhs,
        flags=re.IGNORECASE,
    ) is not None


def _split_top_level_args(arglist: str) -> List[str]:
    """
    Split a comma-separated argument list on commas that are not nested
    inside parentheses.
    """
    parts: List[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(arglist):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(arglist[start:i].strip())
            start = i + 1
    parts.append(arglist[start:].strip())
    return parts


def _zero_literal_fortran(typ: OP.Type) -> str:
    """
    Return a Fortran zero literal whose type matches `typ`.
    """
    if isinstance(typ, OP.Int):
        return "0"
    if isinstance(typ, OP.Float) and typ.size == 32:
        return "0.0"
    if isinstance(typ, OP.Float) and typ.size == 64:
        return "0.0d0"
    raise OpError(f"Error: unexpected arg type while inserting atomics: {typ}")


# Atomic increment insertion and stride insertion

def insert_atomic_incs(
    entities: List[Function],
    loop: OP.Loop,
    match: Callable[[Any], bool],
) -> None:
    """
    Rewrite increments to `op2_ret = atomicAdd(...)` and declare
    `integer(4) :: op2_ret`.

    Mirrors fparser2's `insertAtomicIncs`.
    """

    if not entities:
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

            if not getattr(entity2, "flang_source", None):
                return False

            param_name = entity2.parameters[param_idx2]
            new_source, replaced = _replace_fortran_increments(entity2.flang_source, param_name, typ2)
            if not replaced:
                return False

            entity2.flang_source = new_source
            modified.setdefault(entity2.name, set()).add(param_idx2)
            return False

        map_param(entities[0], arg_idx, entities, op, typ)

    for name, indices in modified.items():
        if not indices:
            continue
        entity = next((e for e in entities if e.name == name), None)
        if entity is not None and getattr(entity, "flang_source", None):
            entity.flang_source = _ensure_op2_ret_decl(entity.flang_source)


def insert_strides(
    entities: List[Function],
    loop: OP.Loop,
    stride: Callable[[Any], str],
    match: Callable[[Any], bool] = lambda arg: True,
    modified: Optional[Dict[str, Set[int]]] = None,
) -> Dict[str, Set[int]]:
    """
    Rewrite indexed accesses of matching loop arguments to apply a stride.

    Mirrors fparser2's `insertStrides`.
    """
    if modified is None:
        modified = {}

    if not entities:
        return modified

    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        if arg_idx in modified.get(entities[0].name, set()):
            continue

        stride_name = stride(loop.args[arg_idx])

        def op(entity2: Function, param_idx2: int, stride2: str) -> bool:
            if param_idx2 in modified.get(entity2.name, set()):
                return False

            _insert_stride_on_entity(entity2, param_idx2, stride2)
            modified.setdefault(entity2.name, set()).add(param_idx2)

            return False

        map_param(entities[0], arg_idx, entities, op, stride_name)

    return modified


# Source emission

def write_source(entities: Sequence[Entity], prologue: Optional[str] = None) -> str:
    """
    Emit the Flang-derived source text for the given entities, concatenated in
    the same order that `fortran.translator.kernels.writeSource` uses
    (innermost dependencies last).

    The optional `prologue` is prepended to each entity.
    """

    # late import to avoid a circular dependency between flang_writer and fortran.translator.kernels
    from fortran.translator.kernels import addLineContinuations

    funcs = _flang_functions(entities)
    if not funcs:
        return ""

    pieces = []
    for entity in reversed(funcs):
        body = entity.flang_source or ""
        body = addLineContinuations(body)
        if prologue:
            body = prologue + body
        pieces.append(body.rstrip("\n"))

    return "\n\n".join(pieces) + "\n"
