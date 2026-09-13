"""
Flang port of `fortran.translator.kernels_c` (translating a Fortran kernel
subroutine/function to C++), operating on the `decls`/ `stmts` JSON
produced by `op2-flang-scan` instead of an fparser2 AST.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import op as OP
from fortran.flang_validator import is_ref, map_param
from fortran.translator.kernels_c import (
    FArray,
    FCharacter,
    FInteger,
    FLogical,
    FPrimitive,
    FReal,
    FType,
    Param,
    indent,
)
from op import OpError
from store import Application, Function
from util import safeFind


# Info / Context

@dataclass
class SubprogramInfo:
    name: str
    body: Dict[str, Any]

    params: List[Param] = field(default_factory=list)
    func_return_var: Optional[str] = None

    types: Dict[str, FType] = field(default_factory=dict)

    def isFunction(self) -> bool:
        return self.func_return_var is not None

    def functionType(self) -> FType:
        assert self.isFunction()
        return self.types[self.func_return_var]

    def paramNames(self) -> List[str]:
        return [p.name for p in self.params]

    def lookupParam(self, name: str) -> Optional[Param]:
        return safeFind(self.params, lambda p: p.name == name)


@dataclass
class Info:
    loop: OP.Loop
    config: Dict[str, Any]

    consts: Dict[str, FType] = field(default_factory=dict)

    entry_subprogram: Optional[str] = None
    subprograms: Dict[str, SubprogramInfo] = field(default_factory=dict)

    def functionNames(self) -> List[str]:
        return [s.name for s in self.subprograms.values() if s.isFunction()]


@dataclass
class Context:
    info: Info
    sub_info: SubprogramInfo

    def isEntry(self) -> bool:
        return self.sub_info.name == self.info.entry_subprogram

    def lookupType(self, name: str) -> Optional[FType]:
        return self.sub_info.types.get(name) or self.info.consts.get(name)

    def error(self, msg: str) -> None:
        raise OpError(f"Error translating {self.sub_info.name}: {msg}")


def _flang_body(entity: Function) -> Dict[str, Any]:
    """
    Return the decls/stmts JSON (`flang_body`) attached to `entity`.

    Raises `OpError` if translator not run with `--parser flang`.
    """
    body = getattr(entity, "flang_body", None)
    if body is None:
        raise OpError(f"Missing Flang body data for {entity.name}")
    return body


def canTranslateWithFlang(entities: List[Function]) -> bool:
    """
    True if every entity has a `flang_body`.
    """
    return all(getattr(e, "flang_body", None) is not None for e in entities)


# parseInfo

def _ftypeFromOpType(typ: OP.Type, dim: int) -> FType:
    """
    Convert an OP2 `Type` (and array `dim`) into the `FType`
    used by the C++ emission.
    """
    if isinstance(typ, OP.Int):
        if not typ.signed:
            raise OpError(f"Unsupported unsigned const type: {typ}")
        inner: FPrimitive = FInteger(typ.size // 8)
    elif isinstance(typ, OP.Float):
        inner = FReal(typ.size // 8)
    elif isinstance(typ, OP.Bool):
        inner = FLogical()
    else:
        raise OpError(f"Unsupported const type for Flang translation: {typ}")

    if dim <= 1:
        return inner

    return FArray([("1", str(dim))], inner)


def buildConstsInfo(app: Application, const_rename: Optional[Callable[[str], str]]) -> Dict[str, FType]:
    """
    Map each application const (optionally renamed by `const_rename`) to
    an `FType`.
    """
    consts: Dict[str, FType] = {}

    for c in app.consts():
        actual_name = const_rename(c.ptr) if const_rename else c.ptr
        consts[actual_name] = _ftypeFromOpType(c.typ, c.dim)

    return consts


def parseArraySpec(dim_obj: Dict[str, Any], ctx: Context) -> List[Tuple[str, str]]:
    """
    Translate an explicit shape array spec into `(lower, upper)` bound pairs.
    """
    if dim_obj.get("kind") != "explicit":
        ctx.error(f"Unsupported array spec: {dim_obj}")

    shape_spec_list = []
    for d in dim_obj.get("shape", []):
        ub = translateExpr(d["ub"], ctx)

        if d.get("lb") is None:
            shape_spec_list.append(("1", ub))
        else:
            shape_spec_list.append((translateExpr(d["lb"], ctx), ub))

    return shape_spec_list


def parseInfo(
    entities: List[Function],
    app: Application,
    loop: OP.Loop,
    config: Dict[str, Any],
    const_rename: Optional[Callable[[str], str]] = None,
) -> Info:
    """
    Build the translation `Info` for a kernel and its callees.

    Walks `entities` (the entry kernel first, then transitive callees),
    records each subprogram, attaches const types from `app`, fills in
    per-subprogram type maps, maps loop arguments onto dummy parameters,
    and resolves which parameters are const.
    """
    info = Info(loop, config)

    is_first = True
    for entity in entities:
        sub_info = parseSubprogramInfo(entity)
        info.subprograms[sub_info.name] = sub_info

        if is_first:
            info.entry_subprogram = sub_info.name
            is_first = False

    info.consts = buildConstsInfo(app, const_rename)

    for sub_info in info.subprograms.values():
        parseSubprogramTypeInfo(Context(info, sub_info))

    resolveOpArgs(entities, info, loop)
    resolveParamAccesses(info)

    return info


def parseIntrinsicType(type_obj: Dict[str, Any], ctx: Context) -> FType:
    """
    Translate a Flang intrinsic type into an `FInteger` / `FReal` /
    `FLogical` / `FCharacter`.
    """
    if type_obj.get("kind") != "intrinsic":
        ctx.error(f"Unable to parse intrinsic type: {type_obj}")

    base = type_obj.get("base")
    kind_text = type_obj.get("kind_text")
    kind_norm = kind_text.upper() if (kind_text is not None and not kind_text.isdigit()) else kind_text

    if base == "integer":
        if kind_norm in (None, "4", "IK", "IK4"):
            return FInteger(4)
        elif kind_norm in ("8", "IK8"):
            return FInteger(8)

    elif base == "real":
        if kind_norm in (None, "4", "RK4"):
            return FReal(4)
        elif kind_norm in ("8", "RK", "RK8"):
            return FReal(8)

    elif base == "logical":
        if kind_norm in (None, "LK"):
            return FLogical()

    elif base == "character":
        charlen = type_obj.get("charlen")
        if charlen is None:
            ctx.error("Unknown character type spec")
        return FCharacter(translateExpr(charlen, ctx))

    ctx.error(f"Unable to parse intrinsic type: {type_obj}")


def parseSubprogramInfo(entity: Function) -> SubprogramInfo:
    """
    Collect the name, dummy parameters, and result variable
    of a kernel or helper from `entity.flang_body`.
    """
    body = _flang_body(entity)
    sub_info = SubprogramInfo(entity.name, body)
    sub_info.params = [Param(p) for p in entity.parameters]

    if body.get("is_function"):
        result_name = body.get("result_name")
        sub_info.func_return_var = f"_{result_name}" if result_name else f"_{sub_info.name}"

    return sub_info


def parseSubprogramTypeInfo(ctx: Context) -> None:
    """
    Fill `ctx.sub_info.types` from the subprogram's type declarations.
    """
    body = ctx.sub_info.body
    ctx.sub_info.types = parseTypes(body.get("decls", []), ctx)

    if not ctx.sub_info.isFunction():
        return

    return_var = ctx.sub_info.func_return_var
    bare_name = return_var[1:]
    if bare_name in ctx.sub_info.types:
        ctx.sub_info.types[return_var] = ctx.sub_info.types.pop(bare_name)

    result_type = body.get("result_type")
    if result_type is not None:
        if return_var in ctx.sub_info.types:
            ctx.error("Unexpected duplicate function type spec")

        ctx.sub_info.types[return_var] = parseIntrinsicType(result_type, ctx)

    if return_var not in ctx.sub_info.types:
        ctx.error("Could not resolve function type")

    if not isinstance(ctx.sub_info.types[return_var], FPrimitive):
        ctx.error(f"Non-primitive function return: {ctx.sub_info.types[return_var]}")


def parseTypes(decls: List[Dict[str, Any]], ctx: Context) -> Dict[str, FType]:
    """
    Build a name-to-`FType` map from `type_decl` nodes.
    """
    type_map: Dict[str, FType] = {}

    for decl in decls:
        if decl.get("kind") != "type_decl":
            continue

        intrinsic_type = parseIntrinsicType(decl["type"], ctx)
        attr_array_spec = parseArraySpec(decl["dim"], ctx) if decl.get("dim") else None

        for entity_decl in decl.get("entities", []):
            name = entity_decl["name"]
            own_spec = parseArraySpec(entity_decl["dim"], ctx) if entity_decl.get("dim") else None

            if attr_array_spec is not None and own_spec is not None:
                ctx.error("Type declaration has both dimension() attr and array spec")

            if own_spec is None:
                type_map[name] = FArray(attr_array_spec, intrinsic_type) if attr_array_spec is not None else intrinsic_type
            else:
                type_map[name] = FArray(own_spec, intrinsic_type)

    return type_map


def resolveOpArgs(entities: List[Function], info: Info, loop: OP.Loop) -> None:
    """
    Attach each `loop` argument to the dummy parameter it maps onto,
    including through nested calls.
    """
    def setOpArg(entity2: Function, param_idx: int, info2: Info, op_arg: OP.Arg) -> bool:
        sub_info2 = info2.subprograms[entity2.name]
        sub_info2.params[param_idx].op_arg = op_arg
        return False

    for i in range(len(loop.args)):
        map_param(entities[0], i, entities, setOpArg, info, loop.args[i])


# resolveParamAccesses

def _exprBaseName(expr: Dict[str, Any]) -> Optional[str]:
    """
    Return the identifier of a `name` / `part_ref` / `funcref` expression,
    or None if `expr` is not a simple reference.
    """
    kind = expr.get("kind")
    if kind == "name":
        return expr.get("value")
    if kind in ("part_ref", "funcref"):
        return expr.get("name")
    return None


def _collectDoVars(stmts: List[Dict[str, Any]]) -> Set[str]:
    """
    Collect counted do loop control variables from `stmts` and nested
    if/do bodies.
    """
    names: Set[str] = set()

    def walk(items: List[Dict[str, Any]]) -> None:
        for stmt in items:
            kind = stmt.get("kind")
            if kind == "do":
                if stmt.get("mode") == "counted" and stmt.get("var"):
                    names.add(stmt["var"])
                walk(stmt.get("body", []))
            elif kind == "if_construct":
                for branch in stmt.get("branches", []):
                    walk(branch.get("body", []))
            elif kind == "if_stmt":
                walk([stmt["stmt"]])

    walk(stmts)
    return names


def _findCallsForParamWithAtomic(body: Dict[str, Any], param: str, known_names: Set[str]) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []

    def record(name: Optional[str], idx: int) -> None:
        if name is None:
            return
        if name.lower() == "atomicadd":
            results.append(("atomicAdd", idx))
        elif name in known_names:
            results.append((name, idx))

    def visit(expr: Dict[str, Any]) -> None:
        kind = expr.get("kind")

        if kind == "name":
            return

        if kind == "part_ref":
            subs = expr.get("subscripts", [])
            for i, sub in enumerate(subs):
                if sub.get("kind") == "triplet":
                    for part in ("lower", "upper", "stride"):
                        if sub.get(part) is not None:
                            visit(sub[part])
                    continue
                if is_ref(sub, param):
                    record(expr.get("name"), i)
                visit(sub)
            return

        if kind == "funcref":
            args = expr.get("args", [])
            for i, arg in enumerate(args):
                if is_ref(arg, param):
                    record(expr.get("name"), i)
                visit(arg)
            return

        if kind == "binary":
            visit(expr["left"])
            visit(expr["right"])
            return

        if kind in ("paren", "unary"):
            visit(expr["expr"])
            return

    for assign in body.get("assignments", []):
        visit(assign["lhs"])
        visit(assign["rhs"])

    for call in body.get("calls", []):
        args = call.get("args", [])
        for i, arg in enumerate(args):
            if is_ref(arg, param):
                record(call.get("name"), i)
            visit(arg)

    return results


def resolveParamAccesses(info: Info) -> None:
    """
    Decide `is_const` for every dummy parameter of every subprogram.
    """
    for sub_info in info.subprograms.values():
        resolveParamAccessesLocal(Context(info, sub_info))

    has_unresolved = True
    while has_unresolved:
        has_unresolved = False

        for sub_info in info.subprograms.values():
            unresolved = tryResolveParams(Context(info, sub_info))
            has_unresolved = has_unresolved or unresolved


def resolveParamAccessesLocal(ctx: Context) -> None:
    """
    Mark local writes and call-site uses for each dummy parameter of `ctx.sub_info`.
    """
    body = ctx.sub_info.body

    assigned_to: Set[str] = set()
    for assign in body.get("assignments", []):
        lhs_name = _exprBaseName(assign["lhs"])
        if lhs_name is None:
            ctx.error(f"Unknown assignment LHS: {assign['lhs']}")
        assigned_to.add(lhs_name)

    assigned_to |= _collectDoVars(body.get("stmts", []))

    known_names = set(ctx.info.subprograms.keys())

    for param in ctx.sub_info.params:
        param.is_const_local = param.name not in assigned_to

        for callee_name, idx in _findCallsForParamWithAtomic(body, param.name, known_names):
            param.as_arg.add((callee_name, idx))

        if param.is_const_local and len(param.as_arg) == 0:
            param.is_const = True
        elif not param.is_const_local:
            param.is_const = False

        if param.is_const is None and ("atomicAdd", 0) in param.as_arg:
            param.is_const = False


def tryResolveParams(ctx: Context) -> bool:
    """
    Propagate callee `is_const` back onto unresolved parameters of `ctx.sub_info`.

    Returns True if any parameter is waiting on a callee.
    """
    has_unresolved = False

    for param in ctx.sub_info.params:
        if param.is_const is not None:
            continue

        all_const = True
        unresolved = False

        for target_name, idx in param.as_arg:
            target_sub = ctx.info.subprograms.get(target_name)

            if target_sub is None:
                continue

            if target_sub.params[idx].is_const is None:
                unresolved = True
                continue

            if target_sub.params[idx].is_const is False:
                all_const = False
                break

        if not all_const:
            param.is_const = False
        elif all_const and not unresolved:
            param.is_const = True
        else:
            has_unresolved = True

    return has_unresolved


# translate

def translate(info: Info) -> str:
    """
    Emit C++ prototypes and definitions for every subprogram in `info`.
    """
    decls = ""
    srcs = ""

    for sub_info in info.subprograms.values():
        decl, src = translateSubprogram(Context(info, sub_info))

        decls += decl + "\n\n"
        srcs += src + "\n\n"

    return decls + "\n" + srcs


def translateCallStmt(call_stmt: Dict[str, Any], ctx: Context) -> str:
    """
    Translate a CALL statement to `name(args);`.
    """
    call_target = translateName(call_stmt["name"], ctx)
    args = call_stmt.get("args", [])

    if len(args) == 0:
        return f"{call_target}();\n"

    return f"{call_target}({translateArgList(args, ctx, call_target)});\n"


def translateDataStmt(data_stmt: Dict[str, Any], ctx: Context) -> str:
    """
    Translate a DATA statement into a sequence of C++ assignments.
    """
    src = ""

    for s in data_stmt.get("sets", []):
        objects = s.get("objects", [])
        values = s.get("values", [])

        if len(values) != 1:
            ctx.error("Unsupported multiple value data statement")

        if values[0].get("repeated"):
            ctx.error("Unsupported repeat in data statement")

        value = translateExpr(values[0]["value"], ctx)

        assert len(objects) > 0
        for obj in objects:
            src += f"{translateExpr(obj, ctx)} = {value};\n"

    return src


def translateDoConstruct(do_stmt: Dict[str, Any], ctx: Context) -> str:
    """
    Translate a WHILE or counted DO loop.
    """
    mode = do_stmt.get("mode")
    body = "".join(translateStmt(s, ctx) for s in do_stmt.get("body", []))

    if mode == "while":
        header = f"while ({translateExpr(do_stmt['cond'], ctx)})"
    elif mode == "counted":
        var = translateName(do_stmt["var"], ctx)
        lb = translateExpr(do_stmt["lb"], ctx)
        ub = translateExpr(do_stmt["ub"], ctx)

        if do_stmt.get("step") is None:
            header = f"for ({var} = {lb}; {var} <= {ub}; ++{var})"
        else:
            step = translateExpr(do_stmt["step"], ctx)
            header = f"for ({var} = {lb}; {var} <= {ub}; {var} += {step})"
    else:
        ctx.error("Unsupported labelled/concurrent do construct")

    return f"{header} {{\n{indent(body)}\n}}\n"


def translateExecutionPart(stmts: List[Dict[str, Any]], ctx: Context) -> str:
    """
    Translate the executable statements of a subprogram.
    """
    src = ""
    last = ""

    for stmt in stmts:
        last = translateStmt(stmt, ctx)
        src += last

    if ctx.sub_info.isFunction() and not last.startswith("return"):
        src += f"return {ctx.sub_info.func_return_var};\n"

    return src


def translateIfConstruct(if_construct: Dict[str, Any], ctx: Context) -> str:
    """
    Translate an IF / ELSE IF / ELSE construct.
    """
    parts = []

    for i, branch in enumerate(if_construct.get("branches", [])):
        body = indent("".join(translateStmt(s, ctx) for s in branch.get("body", [])))

        if branch.get("cond") is not None:
            header = "if" if i == 0 else "} else if"
            parts.append(f"{header} ({translateExpr(branch['cond'], ctx)}) {{\n{body}\n")
        else:
            parts.append(f"}} else {{\n{body}\n")

    parts.append("}\n")
    return "".join(parts)


def translateSpecificationPart(decls: List[Dict[str, Any]], ctx: Context) -> str:
    init_src = ""
    parameters: Dict[str, str] = {}

    for decl in decls:
        kind = decl.get("kind")

        if kind == "parameter_stmt":
            for d in decl.get("defs", []):
                parameters[d["name"]] = translateExpr(d["value"], ctx)
            continue

        if kind == "type_decl":
            if not decl.get("is_parameter"):
                continue

            for entity_decl in decl.get("entities", []):
                name = entity_decl["name"]

                if entity_decl.get("init") is None:
                    ctx.error("Parameter has no initialization")

                parameters[name] = translateExpr(entity_decl["init"], ctx)

            continue

        if kind == "data_stmt":
            init_src += translateDataStmt(decl, ctx)
            continue

        # op2-flang-scan's DeclCollector only emits type_decl/parameter_stmt/data_stmt; nothing else should reach this point
        continue

    src = ""
    for name, type_ in ctx.sub_info.types.items():
        if name in ctx.sub_info.paramNames() and isinstance(type_, FArray):
            param = ctx.sub_info.lookupParam(name)
            src += f"const {type_.asSpan('_f2c_ptr_' + name, name, param.is_const, ctx)};\n"
            continue

        if name in ctx.sub_info.paramNames():
            continue

        if name in ctx.info.functionNames():
            continue

        if name in parameters:
            assert not isinstance(type_, FArray)
            src += f"constexpr {type_.asLocal(name)} = {parameters[name]};\n"
        elif isinstance(type_, FPrimitive):
            src += f"{type_.asLocal(name)};\n"
        else:
            src += f"{type_.asLocal('_f2c_arr_' + name)};\n"
            src += f"const {type_.asSpan('f2c::Ptr{_f2c_arr_' + name + '}', name, False, ctx)};\n"

    return src + "\n" + init_src


def translateStmt(stmt: Dict[str, Any], ctx: Context) -> str:
    """
    Dispatch one statement node to the matching translator.
    """
    kind = stmt.get("kind")

    if kind == "assign":
        return f"{translateExpr(stmt['lhs'], ctx)} = {translateExpr(stmt['rhs'], ctx)};\n"

    if kind == "call":
        return translateCallStmt(stmt, ctx)

    if kind == "continue":
        return "// continue\n"

    if kind == "if_stmt":
        cond = translateExpr(stmt["cond"], ctx)
        return f"if ({cond}) {{\n{indent(translateStmt(stmt['stmt'], ctx))}\n}}\n"

    if kind == "if_construct":
        return translateIfConstruct(stmt, ctx)

    if kind == "do":
        return translateDoConstruct(stmt, ctx)

    if kind == "return":
        if ctx.sub_info.isFunction():
            return f"return {ctx.sub_info.func_return_var};\n"
        return "return;\n"

    if kind == "stop":
        return "f2c::trap();\n"

    if kind == "write":
        return "// write statement\n"

    if kind == "data_stmt":
        return translateDataStmt(stmt, ctx)

    ctx.error(f"Unsupported statement: {stmt.get('tag', kind)}")


def translateSubprogram(ctx: Context) -> Tuple[str, str]:
    """
    Translate one Fortran subprogram to a C++ prototype and definition.
    """
    body = ctx.sub_info.body

    param_decls = []
    for param in ctx.sub_info.params:
        assert param.is_const is not None
        param_type = ctx.sub_info.types[param.name]

        if isinstance(param_type, FPrimitive):
            if param.is_const:
                param_decls.append(f"const {param_type.asC()} {param.name}")
            else:
                param_decls.append(f"{param_type.asC()}& {param.name}")
        else:
            const = "const " if param.is_const else ""
            param_decls.append(f"f2c::Ptr<{const}{param_type.inner.asC()}> _f2c_ptr_{param.name}")

    return_type = "void"
    if ctx.sub_info.isFunction():
        return_type = ctx.sub_info.functionType().asC()

    prefix = ctx.info.config.get("func_prefix")
    prefix = f"{prefix} " if prefix else ""

    src_decl = f"static {prefix}{return_type} {ctx.sub_info.name}(\n    "
    src_decl += ",\n    ".join(param_decls) + "\n)"

    src = src_decl + " {\n"
    src += indent(translateSpecificationPart(body.get("decls", []), ctx)) + "\n"
    src += indent(translateExecutionPart(body.get("stmts", []), ctx))
    src += "\n}"

    return src_decl + ";", src


# Names

_RENAME = {
    "atomicadd": "atomicAdd",
}

_CXX_KEYWORDS = {
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
    "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
    "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
    "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double",
    "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float",
    "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace",
    "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
    "private", "protected", "public", "reflexpr", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static", "static_assert",
    "static_cast", "struct", "switch", "synchronized", "template", "this",
    "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union",
    "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while", "xor",
    "xor_eq",
}


def translateName(raw_name: str, ctx: Optional[Context] = None) -> str:
    raw = raw_name.lower()

    if raw in _RENAME:
        return _RENAME[raw]

    if raw in _CXX_KEYWORDS:
        raw = f"_op2k_{raw}"

    if ctx is not None and ctx.sub_info.isFunction() and raw == ctx.sub_info.func_return_var[1:]:
        return ctx.sub_info.func_return_var

    return raw


# Expressions

INTRINSIC_FUNCS = {
    "abs": "f2c::abs",
    "dble": "f2c::dble",
    "int": "f2c::int_",
    "min": "f2c::min",
    "max": "f2c::max",
    "mod": "f2c::mod",
    "nint": "f2c::nint",
    "sign": "f2c::copysign",
    "acos": "f2c::acos",
    "asin": "f2c::asin",
    "atan": "f2c::atan",
    "atan2": "f2c::atan2",
    "cos": "f2c::cos",
    "cosh": "f2c::cosh",
    "exp": "f2c::exp",
    "log": "f2c::log",
    "log10": "f2c::log10",
    "sin": "f2c::sin",
    "sinh": "f2c::sinh",
    "sqrt": "f2c::sqrt",
    "tan": "f2c::tan",
    "tanh": "f2c::tanh",
    "dabs": "f2c::abs",
    "dacos": "f2c::acos",
    "dasin": "f2c::asin",
    "datan": "f2c::atan",
    "dcos": "f2c::cos",
    "dcosh": "f2c::cosh",
    "dexp": "f2c::exp",
    "dint": "f2c::int",
    "dsign": "f2c::copysign",
    "dsin": "f2c::sin",
    "dsinh": "f2c::sinh",
    "dsqrt": "f2c::sqrt",
    "dtan": "f2c::tan",
    "dtanh": "f2c::tanh",
}


def _refNameAndItems(expr: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Unpack a `part_ref` or `funcref` into `(name, items, allow_slice)`.
    """
    if expr["kind"] == "part_ref":
        return expr["name"], expr.get("subscripts", []), True

    return expr["name"], expr.get("args", []), False


def translateArgList(items: List[Dict[str, Any]], ctx: Context, call_target: str) -> str:
    """
    Translate the argument list of a call.
    """
    if call_target == "atomicAdd":
        assert len(items) == 2
        return ", ".join([f"&({translateExpr(items[0], ctx)})", translateExpr(items[1], ctx)])

    target_sub = ctx.info.subprograms[call_target]

    args = []
    for item, target_type in zip(items, [target_sub.types[p.name] for p in target_sub.params]):
        if isinstance(target_type, FArray) and item.get("kind") in ("part_ref", "funcref"):
            name = translateName(item["name"], ctx)

            if name in ctx.info.functionNames():
                ctx.error("Unsupported function return as array argument")

            if name not in ctx.sub_info.types:
                ctx.error("Unsupported const part-ref as arg")

            subs = item.get("subscripts") if item["kind"] == "part_ref" else item.get("args", [])
            if any(s.get("kind") == "triplet" for s in subs):
                ctx.error("Unsupported slice part-ref as arg")

            arg = f"{name}.ptr_at({', '.join(translateExpr(s, ctx) for s in subs)})"
        elif isinstance(target_type, FArray) and item.get("kind") != "name":
            ctx.error("Unsupported expression passed to array argument")
        else:
            arg = translateExpr(item, ctx)

        args.append(arg)

    return ", ".join(args)


def translateBinaryExpr(expr: Dict[str, Any], ctx: Context) -> str:
    """
    Translate a binary operator.
    """
    op = expr["op"]
    left = translateExpr(expr["left"], ctx)
    right = translateExpr(expr["right"], ctx)

    if op == "**":
        return f"f2c::pow({left}, {right})"

    if op == "//":
        ctx.error("Unsupported character concatenation operator")

    return f"{left} {op} {right}"


def translateExpr(expr: Dict[str, Any], ctx: Context) -> str:
    """
    Translate an expression JSON node to C++.
    """
    kind = expr.get("kind")

    if kind == "name":
        return translateName(expr["value"], ctx)

    if kind in ("part_ref", "funcref"):
        return translateRef(expr, ctx)

    if kind == "paren":
        return f"({translateExpr(expr['expr'], ctx)})"

    if kind == "unary":
        return f"{expr['op']}{translateExpr(expr['expr'], ctx)}"

    if kind == "binary":
        return translateBinaryExpr(expr, ctx)

    if kind == "int_lit":
        return translateIntLiteral(expr, ctx)

    if kind == "real_lit":
        return translateRealLiteral(expr, ctx)

    if kind == "logical_lit":
        return "true" if expr.get("value") else "false"

    if kind == "char_lit":
        return f'"{expr.get("value", "")}"'

    if kind == "unsupported":
        tag = expr.get("tag", "expression")
        source = expr.get("source")
        suffix = f": {source}" if source else ""
        ctx.error(f"Unsupported Fortran construct ({tag}){suffix}")

    ctx.error(f"Unsupported expression kind: {kind}")


def translateIntLiteral(expr: Dict[str, Any], ctx: Context) -> str:
    if expr.get("kind_text") is not None:
        ctx.error(f"Unsupported int literal kind specifier: {expr['kind_text']}")

    return expr["text"]


def translateIntrinsicCall(func_name: str, items: List[Dict[str, Any]], ctx: Context) -> str:
    if func_name == "real":
        if len(items) != 2:
            ctx.error("Expected REAL(x, kind)")

        kind = translateExpr(items[1], ctx)
        if kind == "rk4":
            cast = "float"
        elif kind == "rk8":
            cast = "double"
        else:
            ctx.error(f"Unsupported REAL() kind: {kind}")

        return f"({cast})({translateExpr(items[0], ctx)})"

    return f"{INTRINSIC_FUNCS[func_name]}({', '.join(translateExpr(item, ctx) for item in items)})"


def translateRealLiteral(expr: Dict[str, Any], ctx: Context) -> str:
    kind_text = expr.get("kind_text")
    kind_norm = kind_text.upper() if (kind_text is not None and not kind_text.isdigit()) else kind_text

    kind_spec_is_float = {
        None: True,
        "4": True,
        "RK4": True,
        "8": False,
        "RK": False,
        "RK8": False,
    }

    if kind_norm not in kind_spec_is_float:
        ctx.error(f"Unsupported real literal kind specifier: {kind_text}")

    is_float = kind_spec_is_float[kind_norm]
    raw = expr["text"]
    raw_upper = raw.upper()

    if "E" in raw_upper:
        is_float = True
        raw = re.sub(r"[eE]", "e", raw)
    elif "D" in raw_upper:
        is_float = False
        raw = re.sub(r"[dD]", "e", raw)

    return raw + "f" if is_float else raw


def translateRef(expr: Dict[str, Any], ctx: Context) -> str:
    """
    Translate an array index, function call, or const lookup.
    """
    raw_name, items, allow_slice = _refNameAndItems(expr)
    lname_raw = raw_name.lower()

    if lname_raw == "real" or lname_raw in INTRINSIC_FUNCS:
        return translateIntrinsicCall(lname_raw, items, ctx)

    name = translateName(raw_name, ctx)

    if name in ctx.info.functionNames():
        return f"{name}({translateArgList(items, ctx, name)})"

    if len(items) == 0:
        ctx.error(f"Reference has no subscript/argument list: {raw_name}")

    array_type = ctx.lookupType(name)
    if array_type is None:
        ctx.error(f"Could not find type of reference: {raw_name}")

    if name in ctx.sub_info.types:
        is_slice = allow_slice and any(s.get("kind") == "triplet" for s in items)

        if not is_slice:
            return f"{name}({', '.join(translateExpr(s, ctx) for s in items)})"

        if len(items) != len(array_type.shape):
            ctx.error(f"Number of subscripts doesn't match array type {array_type}")

        extents = []
        for shape, sub in zip(array_type.shape, items):
            if sub.get("kind") != "triplet":
                idx = translateExpr(sub, ctx)
                extents.append(f"f2c::Extent{{{idx}, {idx}}}")
                continue

            if sub.get("stride") is not None:
                ctx.error("Unsupported stride in subscript triplet")

            lb = translateExpr(sub["lower"], ctx) if sub.get("lower") is not None else shape[0]
            ub = translateExpr(sub["upper"], ctx) if sub.get("upper") is not None else shape[1]
            extents.append(f"f2c::Extent{{{lb}, {ub}}}")

        return f"{name}.slice({', '.join(extents)})"

    sizes = []
    for lb, ub in array_type.shape:
        if lb == "1":
            sizes.append(f"({ub})")
        else:
            sizes.append(f"(1 + {ub} - ({lb}))")

    if array_type.shape[0][0] == "1":
        index = f"{translateExpr(items[0], ctx)}"
    else:
        index = f"({translateExpr(items[0], ctx)} + 1 - ({array_type.shape[0][0]}))"

    for i, extra in enumerate(items[1:], start=1):
        index += f" + ({translateExpr(extra, ctx)} - ({array_type.shape[i][0]})) * {'*'.join(sizes[:i])}"

    return f"{name}[({index}) - 1]"
