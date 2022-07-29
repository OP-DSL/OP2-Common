import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from clang.cindex import Cursor, CursorKind, TranslationUnit, TypeKind, conf

import op as OP
from store import Kernel, Location, ParseError, Program, Function, Type
from util import safeFind


def parseKernel(translation_unit: TranslationUnit, name: str, path: Path) -> Optional[Kernel]:
    nodes = translation_unit.cursor.get_children()

    node = safeFind(nodes, lambda n: n.kind == CursorKind.FUNCTION_DECL and n.spelling == name)
    if not node:
        return None

    params = []
    for n in node.get_children():
        if n.kind != CursorKind.PARM_DECL:
            continue

        param_type = n.type.get_canonical()
        while param_type.get_pointee().spelling:
            param_type = param_type.get_pointee()

        typ, _ = parseType(param_type.spelling, parseLocation(n))
        params.append((n.spelling, typ))

    return Kernel(name, path, params)


def parseMeta(node: Cursor, program: Program) -> None:
    if node.kind == CursorKind.TYPE_REF:
        parseTypeRef(node, program)

    if node.kind == CursorKind.FUNCTION_DECL:
        parseFunction(node, program)

    for child in node.get_children():
        parseMeta(child, program)


def parseTypeRef(node: Cursor, program: Program) -> None:
    node = node.get_definition()

    if node is None or Path(str(node.location.file)) != program.path:
        return

    matching_entities = program.findEntities(node.spelling)
    for entity in matching_entities:
        if entity.ast == node:
            return

    typ = Type(node.spelling, node, program)

    for n in node.walk_preorder():
        if n.kind != CursorKind.CALL_EXPR and n.kind != CursorKind.TYPE_REF:
            continue

        n = n.get_definition()

        if n is None or Path(str(n.location.file)) != program.path:
            continue

        typ.depends.add(n.spelling)

    program.entities.append(typ)


def parseFunction(node: Cursor, program: Program) -> None:
    node = node.get_definition()

    if node is None or Path(str(node.location.file)) != program.path:
        return

    matching_entities = program.findEntities(node.spelling)
    for entity in matching_entities:
        if entity.ast == node:
            return

    function = Function(node.spelling, node, program)

    for n in node.get_children():
        if n.kind != CursorKind.PARM_DECL:
            continue

        param_type = n.type.get_canonical()

        typ, _ = parseType(param_type.spelling, parseLocation(n), True)
        function.parameters.append((n.spelling, typ))

    for n in node.walk_preorder():
        if n.kind != CursorKind.CALL_EXPR and n.kind != CursorKind.TYPE_REF:
            continue

        n = n.get_definition()

        if n is None or Path(str(n.location.file)) != program.path:
            continue

        function.depends.add(n.spelling)

    program.entities.append(function)


def parseLoops(translation_unit: TranslationUnit, program: Program) -> None:
    macros: Dict[Location, str] = {}
    nodes: List[Cursor] = []

    for node in translation_unit.cursor.get_children():
        if node.kind == CursorKind.MACRO_DEFINITION:
            continue

        if node.location.file.name != translation_unit.spelling:
            continue

        if node.kind == CursorKind.MACRO_INSTANTIATION:
            macros[parseLocation(node)] = node.spelling
            continue

        nodes.append(node)

    for node in nodes:
        for child in node.walk_preorder():
            if child.kind == CursorKind.CALL_EXPR:
                parseCall(child, macros, program)

    return program


def parseCall(node: Cursor, macros: Dict[Location, str], program: Program) -> None:
    name = node.spelling
    args = list(node.get_arguments())
    loc = parseLocation(node)

    if name == "op_decl_const":
        program.consts.append(parseConst(args, loc))

    elif name == "op_par_loop":
        program.loops.append(parseLoop(args, loc, macros))


def parseConst(args: List[Cursor], loc: Location) -> OP.Const:
    if len(args) != 3:
        raise ParseError("incorrect number of args passed to op_decl_const", loc)

    # TODO dim may not be literal
    dim = parseIntExpression(args[0])
    typ, _ = parseType(parseStringLit(args[1]), loc)
    ptr = parseIdentifier(args[2], raw=False)

    return OP.Const(loc, ptr, dim, typ)


def parseLoop(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> OP.Loop:
    if len(args) < 3:
        raise ParseError("incorrect number of args passed to op_par_loop")

    kernel = parseIdentifier(args[0])
    loop = OP.Loop(loc, kernel)

    for node in args[3:]:
        node = descend(descend(node))

        name = node.spelling

        arg_loc = parseLocation(node)
        arg_args = list(node.get_arguments())

        if name == "op_arg_dat":
            parseArgDat(loop, False, arg_args, arg_loc, macros)

        elif name == "op_opt_arg_dat":
            parseArgDat(loop, True, arg_args, arg_loc, macros)

        elif name == "op_arg_gbl":
            parseArgGbl(loop, False, arg_args, arg_loc, macros)

        elif name == "op_opt_arg_gbl":
            parseArgGbl(loop, True, arg_args, arg_loc, macros)

        else:
            raise ParseError(f"invalid loop argument {name}", parseLocation(node))

    return loop


def parseArgDat(loop: OP.Loop, opt: bool, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if not opt and len(args) != 6:
        raise ParseError("incorrect number of args passed to op_arg_dat", loc)

    if opt and len(args) != 7:
        raise ParseError("incorrect number of args passed to op_opt_arg_dat", loc)

    if opt:
        args = args[1:]

    dat_ptr = parseIdentifier(args[0])

    map_idx = parseIntExpression(args[1])
    map_ptr = None if macros.get(parseLocation(args[2])) == "OP_ID" else parseIdentifier(args[2])

    dat_dim = parseIntExpression(args[3])
    dat_typ, dat_soa = parseType(parseStringLit(args[4]), loc)

    access_type = parseAccessType(args[5], loc, macros)

    loop.addArgDat(loc, dat_ptr, dat_dim, dat_typ, dat_soa, map_ptr, map_idx, access_type, opt)


def parseArgGbl(loop: OP.Loop, opt: bool, args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> None:
    if not opt and len(args) != 4:
        raise ParseError("incorrect number of args passed to op_arg_gbl", loc)

    if opt and len(args) != 5:
        raise ParseError("incorrect number of args passed to op_opt_arg_gbl", loc)

    ptr = parseIdentifier(args[0])
    dim = parseIntExpression(args[1])
    typ, _ = parseType(parseStringLit(args[2]), loc)

    access_type = parseAccessType(args[3], loc, macros)

    loop.addArgGbl(loc, ptr, dim, typ, access_type, opt)


def parseIdentifier(node: Cursor, raw: bool = True) -> str:
    if raw:
        return "".join([t.spelling for t in node.get_tokens()])

    while node.kind == CursorKind.CSTYLE_CAST_EXPR:
        node = list(node.get_children())[1]

    if node.kind == CursorKind.UNEXPOSED_EXPR:
        node = descend(node)

    if node.kind == CursorKind.UNARY_OPERATOR and next(node.get_tokens()).spelling in (
        "&",
        "*",
    ):
        node = descend(node)

    if node.kind == CursorKind.GNU_NULL_EXPR:
        raise ParseError("expected identifier, found NULL", parseLocation(node))

    if node.kind != CursorKind.DECL_REF_EXPR:
        raise ParseError("expected identifier", parseLocation(node))

    return node.spelling


def parseIntExpression(node: Cursor) -> int:
    if node.type.kind != TypeKind.INT:
        raise ParseError("expected int expression", parseLocation(node))

    eval_result = conf.lib.clang_Cursor_Evaluate(node)
    val = conf.lib.clang_EvalResult_getAsInt(eval_result)
    conf.lib.clang_EvalResult_dispose(eval_result)

    return val


def parseStringLit(node: Cursor) -> str:
    if node.kind != CursorKind.UNEXPOSED_EXPR:
        raise ParseError("expected string literal")

    node = descend(node)
    if node.kind != CursorKind.STRING_LITERAL:
        raise ParseError("expected string literal")

    return node.spelling[1:-1]


def parseAccessType(node: Cursor, loc: Location, macros: Dict[Location, str]) -> OP.AccessType:
    access_type_raw = parseIntExpression(node)

    if access_type_raw not in OP.AccessType.values():
        raise ParseError(
            f"invalid access type {access_type_raw}, expected one of {', '.join(OP.AccessType.values())}", loc
        )

    return OP.AccessType(access_type_raw)


def parseType(typ: str, loc: Location, include_custom=False) -> Tuple[OP.Type, bool]:
    typ_clean = typ.strip()
    typ_clean = re.sub(r"\s*const\s*", "", typ_clean)

    soa = False
    if re.search(r":soa", typ_clean):
        soa = True

    typ_clean = re.sub(r"\s*:soa\s*", "", typ_clean)

    typ_map = {
        "int": OP.Int(True, 32),
        "uint": OP.Int(False, 32),
        "ll": OP.Int(True, 64),
        "ull": OP.Int(False, 64),
        "float": OP.Float(32),
        "double": OP.Float(64),
        "bool": OP.Bool(),
    }

    if typ_clean in typ_map:
        return typ_map[typ_clean], soa

    if include_custom:
        return typ_clean, soa

    raise ParseError(f'unable to parse type "{typ}"', loc)


def parseLocation(node: Cursor) -> Location:
    return Location(node.location.file.name, node.location.line, node.location.column)


def descend(node: Cursor) -> Optional[Cursor]:
    return next(node.get_children(), None)
