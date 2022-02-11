import dataclasses
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from clang.cindex import Config, Cursor, CursorKind, Index, TranslationUnit

import op as OP
from store import Kernel, Location, ParseError, Program
from util import enumRegex, safeFind


def parseKernel(path: Path, name: str) -> Kernel:
    translation_unit = Index.create().parse(path)
    nodes = translation_unit.cursor.get_children()

    node = safeFind(nodes, lambda n: n.kind == CursorKind.FUNCTION_DECL and n.spelling == name)
    if not node:
        raise ParseError(f"failed to locate kernel function {name}", parseLocation(node))

    params = []
    for n in node.get_children():
        if n.kind != CursorKind.PARM_DECL:
            continue

        param_type = n.type.get_canonical()
        while param_type.get_pointee().spelling:
            param_type = param_type.get_pointee()

        param = (n.spelling, parseType(param_type.spelling, parseLocation(n)))
        params.append(param)

    return Kernel(name, path, params)


def parseProgram(path: Path, include_dirs: Set[Path], soa: bool) -> Program:
    args = [f"-I{dir}" for dir in include_dirs]
    translation_unit = Index.create().parse(path, args=args, options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)

    error = next(iter(translation_unit.diagnostics), None)
    if error:
        raise ParseError(error.spelling, parseLocation(error))

    program = Program(path)

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
        parseNode(node, translation_unit.cursor, macros, program)

    return program


def parseNode(node: Cursor, parent: Cursor, macros: Dict[Location, str], program: Program) -> None:
    if node.kind == CursorKind.CALL_EXPR:
        parseCall(node, parent, macros, program)

    for child in node.get_children():
        parseNode(child, node, macros, program)


def parseCall(node: Cursor, parent: Cursor, macros: Dict[Location, str], program: Program) -> None:
    name = node.spelling
    args = list(node.get_arguments())
    loc = parseLocation(node)

    if name == "op_init":
        program.recordInit(loc)

    elif name == "op_decl_set":
        ptr = parsePtr(parent)
        program.sets.append(parseSet(args, ptr, loc))

    elif name == "op_decl_map":
        ptr = parsePtr(parent)
        program.maps.append(parseMap(args, ptr, loc))

    elif name == "op_decl_dat":
        ptr = parsePtr(parent)
        program.dats.append(parseDat(args, ptr, loc))

    elif name == "op_decl_const":
        program.consts.append(parseConst(args, loc))

    elif name == "op_par_loop":
        program.loops.append(parseLoop(args, loc, macros))

    elif name == "op_exit":
        program.recordExit()


def parseSet(args: List[Cursor], ptr: str, loc: Location) -> OP.Set:
    if len(args) != 2:
        raise ParseError("incorrect number of args passed to op_decl_set", loc)

    return OP.Set(loc, ptr)


def parseMap(args: List[Cursor], ptr: str, loc: Location) -> OP.Map:
    if len(args) != 5:
        raise ParseError("incorrect number of args passed to op_decl_map", loc)

    from_set_ptr = parseIdentifier(args[0])
    to_set_ptr = parseIdentifier(args[1])
    dim = parseIntLit(args[2])

    return OP.Map(loc, from_set_ptr, to_set_ptr, dim, ptr)


def parseDat(args: List[Cursor], ptr: str, loc: Location) -> OP.Dat:
    if len(args) != 5:
        raise ParseError("incorrect number of args passed to op_decl_dat", loc)

    set_ptr = parseIdentifier(args[0])
    dim = parseIntLit(args[1])
    typ = parseType(parseStringLit(args[2]), loc)

    return OP.Dat(loc, set_ptr, dim, typ, ptr)


def parseConst(args: List[Cursor], loc: Location) -> OP.Const:
    if len(args) != 3:
        raise ParseError("incorrect number of args passed to op_decl_const", loc)

    # TODO dim may not be literal
    dim = parseIntLit(args[0])
    typ = parseType(parseStringLit(args[1]), loc)
    ptr = parseIdentifier(args[2])

    return OP.Const(loc, dim, typ, ptr)


def parseLoop(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> OP.Loop:
    if len(args) < 3:
        raise ParseError("incorrect number of args passed to op_par_loop")

    kernel = parseIdentifier(args[0])
    set_ptr = parseIdentifier(args[2])

    loop_args = []
    for node in args[3:]:
        node = descend(descend(node))

        name = node.spelling

        arg_loc = parseLocation(node)
        arg_args = list(node.get_arguments())

        if name == "op_arg_dat":
            loop_args.append(parseArgDat(arg_args, arg_loc, macros))

        elif name == "op_opt_arg_dat":
            loop_args.append(parseOptArgDat(arg_args, arg_loc, macros))

        elif name == "op_arg_gbl":
            loop_args.append(parseArgGbl(arg_args, arg_loc, macros))

        elif name == "op_opt_arg_gbl":
            loop_args.append(parseOptArgGbl(arg_args, arg_loc, macros))

        else:
            raise ParseError(f"invalid loop argument {name}", parseLocation(node))

    return OP.Loop(loc, kernel, set_ptr, loop_args)


def parseArgDat(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> OP.ArgDat:
    if len(args) != 6:
        raise ParseError("incorrect number of args passed to op_arg_dat", loc)

    dat_ptr = parseIdentifier(args[0])

    map_idx = parseIntLit(args[1])
    map_ptr = None if macros.get(parseLocation(args[2])) == "OP_ID" else parseIdentifier(args[2])

    dat_dim = parseIntLit(args[3])
    dat_typ = parseType(parseStringLit(args[4]), loc)

    access_type = parseAccessType(args[5], loc, macros)

    return OP.ArgDat(loc, access_type, None, dat_ptr, dat_dim, dat_typ, map_ptr, map_idx)


def parseOptArgDat(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> OP.ArgDat:
    if len(args) != 7:
        ParseError("incorrect number of args passed to op_opt_arg_dat", loc)

    opt = parseIdentifier(args[0])
    dat = parseArgDat(args[1:], loc, macros)

    return dataclasses.replace(dat, opt=opt)


def parseArgGbl(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> OP.ArgGbl:
    if len(args) != 4:
        raise ParseError("incorrect number of args passed to op_arg_gbl", loc)

    ptr = parseIdentifier(args[0])
    dim = parseIntLit(args[1])
    typ = parseType(parseStringLit(args[2]), loc)

    access_type = parseAccessType(args[3], loc, macros)

    return OP.ArgGbl(loc, access_type, None, ptr, dim, typ)


def parseOptArgGbl(args: List[Cursor], loc: Location, macros: Dict[Location, str]) -> OP.ArgGbl:
    if len(args) != 5:
        raise ParseError("incorrect number of args passed to op_opt_arg_gbl", loc)

    opt = parseIdentifier(args[0])
    dat = parseArgGbl(args[1:], loc, macros)

    return dataclasses.replace(dat, opt=opt)


def parsePtr(node: Cursor) -> str:
    if node.kind == CursorKind.VAR_DECL:
        return node.spelling

    if node.kind == CursorKind.BINARY_OPERATOR:
        children = list(node.get_children())
        tokens = list(node.get_tokens())
        operator = tokens[len(list(children[0].get_tokens()))].spelling

        if operator != "=":
            raise ParseError(f"unexpected binary operator {operator}", parseLocation(node))

        return parseIdentifier(children[0])

    raise ParseError(f"expected variable declaration or assignment", parseLocation(node))


def parseIdentifier(node: Cursor) -> str:
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


def parseIntLit(node: Cursor) -> int:
    coeff = 1

    if node.kind == CursorKind.UNARY_OPERATOR and next(node.get_tokens()).spelling == "-":
        coeff = -1
        node = descend(node)

    if node.kind != CursorKind.INTEGER_LITERAL:
        raise ParseError("expected integer literal", parseLocation(node))

    return coeff * int(next(node.get_tokens()).spelling)


def parseStringLit(node: Cursor) -> str:
    if node.kind != CursorKind.UNEXPOSED_EXPR:
        raise ParseError("expected string literal")

    node = descend(node)
    if node.kind != CursorKind.STRING_LITERAL:
        raise ParseError("expected string literal")

    return node.spelling[1:-1]


def parseAccessType(node: Cursor, loc: Location, macros: Dict[Location, str]) -> OP.AccessType:
    access_type_str = macros.get(parseLocation(node))

    if access_type_str not in OP.AccessType.values():
        raise ParseError(
            f"invalid access type {access_type_str}, expected one of {', '.join(OP.AccessType.values())}", loc
        )

    return OP.AccessType(access_type_str)


def parseType(typ: str, loc: Location) -> OP.Type:
    typ_clean = typ.strip()
    typ_clean = re.sub(r"\s*const\s*", "", typ_clean)

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
        return typ_map[typ_clean]

    raise ParseError(f'unable to parse type "{typ}"', loc)


def parseLocation(node: Cursor) -> Location:
    return Location(node.location.file.name, node.location.line, node.location.column)


def descend(node: Cursor) -> Optional[Cursor]:
    return next(node.get_children(), None)
