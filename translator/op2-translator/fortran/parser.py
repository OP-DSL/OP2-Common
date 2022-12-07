import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from store import Function, Location, ParseError, Program


def parseParamType(path: Path, subroutine: f2003.Subroutine_Subprogram, param: str) -> OP.Type:
    type_declaration = None

    for candidate_declaration in fpu.walk(subroutine, f2003.Type_Declaration_Stmt):
        entity_decl_list = fpu.get_child(candidate_declaration, f2003.Entity_Decl_List)
        loc = Location(path, candidate_declaration.item.span[0], 0)

        for entity_decl in entity_decl_list.items:
            if parseIdentifier(entity_decl.items[0], loc) == param:
                type_declaration = candidate_declaration
                break

    if type_declaration is None:
        raise ParseError(f"failed to locate type declaration for kernel parameter {param} in {path}: {subroutine}")

    loc = Location(path, type_declaration.item.span[0], 0)
    type_spec = fpu.get_child(type_declaration, f2003.Intrinsic_Type_Spec)

    if type_spec is None:
        type_spec = fpu.get_child(type_declaration, f2003.Declaration_Type_Spec)

    return parseType(type_spec.tofortran(), loc, True)[0]


def parseProgram(ast: f2003.Program, source: str, path: Path) -> Program:
    program = Program(path, ast, source)
    parseNode(ast, program, Location(str(program.path), 0, 0))

    return program


def parseNode(node: Any, program: Program, loc: Location) -> None:
    if isinstance(node, f2003.Call_Stmt):
        parseCall(node, program, loc)

    if isinstance(node, f2003.Subroutine_Subprogram):
        parseSubroutine(node, program, loc)

    if not isinstance(node, f2003.Base):
        return

    for child in node.children:
        if child is None:
            continue

        child_loc = loc

        if hasattr(child, "item") and child.item is not None:
            child_loc = Location(str(program.path), child.item.span[0], 0)

        parseNode(child, program, child_loc)


def parseSubroutine(node: f2003.Subroutine_Subprogram, program: Program, loc: Location) -> None:
    subroutine_statement = fpu.get_child(node, f2003.Subroutine_Stmt)
    name_node = fpu.get_child(subroutine_statement, f2003.Name)

    name = parseIdentifier(name_node, loc)
    function = Function(name, node, program)

    function.parameters = parseSubroutineParameters(program.path, node)

    # for param in param_identifiers:
    #     typ = parseParamType(program.path, node, param)
    #     function.parameters.append((param, typ))

    for call in fpu.walk(node, f2003.Call_Stmt):
        name_node = fpu.get_child(call, f2003.Name)

        if name_node is None:  # Happens for Procedure_Designator (stuff like op%access_i4...)
            continue

        name = parseIdentifier(name_node, loc)
        function.depends.add(name)

    program.entities.append(function)


def parseSubroutineParameters(path: Path, subroutine: f2003.Subroutine_Subprogram) -> List[str]:
    subroutine_statement = fpu.get_child(subroutine, f2003.Subroutine_Stmt)
    arg_list = fpu.get_child(subroutine_statement, f2003.Dummy_Arg_List)

    if arg_list is None:
        return []

    loc = Location(str(path), subroutine_statement.item.span[0], 0)

    parameters = []
    for item in arg_list.items:
        parameters.append(parseIdentifier(item, loc))

    return parameters


def parseCall(node: f2003.Call_Stmt, program: Program, loc: Location) -> None:
    name_node = fpu.get_child(node, f2003.Name)
    if name_node is None:  # Happens for Procedure_Designator (stuff like op%access_i4...)
        return

    name = parseIdentifier(name_node, loc)
    args = fpu.get_child(node, f2003.Actual_Arg_Spec_List)

    if name == "op_decl_const":
        program.consts.append(parseConst(args, loc))

    elif re.match(r"op_par_loop_\d+", name):
        program.loops.append(parseLoop(args, loc))


def parseConst(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Const:
    if args is None or len(args.items) != 3:
        raise ParseError("incorrect number of arguments for op_decl_const", loc)

    ptr = parseIdentifier(args.items[0], loc)
    dim = parseIntLiteral(args.items[1], loc)  # TODO: Might not be an integer literal?

    typ_str = parseStringLiteral(args.items[2], loc).strip().lower()
    typ = parseType(typ_str, loc)[0]

    return OP.Const(loc, ptr, dim, typ)


def parseLoop(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Loop:
    if args is None or len(args.items) < 3:
        raise ParseError("incorrect number of arguments for op_par_loop", loc)

    kernel = parseIdentifier(args.items[0], loc)
    loop = OP.Loop(loc, kernel)

    for arg_node in args.items[2:]:
        if type(arg_node) is not f2003.Structure_Constructor:
            if type(arg_node) is not f2003.Part_Ref:
                raise ParseError(f"unable to parse op_par_loop argument: {arg_node}", loc)

            name = parseIdentifier(fpu.get_child(arg_node, f2003.Name), loc)
            arg_args = fpu.get_child(arg_node, f2003.Section_Subscript_List)

            if name == "op_arg_idx":
                parseArgIdx(loop, arg_args, loc)

            else:
                raise ParseError(f"invalid loop argument {arg_node}", loc)

            continue

        name = parseIdentifier(fpu.get_child(arg_node, f2003.Type_Name), loc)
        arg_args = fpu.get_child(arg_node, f2003.Component_Spec_List)

        if name == "op_arg_dat":
            parseArgDat(loop, False, arg_args, loc)

        elif name == "op_opt_arg_dat":
            parseArgDat(loop, True, arg_args, loc)

        elif name == "op_arg_gbl":
            parseArgGbl(loop, False, arg_args, loc)

        elif name == "op_opt_arg_gbl":
            parseArgGbl(loop, True, arg_args, loc)

        else:
            raise ParseError(f"invalid loop argument {arg_node}", loc)

    return loop


def parseArgDat(loop: OP.Loop, opt: bool, args: Optional[f2003.Component_Spec_List], loc: Location) -> None:
    if args is None or (not opt and len(args.items) != 6):
        raise ParseError("incorrect number of arguments for op_arg_dat", loc)

    if args is None or (opt and len(args.items) != 7):
        raise ParseError("incorrect number of arguments for op_opt_arg_dat", loc)

    args_list = args.items

    if opt:
        args_list = args_list[1:]

    dat_ptr = parseIdentifier(args_list[0], loc)
    map_idx = parseIntLiteral(args_list[1], loc, True)
    map_ptr: Optional[str] = parseIdentifier(args_list[2], loc)

    if map_ptr == "OP_ID":
        map_ptr = None

    dat_dim = parseIntLiteral(args_list[3], loc, True)

    dat_typ, dat_soa = parseType(parseStringLiteral(args_list[4], loc), loc)
    access_type = parseAccessType(args_list[5], loc)

    loop.addArgDat(loc, dat_ptr, dat_dim, dat_typ, dat_soa, map_ptr, map_idx, access_type, opt)


def parseArgGbl(loop: OP.Loop, opt: bool, args: Optional[f2003.Component_Spec_List], loc: Location) -> None:
    if args is None or (not opt and len(args.items) != 4):
        raise ParseError("incorrect number of arguments for op_arg_gbl", loc)

    if args is None or (opt and len(args.items) != 5):
        raise ParseError("incorrect number of arguments for op_opt_arg_gbl", loc)

    args_list = args.items

    if opt:
        args_list = args_list[1:]

    ptr = parseIdentifier(args_list[0], loc)
    dim = parseIntLiteral(args_list[1], loc, True)
    typ = parseType(parseStringLiteral(args_list[2], loc), loc)[0]
    access_type = parseAccessType(args_list[3], loc)

    loop.addArgGbl(loc, ptr, dim, typ, access_type, opt)


def parseArgIdx(loop: OP.Loop, args: Optional[f2003.Component_Spec_List], loc: Location) -> None:
    if args is None or len(args.items) != 2:
        raise ParseError("incorrect number of arguments for op_arg_idx", loc)

    map_idx = parseIntLiteral(args.items[0], loc, True)
    map_ptr: Optional[str] = parseIdentifier(args.items[1], loc)

    if map_ptr == "OP_ID":
        map_ptr = None

    loop.addArgIdx(loc, map_ptr, map_idx)


def parseIdentifier(node: Any, loc: Location) -> str:
    # if not hasattr(node, "string"):
    #    raise ParseError(f"Unable to parse identifier for node: {node}", loc)

    return node.string


# literal_aliases = {
#     "npdes": 6,
#     "npdesdpl": 4,
#     "mpdes": 1000,
#     "ntqmu": 3,
#     # Global dims - known
#     "nzone": 0,
#     "ngrp": 0,
#     "mints": 22,
#     "igrp": 1000,
#     "mpdesdpl": 40,
#     "mspl": 500,
#     # Global dims - unknown
#     "ncline": 64,
#     "ncfts": 64,
#     "ncftm": 64,
#     "ncline": 64,
#     "ntline": 64,
# }

# literal_aliases["njaca"] = literal_aliases["npdes"] - 5

# literal_aliases["nspdes"] = literal_aliases["npdes"]
# literal_aliases["njacs"] = literal_aliases["nspdes"] - 5

# literal_aliases["ngrad"] = 3 * literal_aliases["npdes"]
# literal_aliases["ndets"] = 6 + 3 * (literal_aliases["npdes"] - 4)


def parseIntLiteral(node: Any, loc: Location, optional: bool = False) -> Optional[int]:
    if type(node) is f2003.Parenthesis:
        return parseIntLiteral(node.items[1], loc, optional)

    if type(node) is f2003.Signed_Int_Literal_Constant or type(node) is f2003.Int_Literal_Constant:
        return int(node.items[0])

    if issubclass(type(node), f2003.UnaryOpBase):
        val = parseIntLiteral(node.items[1], loc, optional)
        op = node.items[0]

        if val is None:
            return None

        if op == "+":
            return val
        elif op == "-":
            return -val

    if issubclass(type(node), f2003.BinaryOpBase):
        lhs = parseIntLiteral(node.items[0], loc, optional)
        rhs = parseIntLiteral(node.items[2], loc, optional)

        op = node.items[1]

        if lhs is None or rhs is None:
            return None

        if op == "+":
            return lhs + rhs
        elif op == "-":
            return lhs - rhs
        elif op == "*":
            return lhs * rhs
        elif op == "/":
            return int(lhs / rhs)
        elif op == "**":
            return int(lhs**rhs)

    #    if type(node) is f2003.Name:
    #        ident = parseIdentifier(node, loc)

    #        if ident in literal_aliases:
    #            return literal_aliases[ident]

    if optional:
        return None

    raise ParseError(f"unable to parse int literal: {node}", loc)


def parseStringLiteral(node: Any, loc: Location) -> str:
    if type(node) is not f2003.Char_Literal_Constant:
        raise ParseError("unable to parse string literal", loc)

    return str(node.items[0][1:-1])


def parseAccessType(node: Any, loc: Location) -> OP.AccessType:
    access_type_str = parseIdentifier(node, loc)

    access_type_map = {"OP_READ": 0, "OP_WRITE": 1, "OP_RW": 2, "OP_INC": 3, "OP_MIN": 4, "OP_MAX": 5}

    if access_type_str not in access_type_map:
        raise ParseError(
            f"invalid access type {access_type_str}, expected one of {', '.join(access_type_map.keys())}", loc
        )

    access_type_raw = access_type_map[access_type_str]
    return OP.AccessType(access_type_raw)


def parseType(typ: str, loc: Location, include_custom: bool = False) -> Tuple[OP.Type, bool]:
    typ_clean = typ.strip().lower()
    typ_clean = re.sub(r"\s*kind\s*=\s*", "", typ_clean)

    soa = False
    if re.search(r":soa", typ_clean):
        soa = True

    typ_clean = re.sub(r"\s*:soa\s*", "", typ_clean)
    typ_clean = re.sub(r"\s*", "", typ_clean)

    def mk_type_regex(t, k):
        return rf"{t}(?:\((?:kind=)?{k}\)|\*{k})?$"

    aliases = {
        "i4": OP.Int(True, 32),
        "i8": OP.Int(True, 64),
        "r4": OP.Float(32),
        "r8": OP.Float(64),
    }

    if typ_clean in aliases:
        return aliases[typ_clean], soa

    integer_match = re.match(mk_type_regex("integer", "(?:ik)?(4|8)"), typ_clean)
    if integer_match is not None:
        size = 32

        groups = integer_match.groups()
        size_match = groups[0] or groups[1]
        if size_match is not None:
            size = int(size_match) * 8

        return OP.Int(True, size), soa

    real_match = re.match(mk_type_regex("real", "(?:rk)?(4|8)"), typ_clean)
    if real_match is not None:
        size = 32

        groups = real_match.groups()
        size_match = groups[0] or groups[1]
        if size_match is not None:
            size = int(size_match) * 8

        return OP.Float(size), soa

    logical_match = re.match(mk_type_regex("logical", "(?:lk)?(1|2|4)?"), typ_clean)
    if logical_match is not None:
        return OP.Bool(), soa

    if include_custom:
        return OP.Custom(typ_clean), soa

    raise ParseError(f'unable to parse type "{typ}"', loc)
