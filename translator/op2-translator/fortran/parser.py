import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from store import Function, Kernel, Location, ParseError, Program


def parseKernel(ast: f2003.Program, name: str, path: Path) -> Optional[Kernel]:
    subroutine = findSubroutine(path, ast, name)
    if subroutine is None:
        return None

    param_identifiers = parseSubroutineParameters(path, subroutine)

    params = []
    for param in param_identifiers:
        typ = parseParamType(path, subroutine, param)
        params.append((param, typ))

    return Kernel(name, path, params)


def findSubroutine(path: Path, ast: f2003.Program, name: str) -> Optional[f2003.Subroutine_Subprogram]:
    subroutine = None

    for candidate_routine in fpu.walk(ast, f2003.Subroutine_Subprogram):
        subroutine_statement = fpu.get_child(candidate_routine, f2003.Subroutine_Stmt)
        name_node = fpu.get_child(subroutine_statement, f2003.Name)
        loc = Location(str(path), subroutine_statement.item.span[0], 0)

        if parseIdentifier(name_node, loc) == name:
            subroutine = candidate_routine
            break

    return subroutine


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
        raise ParseError(f"failed to locate type declaration for kernel parameter {param} in {path}")

    loc = Location(path, type_declaration.item.span[0], 0)
    type_spec = fpu.get_child(type_declaration, f2003.Intrinsic_Type_Spec)

    if type_spec is None:
        raise ParseError("derived types are not allowed for kernel arguments", loc)

    return parseType(type_spec.tofortran(), loc)[0]


def parseProgram(ast: f2003.Program, source: str, path: Path) -> Program:
    program = Program(path, ast, source)
    parseNode(ast, program)

    return program


def parseNode(node: Any, program: Program) -> None:
    if isinstance(node, f2003.Call_Stmt):
        parseCall(node, program)

    if isinstance(node, f2003.Subroutine_Subprogram):
        parseSubroutine(node, program)

    if not isinstance(node, f2003.Base):
        return

    for child in node.children:
        parseNode(child, program)


def parseSubroutine(node: f2003.Subroutine_Subprogram, program: Program) -> None:
    subroutine_statement = fpu.get_child(node, f2003.Subroutine_Stmt)
    name_node = fpu.get_child(subroutine_statement, f2003.Name)

    loc = Location(str(program.path), 0, 0)

    name = parseIdentifier(name_node, loc)
    function = Function(name, node, program)

    param_identifiers = parseSubroutineParameters(program.path, node)

    for param in param_identifiers:
        typ = parseParamType(program.path, node, param)
        function.parameters.append((param, typ))

    for call in fpu.walk(node, f2003.Call_Stmt):
        name = parseIdentifier(fpu.get_child(call, f2003.Name), loc)
        function.depends.add(name)

    program.entities.append(function)


def parseCall(node: f2003.Call_Stmt, program: Program) -> None:
    loc = Location(str(program.path), node.item.span[0], 0)
    name = parseIdentifier(fpu.get_child(node, f2003.Name), loc)
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
            raise ParseError("unable to parse op_par_loop argument", loc)

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
            raise ParseError(f"invalid loop argument {name}", loc)

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

    map_idx = parseIntLiteral(args_list[1], loc)
    map_ptr: Optional[str] = parseIdentifier(args_list[2], loc)

    if map_ptr == "OP_ID":
        map_ptr = None

    dat_dim = parseIntLiteral(args_list[3], loc)
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
    dim = parseIntLiteral(args_list[1], loc)
    typ = parseType(parseStringLiteral(args_list[2], loc), loc)[0]

    access_type = parseAccessType(args_list[3], loc)

    loop.addArgGbl(loc, ptr, dim, typ, access_type, opt)


def parseIdentifier(node: Any, loc: Location) -> str:
    return node.string


def parseIntLiteral(node: Any, loc: Location) -> int:
    if type(node) is f2003.Signed_Int_Literal_Constant or type(node) is f2003.Int_Literal_Constant:
        return int(node.items[0])

    if type(node) is f2003.Level_2_Unary_Expr:
        coeff = -1 if node.items[0] == "-" else 1
        return coeff * int(node.items[1].items[0])

    raise ParseError("unable to parse int literal", loc)


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


def parseType(typ: str, loc: Location) -> Tuple[OP.Type, bool]:
    typ_clean = typ.strip().lower()
    typ_clean = re.sub(r"\s*kind\s*=\s*", "", typ_clean)

    soa = False
    if re.search(r":soa", typ_clean):
        soa = True

    typ_clean = re.sub(r"\s*:soa\s*", "", typ_clean)

    def mk_type_regex(t, k):
        return rf"{t}(?:\s*\(\s*{k}\s*\))?\s*$"

    integer_match = re.match(mk_type_regex("integer", "(?:ik)?(4|8)"), typ_clean)
    if integer_match is not None:
        size = 32
        if integer_match.groups()[0] is not None:
            size = int(integer_match.groups()[0]) * 8

        return OP.Int(True, size), soa

    real_match = re.match(mk_type_regex("real", "(?:rk)?(4|8)"), typ_clean)
    if real_match is not None:
        size = 32
        if real_match.groups()[0] is not None:
            size = int(real_match.groups()[0]) * 8

        return OP.Float(size), soa

    logical_match = re.match(mk_type_regex("logical", "(?:lk)?"), typ_clean)
    if logical_match is not None:
        return OP.Bool(), soa

    raise ParseError(f'unable to parse type "{typ}"', loc)
