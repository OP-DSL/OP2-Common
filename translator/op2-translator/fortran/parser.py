import re
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from store import Kernel, Location, ParseError, Program
from util import enumRegex


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
        raise ParseError(f"derived types are not allowed for kernel arguments", loc)

    return parseType(type_spec.tofortran(), loc)


def parseProgram(ast: f2003.Program, path: Path) -> Program:
    program = Program(path)

    for call in fpu.walk(ast, f2003.Call_Stmt):
        loc = Location(str(path), call.item.span[0], 0)
        name = parseIdentifier(fpu.get_child(call, f2003.Name), loc)

        parseCall(name, fpu.get_child(call, f2003.Actual_Arg_Spec_List), loc, program)

    return program


def parseCall(name: str, args: Optional[f2003.Actual_Arg_Spec_List], loc: Location, program: Program) -> None:
    if name == "op_init_base":
        program.recordInit(loc)

    elif name == "op_decl_set":
        program.sets.append(parseSet(args, loc))

    elif name == "op_decl_set_hdf5":
        program.sets.append(parseSetHdf5(args, loc))

    elif name == "op_decl_map":
        program.maps.append(parseMap(args, loc))

    elif name == "op_decl_map_hdf5":
        program.maps.append(parseMapHdf5(args, loc))

    elif name == "op_decl_dat":
        program.dats.append(parseDat(args, loc))

    elif name == "op_decl_dat_hdf5":
        program.dats.append(parseDatHdf5(args, loc))

    elif name == "op_decl_const":
        program.consts.append(parseConst(args, loc))

    elif re.match(r"op_par_loop_\d+", name):
        program.loops.append(parseLoop(args, loc))

    elif name == "op_exit":
        program.recordExit()


def parseSet(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Set:
    if args is None or len(args.items) != 3:
        raise ParseError("incorrect number of arguments for op_decl_set", loc)

    return OP.Set(loc, parseIdentifier(args.items[1], loc))


def parseSetHdf5(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Set:
    if args is None or len(args.items) != 4:
        raise ParseError("incorrect number of arguments for op_decl_set_hdf5", loc)

    return OP.Set(loc, parseIdentifier(args.items[1], loc))


def parseMap(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Map:
    if args is None or len(args.items) != 6:
        raise ParseError("incorrect number of arguments for op_decl_map", loc)

    from_set = parseIdentifier(args.items[0], loc)
    to_set = parseIdentifier(args.items[1], loc)
    dim = parseIntLiteral(args.items[2], loc)
    ptr = parseIdentifier(args.items[4], loc)

    return OP.Map(loc, from_set, to_set, dim, ptr)


def parseMapHdf5(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Map:
    if args is None or len(args.items) != 7:
        raise ParseError("incorrect number of arguments for op_decl_map_hdf5", loc)

    from_set = parseIdentifier(args.items[0], loc)
    to_set = parseIdentifier(args.items[1], loc)
    dim = parseIntLiteral(args.items[2], loc)
    ptr = parseIdentifier(args.items[3], loc)

    return OP.Map(loc, from_set, to_set, dim, ptr)


def parseDat(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Dat:
    if args is None or len(args.items) != 6:
        raise ParseError("incorrect number of arguments for op_decl_dat", loc)

    soa = False

    set_ = parseIdentifier(args.items[0], loc)
    dim = parseIntLiteral(args.items[1], loc)

    typ_str = parseStringLiteral(args.items[2], loc).strip().lower()

    soa_regex = r":soa$"
    if re.search(soa_regex, typ_str):
        soa = True
        typ_str = re.sub(soa_regex, "", typ_str)

    typ = parseType(typ_str, loc)

    ptr = parseIdentifier(args.items[4], loc)

    return OP.Dat(loc, set_, dim, typ, ptr, soa)


def parseDatHdf5(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Dat:
    if args is None or len(args.items) != 7:
        raise ParseError("incorrect number of arguments for op_decl_dat_hdf5", loc)

    soa = False

    set_ = parseIdentifier(args.items[0], loc)
    dim = parseIntLiteral(args.items[1], loc)
    ptr = parseIdentifier(args.items[2], loc)

    typ_str = parseStringLiteral(args.items[3], loc).strip().lower()

    soa_regex = r":soa$"
    if re.search(soa_regex, typ_str):
        soa = True
        typ_str = re.sub(soa_regex, "", typ_str)

    typ = parseType(typ_str, loc)

    return OP.Dat(loc, set_, dim, typ, ptr, soa)


def parseConst(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Const:
    if args is None or len(args.items) != 3:
        raise ParseError("incorrect number of arguments for op_decl_const", loc)

    ptr = parseIdentifier(args.items[0], loc)
    dim = parseIntLiteral(args.items[1], loc)  # TODO: Might not be an integer literal?

    typ_str = parseStringLiteral(args.items[2], loc).strip().lower()
    typ = parseType(typ_str, loc)

    return OP.Const(loc, dim, typ, ptr)


def parseLoop(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Loop:
    if args is None or len(args.items) < 3:
        raise ParseError("incorrect number of arguments for op_par_loop", loc)

    kernel = parseIdentifier(args.items[0], loc)
    set_ptr = parseIdentifier(args.items[1], loc)

    loop_args = []
    for arg_node in args.items[2:]:
        if type(arg_node) is not f2003.Structure_Constructor:
            raise ParseError("unable to parse op_par_loop argument", loc)

        name = parseIdentifier(fpu.get_child(arg_node, f2003.Type_Name), loc)
        arg_args = fpu.get_child(arg_node, f2003.Component_Spec_List)

        if name == "op_arg_dat":
            loop_args.append(parseArgDat(arg_args, loc))

        elif name == "op_opt_arg_dat":
            loop_args.append(parseOptArgDat(arg_args, loc))

        elif name == "op_arg_gbl":
            loop_args.append(parseArgGbl(arg_args, loc))

        elif name == "op_opt_arg_gbl":
            loop_args.append(parseOptArgGbl(arg_args, loc))

        else:
            raise ParseError(f"invalid loop argument {name}", loc)

    return OP.Loop(loc, kernel, set_ptr, loop_args)


def parseArgDat(args: Optional[f2003.Component_Spec_List], loc: Location) -> OP.ArgDat:
    if args is None or len(args.items) != 6:
        raise ParseError("incorrect number of arguments for op_arg_dat", loc)

    dat_ptr = parseIdentifier(args.items[0], loc)

    map_idx = parseIntLiteral(args.items[1], loc)
    map_ptr: Optional[str] = parseIdentifier(args.items[2], loc)

    if map_ptr == "OP_ID":
        map_ptr = None

    dat_dim = parseIntLiteral(args.items[3], loc)
    dat_typ = parseType(parseStringLiteral(args.items[4], loc), loc)

    access_type = parseAccessType(args.items[5], loc)

    return OP.ArgDat(loc, access_type, False, dat_ptr, dat_dim, dat_typ, map_ptr, map_idx)


def parseOptArgDat(args: Optional[f2003.Component_Spec_List], loc: Location) -> OP.ArgDat:
    if args is None or len(args.items) != 7:
        raise ParseError("incorrect number of arguments for op_opt_arg_dat", loc)

    dat_ptr = parseIdentifier(args.items[1], loc)

    map_idx = parseIntLiteral(args.items[2], loc)
    map_ptr: Optional[str] = parseIdentifier(args.items[3], loc)

    if map_ptr == "OP_ID":
        map_ptr = None

    dat_dim = parseIntLiteral(args.items[4], loc)
    dat_typ = parseType(parseStringLiteral(args.items[5], loc), loc)

    access_type = parseAccessType(args.items[6], loc)

    return OP.ArgDat(loc, access_type, True, dat_ptr, dat_dim, dat_typ, map_ptr, map_idx)


def parseArgGbl(args: Optional[f2003.Component_Spec_List], loc: Location) -> OP.ArgGbl:
    if args is None or len(args.items) != 4:
        raise ParseError("incorrect number of arguments for op_arg_gbl", loc)

    ptr = parseIdentifier(args.items[0], loc)
    dim = parseIntLiteral(args.items[1], loc)
    typ = parseType(parseStringLiteral(args.items[2], loc), loc)

    access_type = parseAccessType(args.items[3], loc)

    return OP.ArgGbl(loc, access_type, False, ptr, dim, typ)


def parseOptArgGbl(args: Optional[f2003.Component_Spec_List], loc: Location) -> OP.ArgGbl:
    if args is None or len(args.items) != 5:
        raise ParseError("incorrect number of arguments for op_opt_arg_gbl", loc)

    ptr = parseIdentifier(args.items[1], loc)
    dim = parseIntLiteral(args.items[2], loc)
    typ = parseType(parseStringLiteral(args.items[3], loc), loc)

    access_type = parseAccessType(args.items[4], loc)

    return OP.ArgGbl(loc, access_type, True, ptr, dim, typ)


def parseIdentifier(node: Any, loc: Location) -> str:
    if type(node) is not f2003.Name and type(node) is not f2003.Type_Name:
        raise ParseError("unable to parse identifier", loc)

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

    if access_type_str not in OP.AccessType.values():
        raise ParseError(
            f"invalid access type {access_type_str}, expected one of {', '.join(OP.AccessType.values())}", loc
        )

    return OP.AccessType(access_type_str)


def parseType(typ: str, loc: Location) -> Tuple[OP.Type, bool]:
    typ_clean = typ.strip().lower()
    typ_clean = re.sub(r"\s*kind\s*=\s*", "", typ_clean)

    mk_type_regex = lambda t, k: f"{t}(?:\s*\(\s*{k}\s*\))?\s*$"

    integer_match = re.match(mk_type_regex("integer", "(?:ik)?(4|8)"), typ_clean)
    if integer_match is not None:
        size = 32
        if integer_match.groups()[0] is not None:
            size = int(integer_match.groups()[0]) * 8

        return OP.Int(True, size)

    real_match = re.match(mk_type_regex("real", "(?:rk)?(4|8)"), typ_clean)
    if real_match is not None:
        size = 32
        if real_match.groups()[0] is not None:
            size = int(real_match.groups()[0]) * 8

        return OP.Float(size)

    logical_match = re.match(mk_type_regex("logical", "(?:lk)?"), typ_clean)
    if logical_match is not None:
        return OP.Bool()

    raise ParseError(f'unable to parse type "{typ}"', loc)
