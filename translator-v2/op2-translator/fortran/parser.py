import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from store import Application, Function, Location, ParseError, Program
from util import safeFind

TN = TypeVar("TN")


def getChild(node: f2003.Base, node_type: Type[TN]) -> TN:
    child = fpu.get_child(node, node_type)
    assert child is not None

    return child


def getChildOpt(node: f2003.Base, node_type: Type[TN]) -> Optional[TN]:
    return fpu.get_child(node, node_type)


def parseProgram(ast: f2003.Program, source: str, path: Path) -> Program:
    program = Program(path, ast, source)
    parseNode(ast, program, Location(str(program.path), 0, 0))

    return program


# TODO:
# Function calls turn up as f2003.Part_Ref which we can't distinguish from actual
# array indexes until we have the list of functions.
# Note that this means only functions contained in the same program as the call can be used.
# This also means that contained functions from other subroutines may be registered as dependencies.
def parseFunctionDependencies(program: Program, app: Application) -> None:
    subprograms = []

    for subprogram in fpu.walk(program.ast, f2003.Subroutine_Subprogram):
        subprograms.append(subprogram)

    for subprogram in fpu.walk(program.ast, f2003.Function_Subprogram):
        subprograms.append(subprogram)

    for subprogram in subprograms:
        if isinstance(subprogram, f2003.Subroutine_Subprogram):
            definition_statement = getChild(subprogram, f2003.Subroutine_Stmt)
        else:
            definition_statement = getChild(subprogram, f2003.Function_Stmt)

        name_node = getChild(definition_statement, f2003.Name)
        dependant = safeFind(program.entities, lambda e: e.name == name_node.string.lower())

        if dependant is None:
            continue

        for node in fpu.walk(subprogram, f2003.Part_Ref):
            ref_name_node = getChild(node, f2003.Name)
            dependencies = list(
                filter(
                    lambda e: isinstance(e.ast, f2003.Function_Subprogram),
                    app.findEntities(ref_name_node.string.lower()),
                )
            )

            if len(dependencies) == 0:
                continue

            dependant.depends.add(ref_name_node.string.lower())


def parseNode(node: Any, program: Program, loc: Location) -> None:
    if isinstance(node, f2003.Call_Stmt):
        parseCall(node, program, loc)

    if isinstance(node, f2003.Function_Subprogram):
        parseSubprogram(node, program, loc)

    if isinstance(node, f2003.Subroutine_Subprogram):
        parseSubprogram(node, program, loc)

    if not isinstance(node, f2003.Base):
        return

    for child in node.children:
        if child is None:
            continue

        child_loc = loc

        if hasattr(child, "item") and child.item is not None:
            child_loc = Location(str(program.path), child.item.span[0], 0)

        parseNode(child, program, child_loc)


def parseSubprogram(
    node: Union[f2003.Subroutine_Subprogram, f2003.Function_Subprogram], program: Program, loc: Location
) -> None:
    if isinstance(node, f2003.Subroutine_Subprogram):
        definition_statement = getChild(node, f2003.Subroutine_Stmt)
    else:
        definition_statement = getChild(node, f2003.Function_Stmt)

    name_node = getChild(definition_statement, f2003.Name)

    name = parseIdentifier(name_node, loc)
    function = Function(name, node, program)

    function.parameters = parseSubprogramParameters(program.path, node)

    for call in fpu.walk(node, f2003.Call_Stmt):
        name_node = getChildOpt(call, f2003.Name)

        if name_node is None:  # Happens for Procedure_Designator (stuff like op%access_i4...)
            continue

        name = parseIdentifier(name_node, loc)
        function.depends.add(name)

    program.entities.append(function)


def parseSubprogramParameters(
    path: Path, node: Union[f2003.Subroutine_Subprogram, f2003.Function_Subprogram]
) -> List[str]:
    if isinstance(node, f2003.Subroutine_Subprogram):
        definition_statement = getChild(node, f2003.Subroutine_Stmt)
    else:
        definition_statement = getChild(node, f2003.Function_Stmt)

    arg_list = getChildOpt(definition_statement, f2003.Dummy_Arg_List)

    if arg_list is None:
        return []

    item = getattr(definition_statement, "item")
    loc = Location(str(path), item.span[0], 0)

    parameters = []
    for item in arg_list.items:
        parameters.append(parseIdentifier(item, loc))

    return parameters


def parseCall(node: f2003.Call_Stmt, program: Program, loc: Location) -> None:
    name_node = getChildOpt(node, f2003.Name)
    if name_node is None:  # Happens for Procedure_Designator (stuff like op%access_i4...)
        return

    name = parseIdentifier(name_node, loc)
    args = getChildOpt(node, f2003.Actual_Arg_Spec_List)

    if name == "op_decl_const":
        assert args is not None
        program.consts.append(parseConst(args, loc))

    elif re.match(r"op_par_loop_\d+", name):
        assert args is not None
        program.loops.append(parseLoop(program, args, loc))


def parseConst(args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Const:
    if args is None or len(args.items) != 3:
        raise ParseError("incorrect number of arguments for op_decl_const", loc)

    ptr = parseIdentifier(args.items[0], loc)
    dim = parseIntLiteral(args.items[1], loc)  # TODO: Might not be an integer literal?
    assert dim is not None

    typ_str = parseStringLiteral(args.items[2], loc).strip().lower()
    typ = parseType(typ_str, loc)[0]

    return OP.Const(loc, ptr, dim, typ)


def parseLoop(program: Program, args: Optional[f2003.Actual_Arg_Spec_List], loc: Location) -> OP.Loop:
    if args is None or len(args.items) < 3:
        raise ParseError("incorrect number of arguments for op_par_loop", loc)

    kernel = parseIdentifier(args.items[0], loc)
    name = f"{program.path.stem}_{len(program.loops) + 1}_{kernel}"

    loop = OP.Loop(name, loc, kernel)

    for arg_node in args.items[2:]:
        if type(arg_node) is not f2003.Structure_Constructor:
            if type(arg_node) is not f2003.Part_Ref:
                raise ParseError(f"unable to parse op_par_loop argument: {arg_node}", loc)

            name = parseIdentifier(getChild(arg_node, f2003.Name), loc)
            arg_args = getChild(arg_node, f2003.Section_Subscript_List)

            if name == "op_arg_idx":
                parseArgIdx(loop, arg_args, loc)

            else:
                raise ParseError(f"invalid loop argument {arg_node}", loc)

            continue

        name = parseIdentifier(getChild(arg_node, f2003.Type_Name), loc)
        arg_args = getChild(arg_node, f2003.Component_Spec_List)

        if name == "op_arg_dat":
            parseArgDat(loop, False, arg_args, loc)

        elif name == "op_opt_arg_dat":
            parseArgDat(loop, True, arg_args, loc)

        elif name == "op_arg_gbl":
            parseArgGbl(loop, False, arg_args, loc)

        elif name == "op_opt_arg_gbl":
            parseArgGbl(loop, True, arg_args, loc)

        elif name == "op_arg_info":
            parseArgInfo(loop, arg_args, loc)

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

    if map_ptr.upper() == "OP_ID":
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

    if map_ptr.upper() == "OP_ID":
        map_ptr = None

    loop.addArgIdx(loc, map_ptr, map_idx)


def parseArgInfo(loop: OP.Loop, args: Optional[f2003.Component_Spec_List], loc: Location) -> None:
    if args is None or len(args.items) != 4:
        raise ParseError("incorrect number of arguments for op_arg_info", loc)

    ptr = parseIdentifier(args.items[0], loc)
    dim = parseIntLiteral(args.items[1], loc, True)
    typ = parseType(parseStringLiteral(args.items[2], loc), loc)[0]

    ref = parseIntLiteral(args.items[3], loc)
    assert ref is not None

    loop.addArgInfo(loc, ptr, dim, typ, ref)


def parseIdentifier(node: Any, loc: Location) -> str:
    # if not hasattr(node, "string"):
    #    raise ParseError(f"Unable to parse identifier for node: {node}", loc)

    return node.string.lower()


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
    access_type_str = parseIdentifier(node, loc).upper()

    access_type_map = {"OP_READ": 0, "OP_WRITE": 1, "OP_RW": 2, "OP_INC": 3, "OP_MIN": 4, "OP_MAX": 5, "OP_WORK": 6}

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
