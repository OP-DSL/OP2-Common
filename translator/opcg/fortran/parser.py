import re
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Set
from xml.etree.ElementTree import Element, dump

import open_fortran_parser as fp

import op as OP
from store import Kernel, Location, ParseError, Program
from util import enumRegex, safeFind

# The current current file being parsed
_current_file: str = "?"


def parse(path: Path) -> Element:
    try:
        # Track the current file for parse errors
        global _current_file
        _current_file = str(path)

        # Invoke OFP on the source
        return fp.parse(path, raise_on_error=True)
    except CalledProcessError as error:
        raise ParseError(error.output)


def parseKernel(self, path: Path, name: str) -> Kernel:
    # Parse AST
    ast = parse(path)

    # Search for kernel function
    nodes = ast.findall("file/subroutine")
    node = safeFind(nodes, lambda n: n.attrib["name"] == name)
    if not node:
        raise ParseError(f"failed to locate kernel function {name}")

    # Parse parameter identifiers
    param_identifiers = [n.attrib["name"] for n in node.findall("header/arguments/argument")]
    params = [("", "")] * len(param_identifiers)

    # TODO: Cleanup
    for decl in node.findall("body/specification/declaration"):
        if decl.attrib and decl.attrib["type"] == "variable":
            type = decl.find("type")
            for variable in decl.findall("variables/variable"):
                if variable.attrib:
                    identifier = variable.attrib["name"]
                    if identifier in param_identifiers:
                        index = param_identifiers.index(identifier)
                        params[index] = (identifier, parseType(type))

    return Kernel(name, path, ast, params)


# soa argument is currently only used in C++ parser
def parseProgram(self, path: Path, include_dirs: Set[Path], soa: bool) -> Program:
    # Parse AST
    ast = parse(path)

    # Create a program store
    program = Program(path)

    # Iterate over all Call AST nodes
    for call in ast.findall(".//call"):

        # Store call source location
        loc = parseLocation(call)
        name = parseIdentifier(call)

        name_node = call.find("name")
        assert name_node is not None

        if name_node.attrib["type"] == "procedure":
            # Collect the call arg nodes
            args = call.findall("name/subscripts/subscript")

            if name == "op_init_base":
                program.recordInit(loc)

            elif name == "op_decl_set":
                program.sets.append(parseSet(args, loc))

            elif name == "op_decl_set_hdf5":
                program.sets.append(parseSet_hdf5(args, loc))

            elif name == "op_decl_map":
                program.maps.append(parseMap(args, loc))

            elif name == "op_decl_map_hdf5":
                program.maps.append(parseMap_hdf5(args, loc))

            elif name == "op_decl_dat":
                program.datas.append(parseDat(args, loc))

            elif name == "op_decl_dat_hdf5":
                program.datas.append(parseDat_hdf5(args, loc))

            elif name == "op_decl_const":
                program.consts.append(parseConst(args, loc))

            elif re.search(r"op_par_loop_[1-9]\d*", name):
                program.loops.append(parseLoop(args, loc))

            elif name == "op_exit":
                program.recordExit()

    # Return the program
    return program


def parseSet(nodes: List[Element], loc: Location) -> OP.Set:
    if len(nodes) != 3:
        raise ParseError("incorrect number of args passed to op_decl_set", loc)

    _ = parseIdentifier(nodes[0])
    ptr = parseIdentifier(nodes[1])
    debug = parseStringLit(nodes[2])

    return OP.Set(ptr)


def parseSet_hdf5(nodes: List[Element], loc: Location) -> OP.Set:
    if len(nodes) != 4:
        raise ParseError("incorrect number of args passed to op_decl_set_hdf5", loc)

    _ = parseIdentifier(nodes[0])
    ptr = parseIdentifier(nodes[1])
    file = parseStringLit(nodes[2])
    debug = parseStringLit(nodes[3])

    return OP.Set(ptr)


def parseMap(nodes: List[Element], loc: Location) -> OP.Map:
    if len(nodes) != 6:
        raise ParseError("incorrect number of args passed to op_decl_map", loc)

    from_set = parseIdentifier(nodes[0])
    to_set = parseIdentifier(nodes[1])
    dim = parseIntLit(nodes[2], signed=False)
    _ = parseIdentifier(nodes[3])
    ptr = parseIdentifier(nodes[4])
    debug = parseStringLit(nodes[5])

    return OP.Map(from_set, to_set, dim, ptr, loc)


def parseMap_hdf5(nodes: List[Element], loc: Location) -> OP.Map:
    if len(nodes) != 7:
        raise ParseError("incorrect number of args passed to op_decl_map_hdf5", loc)

    from_set = parseIdentifier(nodes[0])
    to_set = parseIdentifier(nodes[1])
    dim = parseIntLit(nodes[2], signed=False)
    ptr = parseIdentifier(nodes[3])
    file = parseStringLit(nodes[4])
    debug = parseStringLit(nodes[5])
    status = parseIdentifier(nodes[6])

    return OP.Map(from_set, to_set, dim, ptr, loc)


def parseDat(nodes: List[Element], loc: Location) -> OP.Data:
    if len(nodes) != 6:
        raise ParseError("incorrect number of args passed to op_decl_dat", loc)

    set_ = parseIdentifier(nodes[0])
    dim = parseIntLit(nodes[1], signed=False)
    typ = normaliseType(parseStringLit(nodes[2]))
    _ = parseIdentifier(nodes[3])
    ptr = parseIdentifier(nodes[4])
    debug = parseStringLit(nodes[5])

    return OP.Data(set_, dim, typ, ptr, loc)


def parseDat_hdf5(nodes: List[Element], loc: Location) -> OP.Data:
    if len(nodes) != 7:
        raise ParseError("incorrect number of args passed to op_decl_dat", loc)

    set_ = parseIdentifier(nodes[0])
    dim = parseIntLit(nodes[1], signed=False)
    ptr = parseIdentifier(nodes[2])
    typ = normaliseType(parseStringLit(nodes[3]))
    file = parseStringLit(nodes[4])
    debug = parseStringLit(nodes[5])
    status = parseIdentifier(nodes[6])

    return OP.Data(set_, dim, typ, ptr, loc)


def parseConst(nodes: List[Element], loc: Location) -> OP.Const:
    if len(nodes) != 3:
        raise ParseError("incorrect number of args passed to op_decl_const", loc)

    ptr = parseIdentifier(nodes[0])
    dim = parseIntLit(nodes[1], signed=False)
    debug = parseStringLit(nodes[2])

    return OP.Const(ptr, dim, debug, loc)


def parseLoop(nodes: List[Element], loc: Location) -> OP.Loop:
    if len(nodes) < 3:
        raise ParseError("incorrect number of args passed to op_par_loop", loc)

    # Parse loop kernel and set
    kernel = parseIdentifier(nodes[0])
    set_ = parseIdentifier(nodes[1])

    loop_args = []

    # Parse loop args
    for raw_arg in nodes[2:]:
        name = parseIdentifier(raw_arg)
        arg_loc = parseLocation(raw_arg)
        args = raw_arg.findall("name/subscripts/subscript")

        if name == "op_arg_dat":
            loop_args.append(parseArgDat(args, arg_loc))

        elif name == "op_opt_arg_dat":
            loop_args.append(parseOptArgDat(args, arg_loc))

        elif name == "op_arg_gbl":
            loop_args.append(parseArgGbl(args, arg_loc))

        elif name == "op_opt_arg_gbl":
            loop_args.append(parseOptArgGbl(args, arg_loc))

        else:
            raise ParseError(f"invalid loop argument {name}")

    return OP.Loop(kernel, set_, loc, loop_args)


def parseArgDat(nodes: List[Element], loc: Location) -> OP.Arg:
    if len(nodes) != 6:
        raise ParseError("incorrect number of args passed to op_arg_dat", loc)

    access_regex = enumRegex(OP.DAT_ACCESS_TYPES)

    var = parseIdentifier(nodes[0])
    idx = parseIntLit(nodes[1], signed=True)
    map_ = parseIdentifier(nodes[2])
    dim = parseIntLit(nodes[3], signed=False)
    typ = normaliseType(parseStringLit(nodes[4]))
    acc = parseIdentifier(nodes[5], regex=access_regex)

    return OP.Arg(var, dim, typ, acc, loc, map_, idx)


def parseOptArgDat(nodes: List[Element], loc: Location) -> OP.Arg:
    if len(nodes) != 7:
        ParseError("incorrect number of args passed to op_opt_arg_dat", loc)

    # Parse opt argument
    opt = parseIdentifier(nodes[0])

    # Parse standard argDat arguments
    dat = parseArgDat(nodes[1:], loc)

    # Return augmented dat
    dat.opt = opt
    return dat


def parseArgGbl(nodes: List[Element], loc: Location) -> OP.Arg:
    if len(nodes) != 4:
        raise ParseError("incorrect number of args passed to op_arg_gbl", loc)

    access_regex = enumRegex(OP.GBL_ACCESS_TYPES)

    var = parseIdentifier(nodes[0])
    dim = parseIntLit(nodes[1], signed=False)
    typ = normaliseType(parseStringLit(nodes[2]))
    acc = parseIdentifier(nodes[3], regex=access_regex)

    return OP.Arg(var, dim, typ, acc, loc)


def parseOptArgGbl(nodes: List[Element], loc: Location) -> OP.Arg:
    if len(nodes) != 5:
        ParseError("incorrect number of args passed to op_opt_arg_gbl", loc)

    # Parse opt argument
    opt = parseIdentifier(nodes[0])

    # Parse standard argGbl arguments
    dat = parseArgGbl(nodes[1:], loc)

    # Return augmented dat
    dat.opt = opt
    return dat


def parseIdentifier(node: Optional[Element], regex: str = None) -> str:
    assert node is not None

    # Parse location
    loc = parseLocation(node)

    # Descend to child node
    node = node.find("name")

    # Validate the node
    if not node or not node.attrib["id"]:
        raise ParseError("expected identifier", loc)

    value = node.attrib["id"]

    # Apply conditional regex constraint
    if regex and not re.match(regex, value):
        raise ParseError(f"expected identifier matching {regex}", loc)

    return value


def parseIntLit(node: Optional[Element], signed: bool = True) -> int:
    assert node is not None

    # Parse location
    loc = parseLocation(node)

    # Assume the literal is not negated
    negation = False

    # Check if the node is wrapped in a valid unary negation
    if signed and node.find("operation"):
        node = node.find("operation")
        assert node is not None

        operator_node = node.find("operator")
        if node.attrib["type"] == "unary" and operator_node is not None and operator_node.attrib["operator"] == "-":
            negation = True
            node = node.find("operand")

    # Descend to child literal node
    assert node is not None
    node = node.find("literal")

    # Verify and typecheck the literal node
    if not node or node.attrib["type"] != "int":
        if not signed:
            raise ParseError("expected unsigned integer literal", loc)
        else:
            raise ParseError("expected integer literal", loc)

    # Extract the value
    value = int(node.attrib["value"])

    return -value if negation else value


def parseStringLit(node: Optional[Element], regex: str = None) -> str:
    assert node is not None

    # Parse location
    loc = parseLocation(node)

    # Descend to child literal node
    node = node.find("literal")

    # Validate the node
    if not node or node.attrib["type"] != "char":
        raise ParseError("expected string literal", loc)

    # Extract value from string delimeters
    value = node.attrib["value"][1:-1]

    # Apply conditional regex constraint
    if regex and not re.match(regex, value):
        raise ParseError(f"expected string literal matching {regex}", loc)

    return value


def parseLocation(node: Element) -> Location:
    return Location(_current_file, int(node.attrib["line_begin"]), int(node.attrib["col_begin"]))


def parseType(node: Element) -> str:
    # Get the base type
    type = node.attrib["name"]

    # Append kind
    if node.attrib["hasKind"]:
        kind = parseIntLit(node.find("kind"), signed=False)
        type = f"{type}({kind})"

    return normaliseType(type)


def normaliseType(type: str) -> str:
    return re.sub(r"\s*kind\s*=\s*", "", type.lower())
