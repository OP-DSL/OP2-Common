from typing import Any, Callable, Dict, List, Optional, Tuple
import copy

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from store import Function
from util import find, safeFind


# The kind names the translator knows, by lower-case name, each its kind
# number: the standard ones here, an application's own (--kind) added at
# startup.  Fortran names are case-insensitive, so every lookup lowers them.
kinds: Dict[str, int] = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "real32": 4,
    "real64": 8,
    "c_bool": 1,
    "c_int": 4,
    "c_int32_t": 4,
    "c_int64_t": 8,
    "c_long": 8,
    "c_long_long": 8,
    "c_float": 4,
    "c_double": 8,
}


# The intrinsic types the translator supports, by lower-case type name and
# kind number.
intrinsic_types: Dict[Tuple[str, int], OP.Type] = {
    ("integer", 4): OP.Int(True, 32),
    ("integer", 8): OP.Int(True, 64),
    ("real", 4): OP.Float(32),
    ("real", 8): OP.Float(64),
    ("logical", 1): OP.Bool(),
    ("logical", 2): OP.Bool(),
    ("logical", 4): OP.Bool(),
}


# A kind's number, from a literal or a known name, as text or a parse node: 4,
# the default, for no kind and None for an unknown name.
def kindValue(kind: Any) -> Optional[int]:
    if kind is None:
        return 4

    text = str(kind).strip().lower()
    return int(text) if text.isdigit() else kinds.get(text)


# An intrinsic type by name and kind (as kindValue takes it); None if unsupported.
def intrinsicType(name: str, kind: Any) -> Optional[OP.Type]:
    return intrinsic_types.get((name.lower(), kindValue(kind)))


def isRef(node: Any, name: str) -> bool:
    if isinstance(node, f2003.Name) and node.string.lower() == name:
        return True

    if isinstance(node, f2003.Part_Ref) and node.items[0].string.lower() == name:
        return True

    return False


def walkRefs(node: f2003.Base, name: str) -> List[f2003.Name]:
    return list(filter(lambda n: n.string.lower() == name.lower(), fpu.walk(node, f2003.Name)))


def walkOut(node: f2003.Base, node_type) -> f2003.Base:
    parent = node

    while not isinstance(parent, node_type) and getattr(parent, "parent", None) is not None:
        parent = parent.parent

    return parent


def walkOutUntil(node: f2003.Base, node_type) -> f2003.Base:
    parent = node

    while getattr(parent, "parent", None) is not None and not isinstance(parent.parent, node_type):
        parent = parent.parent

    return parent


def getItem(node: f2003.Base) -> Optional[Any]:
    if getattr(node, "item", None) is not None:
        return node.item

    if getattr(node, "parent", None) is not None:
        return getItem(node.parent)

    return None


# Applies the given operation over the specified func parameter, recursing into called functions
# if they are present in funcs
def mapParam(func: Function, param_idx: int, funcs: List[Function], op: Callable[[Function, int], None], *args) -> None:
    called_list = findCalled(func, param_idx, funcs)

    for func2, param_idx2 in called_list:
        stop = op(func2, param_idx2, *args)

        if stop:
            break


def findCalled(func: Function, param_idx: int, funcs: List[Function]) -> List[Tuple[Function, int]]:
    checked = {}
    stack = [(func, param_idx)]

    while len(stack) > 0:
        candidate_func, candidate_param_idx = stack.pop()

        if candidate_func.name not in checked:
            checked[candidate_func.name] = set()
        elif candidate_param_idx in checked[candidate_func.name]:
            continue

        checked[candidate_func.name].add(candidate_param_idx)

        for called_func, called_param_idx in findCalled2(candidate_func, candidate_param_idx, funcs):
            if called_func.name in checked and called_param_idx in checked[called_func.name]:
                continue

            stack.append((called_func, called_param_idx))

    called_list = []
    for func_name, indicies in checked.items():
        for index in indicies:
            called_list.append((find(funcs, lambda f: f.name == func_name), index))

    return called_list


def findCalled2(func: Function, param_idx: int, funcs: List[Function]) -> List[Tuple[Function, int]]:
    called = []

    for node in fpu.walk(func.ast, f2003.Name):
        if node.string.lower() != func.parameters[param_idx]:
            continue

        call = getCall(node, funcs)

        if call is not None:
            called.append(call)

    return called


# Gets the Function and argument index if the passed node is inside a function call
def getCall(node: f2003.Name, funcs: List[Function]) -> Optional[Tuple[Function, int]]:
    if getattr(node, "parent", None) is None:
        return None

    parent = node.parent

    if isinstance(parent, f2003.Part_Ref):
        if getattr(parent, "parent", None) is None:
            return None

        node = node.parent
        parent = node.parent

    # Function calls turn up as Part_Refs with a Section_Subscript_List, but these might
    # also be genuine array indexes
    if isinstance(parent, f2003.Actual_Arg_Spec_List) or isinstance(parent, f2003.Section_Subscript_List):
        func_name_node = fpu.get_child(parent.parent, f2003.Name)

        if func_name_node is None:
            return None

        func = safeFind(funcs, lambda f: f.name == func_name_node.string.lower())
        arg_idx = [id(item) for item in parent.items].index(id(node))

        if func is not None:
            return (func, arg_idx)

    return None


def parseExplicitShapeSpec(spec: f2003.Explicit_Shape_Spec) -> Tuple[str, str]:
    assert len(spec.children) == 2

    if spec.children[0] == None:
        return ("1", str(spec.children[1]))

    return (str(spec.children[0]), str(spec.children[1]))


def parseDimensions(func: Function, param: str) -> Optional[List[Tuple[str, str]]]:
    spec = fpu.get_child(func.ast, f2003.Specification_Part)

    for type_decl in fpu.walk(spec, f2003.Type_Declaration_Stmt):
        check_for_dimension_spec = False

        for entity_decl in fpu.walk(type_decl, f2003.Entity_Decl):
            name = fpu.get_child(entity_decl, f2003.Name).string

            if name != param:
                continue

            shape_spec = fpu.get_child(entity_decl, f2003.Explicit_Shape_Spec_List)

            if shape_spec is None:
                check_for_dimension_spec = True
                break

            return [parseExplicitShapeSpec(spec) for spec in shape_spec.children]

        if not check_for_dimension_spec:
            continue

        dimension_spec = fpu.walk(type_decl, f2003.Dimension_Attr_Spec)
        if len(dimension_spec) == 0:
            return None

        shape_spec = fpu.get_child(dimension_spec[0], f2003.Explicit_Shape_Spec_List)

        if shape_spec is None:
            return None

        return [parseExplicitShapeSpec(spec) for spec in shape_spec.children]

    return None


def hasTarget(func: Function, param: str) -> bool:
    spec = fpu.get_child(func.ast, f2003.Specification_Part)
    param = param.lower()

    for type_decl in fpu.walk(spec, f2003.Type_Declaration_Stmt):
        names = [fpu.get_child(e, f2003.Name).string.lower() for e in fpu.walk(type_decl, f2003.Entity_Decl)]
        attrs = [str(a).upper() for a in fpu.walk(type_decl, f2003.Attr_Spec)]

        if param in names and "TARGET" in attrs:
            return True

    for target_stmt in fpu.walk(spec, f2003.Target_Stmt):
        if param in [n.string.lower() for n in fpu.walk(target_stmt, f2003.Name)]:
            return True

    return False


def eraseDimensions(func: Function, param: str) -> None:
    spec = fpu.get_child(func.ast, f2003.Specification_Part)

    for type_decl in fpu.walk(spec, f2003.Type_Declaration_Stmt):
        check_for_dimension_spec = False

        for entity_decl in fpu.walk(type_decl, f2003.Entity_Decl):
            name = fpu.get_child(entity_decl, f2003.Name).string

            if name != param:
                continue

            shape_spec = fpu.get_child(entity_decl, f2003.Explicit_Shape_Spec_List)

            if shape_spec is None:
                check_for_dimension_spec = True
                break

            items = list(entity_decl.items)

            items[1] = f2003.Array_Spec("*")
            items[1].parent = entity_decl

            entity_decl.items = tuple(items)

        if not check_for_dimension_spec:
            continue

        dimension_spec = fpu.walk(type_decl, f2003.Dimension_Attr_Spec)
        shape_spec = fpu.get_child(dimension_spec[0], f2003.Explicit_Shape_Spec_List)

        # dimension(...) :: x - just erase the dimension
        if len(fpu.get_child(type_decl, f2003.Entity_Decl_List).items) == 1:
            shape_spec.items = (f2003.Array_Spec("*"),)
            shape_spec.items[0].parent = shape_spec
            continue

        # dimension(...) :: x, y, ... - split out a new type decl and erase the dimension there
        new_type_decl = copy.deepcopy(type_decl)
        new_dimension_spec = fpu.walk(new_type_decl, f2003.Dimension_Attr_Spec)
        new_shape_spec = fpu.get_child(new_dimension_spec[0], f2003.Explicit_Shape_Spec_List)

        new_shape_spec.items = (f2003.Array_Spec("*"),)
        new_shape_spec.items[0].parent = shape_spec

        for entity_decl in fpu.walk(new_type_decl, f2003.Entity_Decl):
            name = fpu.get_child(entity_decl, f2003.Name).string

            if name != param:
                continue

            fpu.get_child(new_type_decl, f2003.Entity_Decl_List).items = (entity_decl,)
            break

        spec.content.append(new_type_decl)

        entity_decls = []
        for entity_decl in fpu.walk(type_decl, f2003.Entity_Decl):
            name = fpu.get_child(entity_decl, f2003.Name).string

            if name == param:
                continue

            entity_decls.append(entity_decl)

        fpu.get_child(type_decl, f2003.Entity_Decl_List).items = tuple(entity_decls)
