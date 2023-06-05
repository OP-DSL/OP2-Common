from typing import Any, Callable, List, Optional, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

from store import Function
from util import find, safeFind


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
