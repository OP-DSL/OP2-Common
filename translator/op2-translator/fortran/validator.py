from typing import Any, List, Optional, Union, Callable, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify

import fortran.translator.kernels as ftk
import op as OP
from op import OpError
from store import Application, Entity, Function, Program
from util import find, safeFind


def validateLoop(loop: OP.Loop, program: Program, app: Application) -> None:
    kernel_entities = app.findEntities(loop.kernel, program, [])

    if len(kernel_entities) == 0:
        raise OpError(f"unable to find kernel function for {loop.kernel}")
    elif len(kernel_entities) > 1:
        raise OpError(f"ambiguous kernel function for {loop.kernel}")

    dependencies = ftk.extractDependencies(kernel_entities, app, [])
    entities = kernel_entities + list(filter(lambda e: isinstance(e, Function), dependencies))

    seen_entity_names = []
    for entity in entities:
        if entity.name in seen_entity_names:
            raise OpError(f"ambiguous function {entity.name} used in kernel {loop.kernel}")

        seen_entity_names.append(entity.name)

    if len(loop.args) != len(kernel_entities[0].parameters):
        raise OpError(
            f"op_par_loop argument list length ({len(loop.args)}) mismatch "
            f"(expected: {len(kernel_entities[0].parameters)}, kernel function: {loop.kernel})",
            loop.loc,
        )
        return

    # Check for args marked READ but appear to be written
    for idx, arg in enumerate(loop.args):
        if isinstance(arg, OP.ArgIdx) or isinstance(arg, OP.ArgInfo):
            continue

        if arg.access_type != OP.AccessType.READ:
            continue

        written, in_subcall = paramIsWritten(idx, kernel_entities[0], entities)
        param_name = kernel_entities[0].parameters[idx]

        if written:
            msg = f"{loop.loc}: Warning: arg {idx + 1} ({param_name}) of {loop.kernel} marked OP_READ but was written"
            if in_subcall:
                msg = msg + " (in called subroutine)"

            print(msg)
            loop.fallback = True

    # Check for OP_INC args that don't appear to be incremented
    for idx, arg in enumerate(loop.args):
        if isinstance(arg, OP.ArgIdx) or isinstance(arg, OP.ArgInfo) or arg.access_type != OP.AccessType.INC:
            continue

        violations = []
        mapParam(idx, kernel_entities[0], entities, checkInc, entities, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]

            print(
                f"{loop.loc}: Warning: arg {idx + 1} ({param_name}) of {loop.kernel} marked OP_INC but not incremented:"
            )

            for violation in violations:
                print(f"    {violation}")

            print()
            loop.fallback = True


def isRef(node: Any, name: str) -> bool:
    if isinstance(node, f2003.Name) and node.string.lower() == name:
        return True

    if isinstance(node, f2003.Part_Ref) and node.items[0].string.lower() == name:
        return True

    return False


def paramIsWritten(param_idx: int, func: Function, funcs: List[Function]) -> (bool, bool):
    param_name = func.parameters[param_idx]

    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    for node in fpu.walk(execution_part, f2003.Assignment_Stmt):
        if isRef(node.items[0], param_name):
            return True, False

    for node in fpu.walk(execution_part, f2003.Call_Stmt):
        name_node = fpu.get_child(node, f2003.Name)

        if name_node is None:
            continue

        name = name_node.string.lower()
        args = fpu.get_child(node, f2003.Actual_Arg_Spec_List)

        if args is None:
            continue

        func2 = safeFind(funcs, lambda f: f.name == name)
        if func2 is None:
            continue

        for param2_idx, arg in enumerate(args.items):
            if not isRef(arg, param_name):
                continue

            if paramIsWritten(param2_idx, func2, funcs)[0]:
                return True, True

    return False, False


def checkInc(param_idx: int, func: Function, funcs: List[Function], violations: List[str]) -> None:
    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    def msg(s: str) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {func.parameters[param_idx]}): {s}"

    assignment_lhs_refs = []
    other_refs = []

    # Sort all the Name node refs into assignment LHS or something else
    for node in walkRefs(execution_part, func.parameters[param_idx]):
        if getattr(node, "parent", None) is None:
            continue

        if isinstance(node.parent, f2003.Assignment_Stmt):
            if id(node.parent.items[0]) == id(node):
                assignment_lhs_refs.append(node)
                continue

        if (
            isinstance(node.parent, f2003.Part_Ref)
            and getattr(node.parent, "parent", None) is not None
            and isinstance(node.parent.parent, f2003.Assignment_Stmt)
        ):
            if id(node.parent.parent.items[0]) == id(node.parent):
                assignment_lhs_refs.append(node)
                continue

        other_refs.append(node)

    # Remove the RHS refs from other_refs
    for node in assignment_lhs_refs:
        assignment_node = node.parent

        if isinstance(assignment_node, f2003.Part_Ref):
            assignment_node = assignment_node.parent

        for node2 in walkRefs(assignment_node.items[2], func.parameters[param_idx]):
            other_refs = list(filter(lambda r: id(r) != id(node2), other_refs))

    # Everything left in other_refs must be either passed as a param to a function or a violation
    for node in other_refs:
        call = getCall(node, funcs)

        if call is not None:
            continue

        violations.append(msg(f"invalid context: {getItem(node).line}"))

    # Finally check the assignments in assignment_lhs_refs
    for node in assignment_lhs_refs:
        assignment_node = walkOut(node, f2003.Assignment_Stmt)

        rhs_refs = walkRefs(assignment_node.items[2], node.string)

        if len(rhs_refs) > 1:
            violations.append(msg(f"multi-ref: {getItem(node).line}"))
            continue

        if len(rhs_refs) == 0:
            violations.append(msg(f"no-ref: {getItem(node).line}"))
            continue

        rhs_ref = rhs_refs[0]

        if isinstance(assignment_node.items[0], f2003.Part_Ref):
            rhs_ref = rhs_ref.parent

        if repr(rhs_ref) != repr(assignment_node.items[0]):
            violations.append(msg(f"index mismatch: {getItem(node).line}"))
            continue

        try:
            count = [0]
            expr = simplifyLevel2(assignment_node.items[2], assignment_node.items[0], node.string, count)
        except OpError as e:
            violations.append(msg(f"invalid usage: {getItem(node).line}"))
            continue

        sym_expr = parse_expr(f"({expr}) - (x + {expr.replace('x', '0')})", evaluate=False)
        if simplify(sym_expr) != 0:
            violations.append(msg(f"non increment: {getItem(node).line}"))


def simplifyLevel2(node: f2003.Base, ref: Union[f2003.Name, f2003.Part_Ref], ref_name: str, count: List[int]) -> str:
    def incSym() -> str:
        count[0] += 1
        return f"y{count[0]}"

    if isinstance(node, f2003.Parenthesis):
        return f"({simplifyLevel2(node.children[0], ref, ref_name, count)})"

    if isinstance(node, f2003.Level_2_Expr):
        if len(walkRefs(node.items[0], ref_name)) > 0:
            return f"{simplifyLevel2(node.items[0], ref, ref_name, count)} {node.items[1]} {incSym()}"

        if len(walkRefs(node.items[2], ref_name)) > 0:
            return f"{incSym()} {node.items[1]} {simplifyLevel2(node.items[2], ref, ref_name, count)}"

        assert False

    if isinstance(node, f2003.Name):
        if node.string.lower() == ref.string.lower():
            return "x"

        return incSym()

    if isinstance(node, f2003.Part_Ref):
        if node.items[0].string.lower() == ref.items[0].string.lower():
            return "x"

        return incSym()

    if len(walkRefs(node, ref_name)) > 0:
        raise OpError(str())

    return incSym()


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
def mapParam(param_idx: int, func: Function, funcs: List[Function], op: Callable[[int, Function], None], *args) -> None:
    called_list = findCalled(param_idx, func, funcs)

    for func2, param_idx2 in called_list:
        stop = op(param_idx2, func2, *args)

        if stop:
            break


def findCalled(param_idx: int, func: Function, funcs: List[Function]) -> List[Tuple[Function, int]]:
    checked = {}
    stack = [(func, param_idx)]

    while len(stack) > 0:
        candidate_func, candidate_param_idx = stack.pop()

        if candidate_func.name not in checked:
            checked[candidate_func.name] = set()
        elif candidate_param_idx in checked[candidate_func.name]:
            continue

        checked[candidate_func.name].add(candidate_param_idx)

        for called_func, called_param_idx in findCalled2(candidate_param_idx, candidate_func, funcs):
            if called_func.name in checked and called_param_idx in checked[called_func.name]:
                continue

            stack.append((called_func, called_param_idx))

    called_list = []
    for func_name, indicies in checked.items():
        for index in indicies:
            called_list.append((find(funcs, lambda f: f.name == func_name), index))

    return called_list


def findCalled2(param_idx: int, func: Function, funcs: List[Function]) -> List[Tuple[Function, int]]:
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
        arg_idx = parent.items.index(node)

        if func is not None:
            return (func, arg_idx)

    return None
