from typing import Callable, List, Optional, Dict, Set, Tuple, Any

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP

from store import Application, Entity, Function
from util import find, safeFind


def extractDependencies(entities: List[Entity], app: Application, scope: List[str] = []) -> List[Entity]:
    unprocessed_entities = list(entities)
    extracted_entities = []

    while len(unprocessed_entities) > 0:
        entity = unprocessed_entities.pop(0)

        if safeFind(extracted_entities, lambda e: e == entity):
            continue

        for dependency in entity.depends:
            dependency_entities = app.findEntities(dependency, entity.program, scope)  # TODO: Loop scope
            unprocessed_entities.extend(dependency_entities)

        if not safeFind(entities, lambda e: e == entity):
            extracted_entities.insert(0, entity)

    return extracted_entities


# TODO: types
def renameEntities(entities: List[Entity], replacement: Callable[[str], str]) -> None:
    for entity in entities:
        new_name = replacement(entity.name)
        renameFunctionDefinition(entity, new_name)

        for entity2 in entities:
            renameFunctionCalls(entity2, entity.name, new_name)


def renameFunctionDefinition(entity: Entity, replacement: str) -> None:
    subroutine_statement = fpu.get_child(entity.ast, f2003.Subroutine_Stmt)
    kernel_name = fpu.get_child(subroutine_statement, f2003.Name)

    kernel_name.string = replacement


def renameFunctionCalls(entity: Entity, name: str, replacement: str) -> None:
    for node in fpu.walk(entity.ast, f2003.Call_Stmt):
        name_node = fpu.get_child(node, f2003.Name)

        if name_node.string == name:
            name_node.string = replacement


def renameConsts(entities: List[Entity], app: Application, replacement: Callable[[str], str]) -> None:
    const_ptrs = set(map(lambda const: const.ptr, app.consts()))

    for entity in entities:
        for name in fpu.walk(entity.ast, f2003.Name):
            if name.string in const_ptrs:
                name.string = replacement(name.string)


def insertStrides(
    func: Function,
    funcs: List[Function],
    loop: OP.Loop,
    app: Application,
    stride: Callable[[str], str],
    match: Callable[[OP.ArgDat], bool] = lambda arg: True,
    modified: Dict[str, Set[int]] = {},
) -> Dict[str, Set[int]]:
    if func.name not in modified:
        modified[func.name] = set()

    stack = []
    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        if arg_idx in modified[func.name]:
            continue

        stack.append((func.name, arg_idx, loop.args[arg_idx]))

    while len(stack) > 0:
        item = stack.pop()
        target_func = safeFind(funcs, lambda f: f.name == item[0])

        if target_func is None:
            print(f"{loop.loc}: Warning: Unable to insert strides into unknown function {item[0]}, arg {item[1]}")
            continue

        called = insertStride(target_func, funcs, target_func.parameters[item[1]], item[2], stride)
        modified[target_func.name].add(item[1])

        for item2 in called:
            if item2[0] not in modified:
                modified[item2[0]] = set()
                stack.append(item2)
                continue

            if item2[1] in modified[item2[0]]:
                continue

            if item2 in stack:
                continue

            stack.append(item2)

    return modified


def insertStride(
    func: Function, funcs: List[Function], param: str, arg: OP.Arg, stride: Callable[[str], str]
) -> List[Tuple[str, int, OP.Arg]]:
    called = []

    for node in fpu.walk(func.ast, f2003.Name):
        if node.string != param:
            continue

        parent = node.parent

        if isinstance(parent, f2003.Part_Ref):
            subscript_list = fpu.get_child(parent, f2003.Section_Subscript_List)

            parent.items = (
                node,
                f2003.Section_Subscript_List(f"op2_s({str(subscript_list)}, {stride(arg)})"),
            )

            node = node.parent
            parent = node.parent

        if isinstance(parent, f2003.Actual_Arg_Spec_List):
            func_name_node = fpu.get_child(parent.parent, f2003.Name)

            if func_name_node is None:
                continue

            arg_idx = parent.items.index(node)
            called.append((func_name_node.string, arg_idx, arg))

        # Function calls turn up as Part_Refs with a Section_Subscript_List, but these might
        # also be genuine array indexes
        if isinstance(parent, f2003.Section_Subscript_List):
            func_name_node = fpu.get_child(parent.parent, f2003.Name)

            if func_name_node is None:
                continue

            if safeFind(funcs, lambda f: f.name == func_name_node.string) is None:
                continue

            arg_idx = parent.items.index(node)
            called.append((func_name_node.string, arg_idx, arg))

    return called


def insertAtomicIncs(
    func: Function,
    funcs: List[Function],
    loop: OP.Loop,
    app: Application,
    match: Callable[[OP.ArgDat], bool] = lambda arg: True,
) -> Dict[str, Set[int]]:
    modified = {}

    if func.name not in modified:
        modified[func.name] = set()

    stack = []
    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        if arg_idx in modified[func.name]:
            continue

        stack.append((func.name, arg_idx, loop.args[arg_idx]))

    while len(stack) > 0:
        item = stack.pop()
        target_func = safeFind(funcs, lambda f: f.name == item[0])

        if target_func is None:
            print(f"{loop.loc}: Warning: Unable to atomics into unknown function {item[0]}, arg {item[1]}")
            continue

        called = insertAtomicInc(target_func, funcs, target_func.parameters[item[1]], item[2])
        modified[target_func.name].add(item[1])

        for item2 in called:
            if item2[0] not in modified:
                modified[item2[0]] = set()
                stack.append(item2)
                continue

            if item2[1] in modified[item2[0]]:
                continue

            if item2 in stack:
                continue

            stack.append(item2)

    for func_name in modified.keys():
        if len(modified[func_name]) == 0:
            continue

        func = find(funcs, lambda f: f.name == func_name)
        spec = fpu.walk(func.ast, f2003.Specification_Part)[0]

        spec.children.append(f2003.Type_Declaration_Stmt("integer(4) :: op2_ret"))

def insertAtomicInc(func: Function, funcs: List[Function], param: str, arg: OP.Arg) -> List[Tuple[str, int, OP.Arg]]:
    called = []

    for node in fpu.walk(func.ast, f2003.Name):
        if node.string != param:
            continue

        parent = node.parent

        if isinstance(parent, f2003.Part_Ref):
            node = node.parent
            parent = node.parent

        if isinstance(parent, f2003.Actual_Arg_Spec_List):
            func_name_node = fpu.get_child(parent.parent, f2003.Name)

            if func_name_node is None:
                continue

            arg_idx = parent.items.index(node)
            called.append((func_name_node.string, arg_idx, arg))

        # Function calls turn up as Part_Refs with a Section_Subscript_List, but these might
        # also be genuine array indexes
        if isinstance(parent, f2003.Section_Subscript_List):
            func_name_node = fpu.get_child(parent.parent, f2003.Name)

            if func_name_node is None:
                continue

            if safeFind(funcs, lambda f: f.name == func_name_node.string) is None:
                continue

            arg_idx = parent.items.index(node)
            called.append((func_name_node.string, arg_idx, arg))

    modified = insertAtomicInc2(func.ast, param)
    return called


def isRef(node: Any, name: str) -> bool:
    if isinstance(node, f2003.Name) and node.string.lower() == name.lower():
        return True

    if isinstance(node, f2003.Part_Ref) and node.items[0].string.lower() == name.lower():
        return True

    return False


def insertAtomicInc2(node: Any, param: str) -> Tuple[Optional[Any], bool]:
    if isinstance(node, f2003.Assignment_Stmt):
        if not isRef(node.items[0], param):
            return None, False

        if not isinstance(node.items[2], f2003.Level_2_Expr):
            raise OpError("Error: unexpected RHS while inserting atomics: {node.items[2]}")

        op = node.items[2].items[1]

        if op == "+":
            if str(node.items[2].items[0]) == str(node.items[0]):
                term = str(node.items[2].items[2])
            elif str(node.items[2].items[2]) == str(node.items[0]):
                term = str(node.items[2].items[0])
            else:
                raise OpError("Error: unexpected RHS while inserting atomics: {node.items[2]}")

            return f2003.Assignment_Stmt(f"op2_ret = atomicAdd({node.items[0]}, {term})"), False

        if op == "-":
            if str(node.items[2].items[0]) == str(node.items[0]):
                term = str(node.items[2].items[2])
            else:
                raise OpError("Error: unexpected RHS while inserting atomics: {node.items[2]}")

            return f2003.Assignment_Stmt(f"op2_ret = atomicSub({node.items[0]}, {term})"), False

        raise OpError("Error: unexpected RHS while inserting atomics: {node.items[2]}")

    if not isinstance(node, f2003.Base):
        return None, False

    modified = False
    for i in range(len(node.children)):
        if node.children[i] is None:
            continue

        replacement, modified2 = insertAtomicInc2(node.children[i], param)

        if replacement is not None:
            node.children[i] = replacement

        if replacement is not None or modified2:
            modified = True

    return None, modified



def writeSource(entities: List[Entity], prologue: Optional[str] = None) -> str:
    if len(entities) == 0:
        return ""

    source = (prologue or "") + str(entities[-1].ast)
    for entity in reversed(entities[:-1]):
        source = source + "\n\n" + (prologue or "") + str(entity.ast)

    return source
