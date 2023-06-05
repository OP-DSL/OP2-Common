from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import fortran.util as fu
import op as OP
from language import Lang
from op import OpError
from store import Application, Entity, Function
from util import find, safeFind


def extractDependencies(
    entities: List[Entity], app: Application, scope: List[str] = []
) -> Tuple[List[Entity], List[str]]:
    unprocessed_entities = list(entities)
    extracted_entities = []
    unknown_entities = []

    while len(unprocessed_entities) > 0:
        entity = unprocessed_entities.pop(0)

        if safeFind(extracted_entities, lambda e: e == entity):
            continue

        for dependency in entity.depends:
            if dependency in ["hyd_print", "hyd_dump", "hyd_kill", "hyd_error_print", "hyd_error_dump"]:
                continue

            dependency_entities = app.findEntities(dependency, entity.program, scope)  # TODO: Loop scope

            if len(dependency_entities) == 0:
                unknown_entities.append(dependency)
            else:
                unprocessed_entities.extend(dependency_entities)

        if not safeFind(entities, lambda e: e == entity):
            extracted_entities.insert(0, entity)

    return extracted_entities, unknown_entities


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

        if name_node.string.lower() == name:
            name_node.string = replacement


def renameConsts(lang: Lang, entities: List[Entity], app: Application, replacement: Callable[[str], str]) -> None:
    const_ptrs = app.constPtrs()

    for entity in entities:
        for name in fpu.walk(entity.ast, f2003.Name):
            if name.string.lower() in const_ptrs and name.string.lower() not in entity.parameters:
                name.string = replacement(name.string.lower())


def fixHydraIO(func: Function) -> None:
    replaceNodes(func.ast, lambda n: isinstance(n, f2003.Write_Stmt), f2003.Continue_Stmt("continue"))

    def match_hyd_call(n):
        if not isinstance(n, f2003.Call_Stmt):
            return False

        name_node = fpu.get_child(n, f2003.Name)
        return name_node.string.lower() in ["hyd_print", "hyd_dump", "hyd_kill", "hyd_error_print", "hyd_error_dump"]

    replaceNodes(func.ast, match_hyd_call, f2003.Stop_Stmt("stop"))


def removeExternals(func: Function) -> None:
    for spec in fpu.walk(func.ast, f2003.Specification_Part):
        content = list(spec.content)
        content = filter(lambda n: not isinstance(n, f2003.External_Stmt), content)
        spec.content = list(content)


def insertStrides(
    func: Function,
    funcs: List[Function],
    loop: OP.Loop,
    app: Application,
    stride: Callable[[str], str],
    match: Callable[[OP.ArgDat], bool] = lambda arg: True,
    modified: Optional[Dict[str, Set[int]]] = None,
):
    if modified is None:
        modified = {}

    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        if arg_idx in modified.get(func.name, set()):
            continue

        fu.mapParam(func, arg_idx, funcs, insertStride, modified, stride(loop.args[arg_idx]))

    return modified


def insertStride(func: Function, param_idx: int, modified: Dict[str, Set[int]], stride: str) -> None:
    if func.name not in modified:
        modified[func.name] = {param_idx}
    elif param_idx not in modified[func.name]:
        modified[func.name].add(param_idx)
    else:
        return

    param = func.parameters[param_idx]
    for node in fpu.walk(func.ast, f2003.Name):
        if node.string.lower() != param:
            continue

        parent = node.parent
        if not isinstance(parent, f2003.Part_Ref):
            continue

        subscript_list = fpu.get_child(parent, f2003.Section_Subscript_List)

        if len(subscript_list.children) > 1:
            dims = fu.parseDimensions(func, param)

            if dims is None or len(dims) != len(subscript_list.children):
                raise OpError(f"Unexpected dimension mismatch ({subscript_list}, {dims})")

            index = str(subscript_list.children[0])
            sizes = [f"(1 + {dim[1]} - ({dim[0]}))" for dim in dims]
            for i, extra_index in enumerate(subscript_list.children[1:]):
                index += f" + ({extra_index} - ({dims[i + 1][0]})) * {'*'.join(sizes[:i + 1])}"
        else:
            index = str(subscript_list)

        parent.items = [
            node,
            f2003.Section_Subscript_List(f"op2_s({index}, {stride})"),
        ]

        parent.items[1].parent = parent


def insertAtomicIncs(
    func: Function,
    funcs: List[Function],
    loop: OP.Loop,
    app: Application,
    match: Callable[[OP.ArgDat], bool] = lambda arg: True,
) -> Dict[str, Set[int]]:
    modified = {}

    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        if arg_idx in modified.get(func.name, set()):
            continue

        fu.mapParam(func, arg_idx, funcs, insertAtomicInc, modified)

    for func_name in modified.keys():
        if len(modified[func_name]) == 0:
            continue

        func = find(funcs, lambda f: f.name == func_name)
        spec = fpu.walk(func.ast, f2003.Specification_Part)[0]

        spec.children.append(f2003.Type_Declaration_Stmt("integer(4) :: op2_ret"))


def insertAtomicInc(func: Function, param_idx: int, modified: Dict[str, Set[int]]) -> None:
    if param_idx in modified.get(func.name, set()):
        return

    _, replaced = insertAtomicInc2(func.ast, func.parameters[param_idx])

    if not replaced:
        return

    if func.name not in modified:
        modified[func.name] = {param_idx}
    else:
        modified[func.name].add(param_idx)


def insertAtomicInc2(node: f2003.Base, param: str) -> Tuple[Optional[Any], bool]:
    if isinstance(node, f2003.Assignment_Stmt):
        if not fu.isRef(node.items[0], param):
            return None, False

        if not isinstance(node.items[2], f2003.Level_2_Expr):
            raise OpError(f"Error: unexpected statement while inserting atomics: {node}")

        replaceNodes(node.items[2], lambda n: str(n) == str(node.items[0]), f2003.Int_Literal_Constant("0"))
        return f2003.Assignment_Stmt(f"op2_ret = atomicAdd({node.items[0]}, {node.items[2]})"), False

    if not isinstance(node, f2003.Base):
        return None, False

    modified = False
    for i in range(len(node.children)):
        if node.children[i] is None:
            continue

        replacement, modified2 = insertAtomicInc2(node.children[i], param)

        if replacement is not None:
            replaceChild(node, i, replacement)

        if replacement is not None or modified2:
            modified = True

    return None, modified


def replaceChild(node: f2003.Base, index: int, replacement: Any) -> None:
    children = []
    use_tuple = False
    use_content = False

    if hasattr(replacement, "parent"):
        replacement.parent = node

    if getattr(node, "items", None) is not None:
        if isinstance(node.items, tuple):
            use_tuple = True

        children = list(node.items)
    else:
        if isinstance(node.content, tuple):
            use_tuple = True

        children = list(node.content)
        use_content = True

    children[index] = replacement

    if use_tuple:
        children = tuple(children)

    if not use_content:
        node.items = children
    else:
        node.content = children


def replaceNodes(node: Any, match: Callable[[f2003.Base], bool], replacement: f2003.Base) -> Optional[Any]:
    if isinstance(node, tuple) or isinstance(node, list):
        children = list(node)

        for i in range(len(children)):
            if children[i] is None:
                continue

            child_replacement = replaceNodes(children[i], match, replacement)
            if child_replacement is not None:
                children[i] = child_replacement

        if isinstance(node, tuple):
            return tuple(children)
        else:
            return children

    if not isinstance(node, f2003.Base):
        return None

    if match(node):
        return replacement

    for i in range(len(node.children)):
        if node.children[i] is None:
            continue

        child_replacement = replaceNodes(node.children[i], match, replacement)

        if child_replacement is not None:
            replaceChild(node, i, child_replacement)

    return None


def writeSource(entities: List[Entity], prologue: Optional[str] = None) -> str:
    if len(entities) == 0:
        return ""

    source = (prologue or "") + str(entities[-1].ast)
    for entity in reversed(entities[:-1]):
        source = source + "\n\n" + (prologue or "") + str(entity.ast)

    return source
