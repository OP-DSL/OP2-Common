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
    erase_dimensions = False

    for node in fpu.walk(func.ast, f2003.Name):
        if node.string.lower() != param:
            continue

        parent = node.parent
        if not isinstance(parent, f2003.Part_Ref):
            continue

        subscript_list = fpu.get_child(parent, f2003.Section_Subscript_List)

        dims = fu.parseDimensions(func, param)

        if dims is None:
            dims = [("1", None)]
        else:
            erase_dimensions = True

        if len(dims) != len(subscript_list.children):
            raise OpError(f"Unexpected dimension mismatch ({subscript_list}, {dims})")

        sizes = []
        for dim in dims:
            if dim[0] == "1":
                sizes.append(f"({dim[1]})")
            else:
                sizes.append(f"(1 + {dim[1]} - ({dim[0]}))")

        if dims[0][0] == "1":
            index = f"{subscript_list.children[0]}"
        else:
            index = f"({subscript_list.children[0]} + 1 - ({dims[0][0]}))"

        for i, extra_index in enumerate(subscript_list.children[1:], start=1):
            index += f" + ({extra_index} - ({dims[i][0]})) * {'*'.join(sizes[:i])}"

        parent.items = [
            node,
            f2003.Part_Ref(f"op2_s({index}, {stride})"),
        ]

        parent.items[1].parent = parent

    if erase_dimensions:
        fu.eraseDimensions(func, param)


def insertAtomicIncs(
    func: Function,
    funcs: List[Function],
    loop: OP.Loop,
    app: Application,
    match: Callable[[OP.Arg], bool],
    c_api: bool = False
) -> Dict[str, Set[int]]:
    modified = {}

    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        if arg_idx in modified.get(func.name, set()):
            continue

        if isinstance(loop.args[arg_idx], OP.ArgDat):
            typ = loop.dat(loop.args[arg_idx]).typ
        elif hasattr(loop.args[arg_idx], "typ"):
            typ = loop.args[arg_idx].typ
        else:
            raise OpError(f"Error: could not find type of arg while inserting atomics: {loop.args[arg_idx]}")

        fu.mapParam(func, arg_idx, funcs, insertAtomicInc, modified, typ, c_api)

    if c_api:
        return

    for func_name in modified.keys():
        if len(modified[func_name]) == 0:
            continue

        func = find(funcs, lambda f: f.name == func_name)
        spec = fpu.walk(func.ast, f2003.Specification_Part)[0]

        spec.children.append(f2003.Type_Declaration_Stmt("integer(4) :: op2_ret"))


def insertAtomicInc(func: Function, param_idx: int, modified: Dict[str, Set[int]], typ: OP.Type, c_api: bool) -> None:
    if param_idx in modified.get(func.name, set()):
        return

    _, replaced = insertAtomicInc2(func.ast, func.parameters[param_idx], typ, c_api)

    if not replaced:
        return

    if func.name not in modified:
        modified[func.name] = {param_idx}
    else:
        modified[func.name].add(param_idx)


def insertAtomicInc2(node: f2003.Base, param: str, typ: OP.Type, c_api: bool) -> Tuple[Optional[Any], bool]:
    if isinstance(node, f2003.Assignment_Stmt):
        if not fu.isRef(node.items[0], param):
            return None, False

        if not isinstance(node.items[2], f2003.Level_2_Expr):
            raise OpError(f"Error: unexpected statement while inserting atomics: {node}")

        if isinstance(typ, OP.Int):
            literal = f2003.Int_Literal_Constant("0")
        elif isinstance(typ, OP.Float) and typ.size == 32:
            literal = f2003.Real_Literal_Constant("0.0")
        elif isinstance(typ, OP.Float) and typ.size == 64:
            literal = f2003.Real_Literal_Constant("0.0d0")
        else:
            raise OpError(f"Error: unexpected arg type while inserting atomics: {typ}")

        replaceNodes(node.items[2], lambda n: str(n) == str(node.items[0]), literal)

        if not c_api:
            return f2003.Assignment_Stmt(f"op2_ret = atomicAdd({node.items[0]}, {node.items[2]})"), False
        else:
            return f2003.Call_Stmt(f"call atomicAdd({node.items[0]}, {node.items[2]})"), False

    if not isinstance(node, f2003.Base):
        return None, False

    modified = False
    for i in range(len(node.children)):
        if node.children[i] is None:
            continue

        replacement, modified2 = insertAtomicInc2(node.children[i], param, typ, c_api)

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


# op2_s macro aware line split
def splitLine(line: str, max_length: int = 264) -> str:
    if len(line) <= max_length:
        return (line, "")

    i = max_length - 1
    depth = 0

    while i >= 0:
        if line[i] == ')':
            depth += 1
        elif line[i] == '(':
            depth -= 1

        if i >= max_length - 5 and line[i:].startswith('op2_s'):
            return (line[:i], line[i:])

        if line[:i + 1].endswith('op2_s'):
            if depth >= 0 and i != max_length - 1:
                return (line[:max_length], line[max_length:])
            else:
                return (line[:i - 4], line[i - 4:])

        i -= 1

    return (line[:max_length], line[max_length:])


def addLineContinuations(source: str, max_length: int = 264) -> str:
    lines = source.splitlines()

    source2 = ""
    first_line = True

    for line in lines:
        if len(line) > max_length:
            new_line, remainder = splitLine(line, max_length - 1)

            while len(remainder) > 0:
                new_line2, remainder = splitLine(remainder, max_length - 2)
                new_line += "&\n&" + new_line2
        else:
            new_line = line

        source2 += ("" if first_line else "\n") + new_line
        first_line = False

    return source2


def writeSource(entities: List[Entity], prologue: Optional[str] = None) -> str:
    if len(entities) == 0:
        return ""

    source = (prologue or "") + addLineContinuations(str(entities[-1].ast))
    for entity in reversed(entities[:-1]):
        source += "\n\n" + (prologue or "") + addLineContinuations(str(entity.ast))

    return source
