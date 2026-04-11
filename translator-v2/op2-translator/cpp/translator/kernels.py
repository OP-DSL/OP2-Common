from typing import Callable, Dict, List, Optional, Set, Tuple

from clang.cindex import Cursor, CursorKind, SourceRange

import op as OP
from store import Application, Entity, Function, Type
from util import Location, Rewriter, Span, find, safeFind


def extentToSpan(extent: SourceRange) -> Span:
    start = Location(extent.start.line, extent.start.column)
    end = Location(extent.end.line, extent.end.column)

    return Span(start, end)


def extractDependencies(entities: List[Entity], app: Application) -> List[Tuple[Entity, Rewriter]]:
    unprocessed_entities = list(entities)
    extracted_entities = []

    while len(unprocessed_entities) > 0:
        entity = unprocessed_entities.pop(0)

        if safeFind(extracted_entities, lambda e: e[0] == entity):
            continue

        for dependency in entity.depends:
            dependency_entities = app.findEntities(dependency, entity.program)
            unprocessed_entities.extend(dependency_entities)

        rewriter = Rewriter(entity.program.source, [extentToSpan(entity.ast.extent)])
        extracted_entities.insert(0, (entity, rewriter))

    return extracted_entities


def updateFunctionTypes(entities: List[Tuple[Entity, Rewriter]], replacement: Callable[[str, Entity], str]) -> None:
    for entity, rewriter in filter(lambda e: isinstance(e[0], Function), entities):
        function_type_span = extentToSpan(next(entity.ast.get_tokens()).extent)
        rewriter.update(function_type_span, lambda s: replacement(s, entity))


def renameConsts(
    entities: List[Tuple[Entity, Rewriter]], app: Application, replacement: Callable[[str, Entity], str]
) -> None:
    const_ptrs = app.constPtrs()

    for entity, rewriter in entities:
        for node in entity.ast.walk_preorder():
            if node.kind != CursorKind.DECL_REF_EXPR:
                continue

            if node.spelling in const_ptrs:
                rewriter.update(extentToSpan(node.extent), lambda s: replacement(s, entity))


def mapParam(
    func: Function,
    param_idx: int,
    funcs: List[Function],
    op: Callable[[Function, int], Optional[bool]],
    *args,
) -> None:
    called_list = findCalled(func, param_idx, funcs)

    for func2, param_idx2 in called_list:
        stop = op(func2, param_idx2, *args)
        if stop:
            break


def findCalled(func: Function, param_idx: int, funcs: List[Function]) -> List[Tuple[Function, int]]:
    checked: Dict[str, Set[int]] = {}
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
    called: List[Tuple[Function, int]] = []

    if param_idx >= len(func.parameters):
        return called

    param = func.parameters[param_idx]

    for node in func.ast.walk_preorder():
        if node.kind != CursorKind.CALL_EXPR:
            continue

        callee = node.get_definition() or node.referenced
        if callee is None:
            continue

        callee_func = safeFind(funcs, lambda f: f.name == callee.spelling)
        if callee_func is None:
            continue

        for arg_idx, arg in enumerate(node.get_arguments()):
            if isDirectParamRef(arg, param):
                called.append((callee_func, arg_idx))

    return called


def isDirectParamRef(node: Cursor, param: str) -> bool:
    node = stripImplicit(node)
    return node.kind == CursorKind.DECL_REF_EXPR and node.spelling == param


def stripImplicit(node: Cursor) -> Cursor:
    implicit_kinds = []
    for kind_name in [
        "UNEXPOSED_EXPR",
        "IMPLICIT_CAST_EXPR",
        "PAREN_EXPR",
        "CSTYLE_CAST_EXPR",
        "CXX_STATIC_CAST_EXPR",
        "CXX_REINTERPRET_CAST_EXPR",
        "CXX_CONST_CAST_EXPR",
        "CXX_FUNCTIONAL_CAST_EXPR",
    ]:
        kind = getattr(CursorKind, kind_name, None)
        if kind is not None:
            implicit_kinds.append(kind)

    while node.kind in implicit_kinds:
        children = list(node.get_children())
        if len(children) != 1:
            break

        node = children[0]

    return node


def insertStrides(
    entity: Entity,
    rewriter: Rewriter,
    app: Application,
    loop: OP.Loop,
    stride: Callable[[int], str],
    skip: Optional[Callable[[OP.ArgDat], bool]] = None,
    entities: Optional[List[Tuple[Entity, Rewriter]]] = None,
) -> None:
    if not isinstance(entity, Function):
        return

    if entities is None:
        entities = [(entity, rewriter)]

    funcs = [e for e, _ in entities if isinstance(e, Function)]
    rewriters = {e.name: r for e, r in entities if isinstance(e, Function)}

    for arg_idx in range(len(loop.args)):
        arg = loop.args[arg_idx]
        if not isinstance(arg, OP.ArgDat):
            continue

        if skip is not None and skip(arg):
            continue

        dat = loop.dats[arg.dat_id]
        if not dat.soa:
            continue

        is_vec = arg.map_idx is not None and arg.map_idx < -1

        def applyStride(func: Function, param_idx: int, dat_id: int, is_vec: bool) -> None:
            if param_idx >= len(func.parameters):
                return

            func_rewriter = rewriters.get(func.name)
            if func_rewriter is None:
                return

            insertStride(func.ast, func_rewriter, func.parameters[param_idx], dat_id, is_vec, stride)

        mapParam(entity, arg_idx, funcs, applyStride, dat.id, is_vec)

def insertArgGblStrides(
    entity: Entity,
    rewriter: Rewriter,
    app: Application,
    loop: OP.Loop,
    stride: Callable[[int], str],
    skip: Optional[Callable[[OP.ArgDat], bool]] = None,
) -> None:

    if not isinstance(entity, Function):
        return

    for arg_idx in range(len(loop.args)):
        arg = loop.args[arg_idx]
        if not isinstance(arg, OP.ArgGbl):
            continue

        if skip is not None and skip(arg):
            continue

        insertStride(entity.ast, rewriter, entity.parameters[arg_idx], 0, False, stride)

def insertStride(
    ast: Cursor, rewriter: Rewriter, param: str, id: int, is_vec: bool, stride: Callable[[int], str]
) -> None:
    for node in ast.walk_preorder():
        if node.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
            continue

        ident, subscript = node.get_children()
        if is_vec:
            if next(ident.get_children()).kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
                continue

            ident, _ = next(ident.get_children()).get_children()

        if ident.spelling == param:
            rewriter.update(extentToSpan(subscript.extent), lambda s: f"({s}) * {stride(id)}")


def writeSource(entities: List[Tuple[Entity, Rewriter]]) -> str:
    source = ""
    while len(entities) > 0:
        for i in range(len(entities)):
            entity, rewriter = entities[i]
            resolved = True

            for dependency in entity.depends:
                if safeFind(entities, lambda e: e[0].name == dependency):
                    resolved = False
                    break

            if resolved:
                entities.pop(i)
                if source == "":
                    source = rewriter.rewrite()
                else:
                    source = source + "\n\n" + rewriter.rewrite()

                if isinstance(entity, Type):
                    source = source + ";"

                break

    return source
