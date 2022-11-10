from typing import Callable, List, Optional, Tuple

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
    const_ptrs = set(map(lambda const: const.ptr, app.consts()))

    for entity, rewriter in entities:
        for node in entity.ast.walk_preorder():
            if node.kind != CursorKind.DECL_REF_EXPR:
                continue

            if node.spelling in const_ptrs:
                rewriter.update(extentToSpan(node.extent), lambda s: replacement(s, entity))


def insertStrides(
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
        if not isinstance(loop.args[arg_idx], OP.ArgDat):
            continue

        if skip is not None and skip(loop.args[arg_idx]):
            continue

        dat = loop.dats[loop.args[arg_idx].dat_id]
        if not dat.soa:
            continue

        is_vec = loop.args[arg_idx].map_idx < -1
        insertStride(entity.ast, rewriter, entity.parameters[arg_idx], dat.id, is_vec, stride)


def insertStride(
    ast: Cursor, rewriter: Rewriter, param: str, dat_id: int, is_vec: bool, stride: Callable[[int], str]
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
            rewriter.update(extentToSpan(subscript.extent), lambda s: f"({s}) * {stride(dat_id)}")


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
