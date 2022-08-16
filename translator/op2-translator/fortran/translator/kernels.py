from typing import Callable, List

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
    entity: Entity,
    loop: OP.Loop,
    app: Application,
    stride: Callable[[str], str],
    match: Callable[[OP.ArgDat], bool] = lambda arg: True,
) -> None:
    if not isinstance(entity, Function):
        return

    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        insertStride(entity, entity.parameters[arg_idx][0], loop.args[arg_idx], stride)


def insertStride(entity: Entity, param: str, arg: OP.Arg, stride: Callable[[str], str]) -> None:
    for name in fpu.walk(entity.ast, f2003.Name):
        if name.string != param:
            continue

        parent = name.parent
        if not isinstance(name.parent, f2003.Part_Ref):
            continue

        subscript_list = fpu.get_child(parent, f2003.Section_Subscript_List)

        parent.items = (
            name,
            f2003.Section_Subscript_List(f"(({str(subscript_list)}) - 1) * {stride(arg)} + 1"),
        )


def writeSource(entities: List[Entity]) -> str:
    if len(entities) == 0:
        return ""

    source = str(entities[-1].ast)
    for entity in reversed(entities[:-1]):
        source = source + "\n\n" + str(entity.ast)

    return source
