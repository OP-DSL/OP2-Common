from typing import List, Optional

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import fortran.translator.kernels as ftk
import op as OP
from op import OpError
from store import Application, Entity, Function, Program


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

    for idx, arg in enumerate(loop.args):
        if isinstance(arg, OP.ArgIdx):
            continue

        if arg.access_type != OP.AccessType.READ:
            continue

        written = checkForWrites(idx, kernel_entities[0], entities)

        if written:
            param_name = kernel_entities[0].parameters[idx]
            print(f"Warning: arg {idx} ({param_name}) of {loop.kernel} marked OP_READ but was written")


def checkForWrites(param_idx: int, func: Function, funcs: List[Function]) -> bool:
    assert isinstance(func.ast, f2003.Subroutine_Subprogram)
    param_name = func.parameters[param_idx]

    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    for node in fpu.walk(execution_part, f2003.Assignment_Stmt):
        lhs = node.items[0]

        if isinstance(lhs, f2003.Name) and lhs.string == param_name:
            return True

        if isinstance(lhs, f2003.Part_Ref) and lhs.items[0].string == param_name:
            return True

    return False
