from typing import List, Optional, Any

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import fortran.translator.kernels as ftk
import op as OP
from op import OpError
from store import Application, Entity, Function, Program
from util import safeFind


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

        written, in_subcall = paramIsWritten(idx, kernel_entities[0], entities)
        param_name = kernel_entities[0].parameters[idx]

        if written:
            msg = f"Warning: arg {idx + 1} ({param_name}) of {loop.kernel} marked OP_READ but was written"
            if in_subcall:
                msg = msg + " (in called subroutine)"

            print(msg)

def isRef(node: Any, name: str) -> bool: 
    if isinstance(node, f2003.Name) and node.string == name:
        return True

    if isinstance(node, f2003.Part_Ref) and node.items[0].string == name:
        return True

    return False

def paramIsWritten(param_idx: int, func: Function, funcs: List[Function]) -> (bool, bool):
    assert isinstance(func.ast, f2003.Subroutine_Subprogram)
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

        name = name_node.string
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
