from pathlib import Path
from typing import Any, Dict, Set, Tuple, List

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from fortran.parser import findSubroutine
from language import Lang
from store import Application, Kernel
from util import find


def translateKernel(
    lang: Lang, include_dirs: Set[Path], defines: List[str], config: Dict[str, Any], kernel: Kernel, app: Application
) -> str:
    ast = lang.parseFile(kernel.path, frozenset(include_dirs), defines)

    kernel_ast = findSubroutine(kernel.path, ast, kernel.name)
    assert kernel_ast is not None

    subroutine_statement = fpu.get_child(kernel_ast, f2003.Subroutine_Stmt)
    kernel_name = fpu.get_child(subroutine_statement, f2003.Name)

    kernel_name.string = kernel_name.string + "_seq"

    const_ptrs = set(map(lambda const: const.ptr, app.consts()))
    for name in fpu.walk(kernel_ast, f2003.Name):
        if name.string in const_ptrs:
            name.string = f"op2_const_{name.string}"

    loop = find(app.loops(), lambda loop: loop.kernel == kernel.name)

    for arg_idx in range(len(loop.args)):
        if not isinstance(loop.args[arg_idx], OP.ArgDat):
            continue

        dat = find(app.dats(), lambda dat: dat.ptr == loop.args[arg_idx].dat_ptr)
        if not dat.soa:
            continue

        insertStride(kernel.params[arg_idx][0], dat.ptr, kernel_ast)

    return str(kernel_ast)


def insertStride(param: str, dat_ptr: str, kernel_ast: f2003.Subroutine_Subprogram) -> None:
    for name in fpu.walk(kernel_ast, f2003.Name):
        if name.string != param:
            continue

        parent = name.parent
        if not isinstance(name.parent, f2003.Part_Ref):
            continue

        subscript_list = fpu.get_child(parent, f2003.Section_Subscript_List)

        parent.items = (
            name,
            f2003.Section_Subscript_List(f"((({str(subscript_list)}) - 1) * op2_dat_{dat_ptr}_stride + 1)"),
        )
