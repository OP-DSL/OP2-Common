from pathlib import Path
from typing import Callable, List, Set

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from fortran.parser import findSubroutine
from language import Lang
from store import Application, Kernel
from util import find


def findKernel(lang: Lang, kernel: Kernel, include_dirs: Set[Path], defines: List[str]) -> f2003.Subroutine_Subprogram:
    ast = lang.parseFile(kernel.path, frozenset(include_dirs), frozenset(defines))

    kernel_ast = findSubroutine(kernel.path, ast, kernel.name)
    assert kernel_ast is not None

    return kernel_ast


def renameKernel(kernel_ast: f2003.Subroutine_Subprogram, replacement: Callable[[str], str]) -> None:
    subroutine_statement = fpu.get_child(kernel_ast, f2003.Subroutine_Stmt)
    kernel_name = fpu.get_child(subroutine_statement, f2003.Name)

    kernel_name.string = replacement(kernel_name.string)


def renameConsts(kernel_ast: f2003.Subroutine_Subprogram, app: Application, replacement: Callable[[str], str]) -> None:
    const_ptrs = set(map(lambda const: const.ptr, app.consts()))

    for name in fpu.walk(kernel_ast, f2003.Name):
        if name.string in const_ptrs:
            name.string = replacement(name.string)


def insertStrides(
    kernel_ast: f2003.Subroutine_Subprogram,
    kernel: Kernel,
    app: Application,
    stride: Callable[[str], str],
    match: Callable[[OP.ArgDat], bool] = lambda arg: True,
) -> None:
    loop = find(app.loops(), lambda loop: loop.kernel == kernel.name)

    for arg_idx in range(len(loop.args)):
        if not match(loop.args[arg_idx]):
            continue

        insertStride(kernel.params[arg_idx][0], loop.args[arg_idx], kernel_ast, stride)


def insertStride(
    param: str, arg: OP.Arg, kernel_ast: f2003.Subroutine_Subprogram, stride: Callable[[str], str]
) -> None:
    for name in fpu.walk(kernel_ast, f2003.Name):
        if name.string != param:
            continue

        parent = name.parent
        if not isinstance(name.parent, f2003.Part_Ref):
            continue

        subscript_list = fpu.get_child(parent, f2003.Section_Subscript_List)

        parent.items = (
            name,
            f2003.Section_Subscript_List(f"((({str(subscript_list)}) - 1) * {stride(arg)} + 1)"),
        )
