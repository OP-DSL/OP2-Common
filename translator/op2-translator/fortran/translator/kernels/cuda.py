from pathlib import Path
from typing import Any, Dict, Set, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import op as OP
from fortran.parser import findSubroutine
from language import Lang
from store import Application, Kernel
from util import find


def translateKernel(
    lang: Lang, include_dirs: Set[Path], config: Dict[str, Any], kernel: Kernel, app: Application
) -> str:
    ast = lang.parseFile(kernel.path, frozenset(include_dirs))

    kernel_ast = findSubroutine(kernel.path, ast, kernel.name)
    assert kernel_ast is not None

    subroutine_statement = fpu.get_child(kernel_ast, f2003.Subroutine_Stmt)
    kernel_name = fpu.get_child(subroutine_statement, f2003.Name)

    kernel_name.string = kernel_name.string + "_gpu"

    const_ptrs = set(map(lambda const: const.ptr, app.consts()))
    for name in fpu.walk(kernel_ast, f2003.Name):
        if name.string in const_ptrs:
            name.string = f"op2_const_{name.string}_d"

    loop = find(app.loops(), lambda loop: loop.kernel == kernel.name)

    for arg_idx in range(len(loop.args)):
        if not isinstance(loop.args[arg_idx], OP.ArgDat):
            continue

        if loop.args[arg_idx].access_type == OP.AccessType.INC and config["atomics"]:
            continue

        dat = find(app.dats(), lambda dat: dat.ptr == loop.args[arg_idx].dat_ptr)
        if not dat.soa:
            continue

        insertStride(kernel.params[arg_idx][0], dat.ptr, kernel_ast)

    # if kernel.name == "res_calc":
    #     print(repr(kernel_ast))

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
            f2003.Section_Subscript_List(f"((({str(subscript_list)}) - 1) * op2_dat_{dat_ptr}_stride_d + 1)"),
        )


def translateKernel2(config: Dict[str, Any], source: str, kernel: Kernel, app: Application) -> str:
    buffer = SourceBuffer(source)

    # Collect indirect increment identifiers TODO: Tidy
    loop = find(app.loops, lambda l: l.kernel == kernel.name)
    increments = []
    for param, arg in zip(kernel.params, loop.args):
        if arg.indirect and arg.acc == "OP_INC":
            increments.append(param[0])

    # Ast traversal
    subroutine = kernel.ast.find("file/subroutine")
    body = subroutine.find("body")

    # Augment kernel subroutine name
    index = int(subroutine.attrib["line_begin"]) - 1
    buffer.apply(index, lambda line: line.replace(kernel.name, kernel.name + "_gpu"))

    needs_istat = False

    # Atomize incremenal assignments
    if config["atomics"]:
        for assignment in body.findall(".//assignment"):
            # Traverse AST
            name = assignment.find("target/name").attrib["id"]
            operands = assignment.findall("value/operation/operand")
            operator = assignment.find("value/operation/operator/add-op")

            # Determine if the assignment is a valid increment that should be atomised
            atomise = (
                name is not None
                and operator is not None
                and len(operands) == 2
                and name in increments
                and any(o.find("name") and o.find("name").attrib["id"] == name for o in operands)
            )

            if atomise:
                # Extract source locations from AST
                line_index = int(assignment.attrib["line_begin"]) - 1
                assignment_offset = int(assignment.attrib["col_begin"]) - 1
                value_offset = int(assignment.find("value").attrib["col_begin"]) - 1
                operator_offset = int(operator.attrib["col_begin"]) - 1

                # Fold continuations
                line = buffer.get(line_index).rstrip()
                continuations = 0
                while line.endswith("&"):
                    continuations += 1
                    line = line[:-1] + buffer.get(line_index + continuations).lstrip()[1:]

                # Atomize the assignment
                _, value = indexSplit(line, value_offset)
                l, r = indexSplit(value, operator_offset - value_offset)
                line = (assignment_offset + 1) * " " + "istat = AtomicAdd(" + l.strip() + ", " + r.strip() + ")"
                buffer.update(line_index, line)

                # Remove old continuations
                for i in range(1, continuations + 1):
                    buffer.remove(line_index + i)

                needs_istat = True

        # Insert istat typing
        if needs_istat:
            spec = body.find("specification")
            indent = " " * int(spec.attrib["col_begin"])
            buffer.insert(int(spec.attrib["line_end"]), indent + "INTEGER(kind=4) :: istat(4)")

    # Augment OP2 constant references
    source = buffer.translate()
    for const in app.consts:
        source = source.replace(const.ptr, const.ptr + "_OP2")

    return source
