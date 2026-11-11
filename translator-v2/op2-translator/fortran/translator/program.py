import re

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

from diagnostics import OpError
from store import Program


def translateProgram2(program: Program, force_soa: bool) -> str:
    flags = re.MULTILINE | re.IGNORECASE

    # A call anywhere on a line outside a comment, if (...) call op_par_loop_2(...)
    # too, its kernel name perhaps on a continuation line.
    loop_call = re.compile(r"^([^!\n]*?)\bcall\s*op_par_loop_\d+\s*\(\s*(?:&\s*)?\w+\s*(?:&\s*)?,\s*", flags)

    calls = len(loop_call.findall(program.source))
    if calls != len(program.loops):
        raise OpError(f"{program.path}: found {calls} op_par_loop calls to rewrite, but parsed {len(program.loops)}")

    loops = iter(program.loops)

    def repl(m: re.Match) -> str:
        loop = next(loops)
        return f'{m.group(1)}call op2_k_{loop.name}("{loop.kernel}", '

    src = loop_call.sub(repl, program.source)
    src = re.sub(r"^(\s*)(use op2_fortran_reference)", r"\1\2\n\1use op2_kernels", src, flags=flags)

    if force_soa:
        src = re.sub(
            r"\bcall\s+(op_init|op_init_base|op_mpi_init)\s*\(([^()]*)\)", r"call \1_soa(\2, 1)", src, flags=flags
        )

    return src


def translateProgram(program: Program, force_soa: bool) -> str:
    ast = program.ast
    loops = iter(program.loops)

    for call in fpu.walk(ast, f2003.Call_Stmt):
        name = fpu.get_child(call, f2003.Name)
        if name is None or name.string.lower() != "op_decl_const":
            continue

        args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
        args.items = tuple(list(args.items[:-1]))

        const_ptr = args.items[0].string.lower()

        name.string = f"{name.string}_{const_ptr}"

    for call in fpu.walk(ast, f2003.Call_Stmt):
        name = fpu.get_child(call, f2003.Name)

        if name is None:
            continue

        if not re.match(r"op_par_loop_\d+", name.string.lower()):
            continue

        args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
        arg_list = list(args.items)

        loop = next(loops)

        arg_list[0] = f2003.Char_Literal_Constant(f'"{loop.kernel}"')
        args.items = tuple(arg_list)

        name.string = f"op2_k_{loop.name}"

    for main_program in fpu.walk(ast, f2003.Main_Program):
        spec = fpu.get_child(main_program, f2003.Specification_Part)
        new_content = [f2003.Use_Stmt("use op2_kernels")]

        for node in spec.content:
            if (
                isinstance(node, f2003.Use_Stmt)
                and fpu.get_child(node, f2003.Name).string.lower() == "op2_fortran_reference"
            ):
                continue

            new_content.append(node)

        spec.content = new_content

    if force_soa:
        for call in fpu.walk(ast, f2003.Call_Stmt):
            name = fpu.get_child(call, f2003.Name)
            if name is None:
                continue

            init_funcs = ["op_init", "op_init_base", "op_mpi_init"]
            if name.string.lower() not in init_funcs:
                continue

            args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
            args.items = tuple(list(args.items) + [f2003.Int_Literal_Constant("1")])

            name.string = f"{name.string}_soa"

    return unindent_cpp_directives(str(ast))


def unindent_cpp_directives(s: str) -> str:
    directives = [
        "if",
        "ifdef",
        "ifndef",
        "elif",
        "else",
        "endif",
        "include",
        "define",
        "undef",
        "line",
        "error",
        "warning",
    ]

    return re.sub(rf"^\s*#({'|'.join(directives)})(\s+|\s*$)", r"#\1\2", s, flags=re.MULTILINE)
