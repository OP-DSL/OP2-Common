import re

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

from store import Program

KERNEL_ID = 1

def translateProgram(ast: f2003.Program, program: Program, force_soa: bool) -> str:
    src = program.path.read_text()

    def repl(m):
        global KERNEL_ID
        r = f"{m.group(1)}call op2_k{KERNEL_ID}_{m.group(2)}(\"{m.group(2)}\", "
        KERNEL_ID = KERNEL_ID + 1

        return r

    src = re.sub(
        r"^(\s*)call\s*op_par_loop_\d+\s*\(\s*(\w+)\s*,\s*",
        repl,
        src,
        flags=re.MULTILINE | re.IGNORECASE
    )

    src = re.sub(
        r"^(\s*)(use op2_fortran_reference)",
        r"\1\2\n\1use op2_kernels",
        src,
        flags=re.MULTILINE | re.IGNORECASE
    )

    return src


def translateProgram2(ast: f2003.Program, program: Program, force_soa: bool) -> str:
    for call in fpu.walk(ast, f2003.Call_Stmt):
        name = fpu.get_child(call, f2003.Name)
        if name is None or name.string != "op_decl_const":
            continue

        args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
        args.items = tuple(list(args.items[:-1]))

        const_ptr = args.items[0].string

        name.string = f"{name.string}_{const_ptr}"

    for call in fpu.walk(ast, f2003.Call_Stmt):
        name = fpu.get_child(call, f2003.Name)

        if name is None:
            continue

        if not re.match(r"op_par_loop_\d+", name.string):
            continue

        args = fpu.get_child(call, f2003.Actual_Arg_Spec_List)
        arg_list = list(args.items)

        kernel_name = arg_list[0].string

        arg_list[0] = f2003.Char_Literal_Constant(f'"{kernel_name}"')
        args.items = tuple(arg_list)

        # TODO: make this more robust
        global KERNEL_ID
        name.string = f"op2_k{KERNEL_ID}_{kernel_name}"

        KERNEL_ID = KERNEL_ID + 1

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
            if name.string not in init_funcs:
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
