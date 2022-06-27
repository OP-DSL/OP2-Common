from pathlib import Path
from typing import List, Set

import cpp.translator.kernels as ctk
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, Kernel
from target import Target


class CppSeq(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("seq")

    loop_host_template = Path("cpp/seq/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/seq/master_kernel.cpp.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast, kernel_path, rewriter = ctk.findKernel(self.lang, kernel, include_dirs, defines)

        funcs = ctk.extractFunctions(kernel_ast, rewriter)
        types = ctk.extractTypes(kernel_ast, rewriter)

        asts = [kernel_ast] + [func_def for _, func_def in funcs] + [type_def for _, type_def in types]

        for name, func_def in funcs:
            new_name = f"op2_func_{kernel.name}_{name}"
            ctk.renameFunction(func_def, asts, rewriter, name, new_name)

        for name, type_def in types:
            new_name = f"op2_type_{kernel.name}_{name}"
            ctk.renameType(type_def, asts, rewriter, name, new_name)

        return rewriter.rewrite()


Scheme.register(CppSeq)


class CppOpenMP(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("openmp")

    loop_host_template = Path("cpp/openmp/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/openmp/master_kernel.cpp.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast, kernel_path, rewriter = ctk.findKernel(self.lang, kernel, include_dirs, defines)

        funcs = ctk.extractFunctions(kernel_ast, rewriter)
        types = ctk.extractTypes(kernel_ast, rewriter)

        asts = [kernel_ast] + [func_def for name, func_def in funcs] + [type_def for name, type_def in types]

        for name, func_def in funcs:
            new_name = f"op2_func_{kernel.name}_{name}"
            ctk.renameFunction(func_def, asts, rewriter, name, new_name)

        for name, type_def in types:
            new_name = f"op2_type_{kernel.name}_{name}"
            ctk.renameType(type_def, asts, rewriter, name, new_name)

        return rewriter.rewrite()


Scheme.register(CppOpenMP)


class CppCuda(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("cuda")

    loop_host_template = Path("cpp/cuda/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/cuda/master_kernel.cu.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast, kernel_path, rewriter = ctk.findKernel(self.lang, kernel, include_dirs, defines)

        ctk.updateFunctionType(kernel_ast, rewriter, lambda typ: f"__device__ {typ}")
        ctk.renameKernel(kernel_ast, rewriter, kernel, lambda name: f"{name}_gpu")

        funcs = ctk.extractFunctions(kernel_ast, rewriter)
        types = ctk.extractTypes(kernel_ast, rewriter)

        asts = [kernel_ast] + [func_def for name, func_def in funcs] + [type_def for name, type_def in types]

        for name, func_def in funcs:
            new_name = f"op2_func_{kernel.name}_{name}"
            ctk.renameFunction(func_def, asts, rewriter, name, new_name)
            ctk.updateFunctionType(func_def, rewriter, lambda typ: f"__device__ {typ}")

        for name, type_def in types:
            new_name = f"op2_type_{kernel.name}_{name}"
            ctk.renameType(type_def, asts, rewriter, name, new_name)

        for ast in asts:
            ctk.renameConsts(ast, rewriter, app, lambda const: f"{const}_d")

            ctk.insertStrides(
                ast,
                rewriter,
                app,
                kernel,
                lambda dat_id: f"op2_{kernel.name}_dat{dat_id}_stride_d",
                skip=lambda arg: arg.access_type == OP.AccessType.INC and self.target.config["atomics"],
            )

        return rewriter.rewrite()


Scheme.register(CppCuda)
