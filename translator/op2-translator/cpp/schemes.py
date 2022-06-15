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
        return ctk.preprocess(rewriter.rewrite(), kernel_path, include_dirs, defines)


Scheme.register(CppSeq)


class CppOpenMP(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("openmp")

    loop_host_template = Path("cpp/openmp/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/openmp/master_kernel.cpp.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast, kernel_path, rewriter = ctk.findKernel(self.lang, kernel, include_dirs, defines)
        return ctk.preprocess(rewriter.rewrite(), kernel_path, include_dirs, defines)


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

        ctk.renameConsts(kernel_ast, rewriter, app, lambda const: f"{const}_d")
        ctk.insertStrides(
            kernel_ast,
            rewriter,
            app,
            kernel,
            lambda dat_ptr: f"op2_dat_{dat_ptr}_stride_d",
            skip=lambda arg: arg.access_type == OP.AccessType.INC and self.target.config["atomics"],
        )

        return ctk.preprocess(rewriter.rewrite(), kernel_path, include_dirs, defines)


# Scheme.register(CppCuda)
