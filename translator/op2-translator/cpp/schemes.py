from pathlib import Path

import cpp.translator.kernels as ctk
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, Kernel, Program, ParseError
from target import Target


class CppSeq(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("seq")

    loop_host_template = Path("cpp/seq/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/seq/master_kernel.cpp.jinja")

    def translateKernel(
        self,
        kernel: Kernel,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(kernel.name, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {kernel.name}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)


Scheme.register(CppSeq)


class CppOpenMP(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("openmp")

    loop_host_template = Path("cpp/openmp/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/openmp/master_kernel.cpp.jinja")

    def translateKernel(
        self,
        kernel: Kernel,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(kernel.name, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {kernel.name}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)


Scheme.register(CppOpenMP)


class CppCuda(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("cuda")

    loop_host_template = Path("cpp/cuda/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/cuda/master_kernel.cu.jinja")

    def translateKernel(
        self,
        kernel: Kernel,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(kernel.name, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {kernel.name}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)

        ctk.updateFunctionTypes(extracted_entities, lambda typ, _: f"__device__ {typ}")
        ctk.renameConsts(extracted_entities, app, lambda const, _: f"{const}_d")

        for entity, rewriter in filter(lambda e: e[0] in kernel_entities, extracted_entities):
            ctk.insertStrides(
                entity,
                rewriter,
                app,
                kernel,
                lambda dat_id: f"op2_{kernel.name}_dat{dat_id}_stride_d",
                skip=lambda arg: arg.access_type == OP.AccessType.INC and self.target.config["atomics"],
            )

        return ctk.writeSource(extracted_entities)


Scheme.register(CppCuda)
