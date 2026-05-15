from pathlib import Path
from typing import Any, Dict

import op as OP
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target


class CppSeq(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("seq")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/seq/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/seq/master_kernel.cpp.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        import cpp.translator.kernels as ctk

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)


Scheme.register(CppSeq)


class CppOpenMP(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("openmp")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/openmp/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/openmp/master_kernel.cpp.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        import cpp.translator.kernels as ctk

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)


Scheme.register(CppOpenMP)


class CppCuda(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("cuda")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/cuda/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/cuda/master_kernel.cu.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        import cpp.translator.kernels as ctk

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)

        ctk.updateFunctionTypes(extracted_entities, lambda typ, _: f"__device__ {typ}")
        ctk.renameConsts(extracted_entities, app, lambda const, _: f"{const}_d")

        def indirect(loop: OP.Loop) -> bool:
            return len(loop.maps) > 0

        for entity, rewriter in filter(lambda e: e[0] in kernel_entities, extracted_entities):
            ctk.insertStrides(
                entity,
                rewriter,
                app,
                loop,
                lambda dat_id: f"op2_{loop.kernel}_dat{dat_id}_stride_d",
                skip=lambda arg: arg.access_type == OP.AccessType.INC and config["atomics"] and indirect(loop),
                entities=extracted_entities,
            )

        return ctk.writeSource(extracted_entities)


Scheme.register(CppCuda)

class CppHip(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("hip")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/hip/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/hip/master_kernel.cpp.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        import cpp.translator.kernels as ctk

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)

        ctk.updateFunctionTypes(extracted_entities, lambda typ, _: f"__device__ {typ}")
        ctk.renameConsts(extracted_entities, app, lambda const, _: f"{const}_d")

        def indirect(loop: OP.Loop) -> bool:
            return len(loop.maps) > 0

        for entity, rewriter in filter(lambda e: e[0] in kernel_entities, extracted_entities):
            ctk.insertStrides(
                entity,
                rewriter,
                app,
                loop,
                lambda dat_id: f"op2_{loop.kernel}_dat{dat_id}_stride_d",
                skip=lambda arg: arg.access_type == OP.AccessType.INC and config["atomics"] and indirect(loop),
                entities=extracted_entities,
            )

        return ctk.writeSource(extracted_entities)


Scheme.register(CppHip)

class CppJitCuda(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("c_cuda")

    fallback = Scheme.get((Lang.get("cpp"), Target.get("seq")))

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/jit_cuda/loop_host.h.jinja")]
    master_kernel_templates = [Path("cpp/jit_cuda/master_kernel.cu.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        import cpp.translator.kernels as ctk

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)

        ctk.updateFunctionTypes(extracted_entities, lambda typ, _: f"__device__ {typ}")
        ctk.renameConsts(extracted_entities, app, lambda const, _: f"op2_const_{const}_d")

        def indirect(loop: OP.Loop) -> bool:
            return len(loop.maps) > 0

        for entity, rewriter in filter(lambda e: e[0] in kernel_entities, extracted_entities):
            ctk.insertStrides(
                entity,
                rewriter,
                app,
                loop,
                lambda dat_id: f"op2_{loop.kernel}_dat{dat_id}_stride_d",
                skip=lambda arg: arg.access_type == OP.AccessType.INC and config["atomics"] and indirect(loop),
                entities=extracted_entities,
            )

            ctk.insertArgGblStrides(
                entity,
                rewriter,
                app,
                loop,
                lambda dat_id: f"op2_{loop.kernel}_gbl_stride_d",
                skip=lambda arg: arg.access_type not in [OP.AccessType.INC, OP.AccessType.MAX, OP.AccessType.MIN],
            )

        return ctk.writeSource(extracted_entities)


Scheme.register(CppJitCuda)


class CppJitHip(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("c_hip")

    fallback = Scheme.get((Lang.get("cpp"), Target.get("seq")))

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/jit_hip/loop_host.h.jinja")]
    master_kernel_templates = [Path("cpp/jit_hip/master_kernel.cpp.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        import cpp.translator.kernels as ctk

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)

        ctk.updateFunctionTypes(extracted_entities, lambda typ, _: f"__device__ {typ}")
        ctk.renameConsts(extracted_entities, app, lambda const, _: f"op2_const_{const}_d")

        def indirect(loop: OP.Loop) -> bool:
            return len(loop.maps) > 0

        for entity, rewriter in filter(lambda e: e[0] in kernel_entities, extracted_entities):
            ctk.insertStrides(
                entity,
                rewriter,
                app,
                loop,
                lambda dat_id: f"op2_{loop.kernel}_dat{dat_id}_stride_d",
                skip=lambda arg: arg.access_type == OP.AccessType.INC and config["atomics"] and indirect(loop),
                entities=extracted_entities,
            )

            ctk.insertArgGblStrides(
                entity,
                rewriter,
                app,
                loop,
                lambda dat_id: f"op2_{loop.kernel}_gbl_stride_d",
                skip=lambda arg: arg.access_type not in [OP.AccessType.INC, OP.AccessType.MAX, OP.AccessType.MIN],
            )

        return ctk.writeSource(extracted_entities)


Scheme.register(CppJitHip)