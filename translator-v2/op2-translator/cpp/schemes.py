from pathlib import Path
from typing import Any, Dict, List

import op as OP
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from strategy import indirectRW, setChunkConfig
from target import Target


# Atomics serve only the increments: a loop that reads and writes an indirect
# dat is coloured.
def colourIndirectRW(config: Dict[str, Any], loop: OP.Loop) -> None:
    if indirectRW(loop):
        config["atomics"] = False
        config["color2"] = True


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

    # SIMD lanes sharing a read-write target would lose updates.
    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = super().getConfig(loop, config_overrides)
        if indirectRW(loop):
            config["vectorise"] = config["vectorise"] | {"enable": False}
        return config

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


class CppCudaLegacy(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("cuda_legacy")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/cuda_legacy/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/cuda_legacy/master_kernel.cu.jinja")]

    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = super().getConfig(loop, config_overrides)
        colourIndirectRW(config, loop)
        return config

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


Scheme.register(CppCudaLegacy)

class CppHipLegacy(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("hip_legacy")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/hip_legacy/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/hip_legacy/master_kernel.cpp.jinja")]

    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = super().getConfig(loop, config_overrides)
        colourIndirectRW(config, loop)
        return config

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


Scheme.register(CppHipLegacy)

class CppCuda(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("cuda")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/cuda/loop_host.h.jinja")]
    master_kernel_templates = [Path("cpp/cuda/master_kernel.cu.jinja")]

    staged = frozenset({"chunk_atomics", "chunk_colouring"})

    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = super().getConfig(loop, config_overrides)
        setChunkConfig(config, loop, self.staged)
        return config

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        import cpp.translator.kernels as ctk

        if any(isinstance(arg, OP.ArgDat) and arg.map_idx is not None and arg.map_idx < -1 for arg in loop.args):
            raise OP.OpError(f"{loop.kernel}: vector map arguments are not supported by {self.target.name}", loop.loc)

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}")

        extracted_entities = ctk.extractDependencies(kernel_entities, app)

        def indirect_inc(arg: OP.Arg) -> bool:
            return isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.INC

        # A global OP_INC incremented in place, which no strategy's op2_inc
        # covers: every thread adds to the one copy.
        def atomic_gbl_inc(arg: OP.Arg) -> bool:
            return config["gbl_inc_atomic"] and isinstance(arg, OP.ArgGbl) and arg.access_type == OP.AccessType.INC

        # Every other dat, global and info argument is an op::kernel::Ptr too,
        # whose stride carries its layout and its thread's copy.
        def ptr_arg(arg: OP.Arg) -> bool:
            return (
                isinstance(arg, (OP.ArgDat, OP.ArgGbl, OP.ArgInfo))
                and not indirect_inc(arg)
                and not atomic_gbl_inc(arg)
            )

        for entity, _ in filter(lambda e: e[0] in kernel_entities, extracted_entities):
            ctk.insertIncrements(entity, loop, extracted_entities, indirect_inc)
            ctk.insertIncrements(entity, loop, extracted_entities, atomic_gbl_inc, call="op::kernel::atomic_inc")
            ctk.insertPtrs(entity, loop, extracted_entities, ptr_arg)

        ctk.updateFunctionTypes(extracted_entities, lambda typ, _: f"__device__ {typ}")
        ctk.renameConsts(extracted_entities, app, lambda const, _: f"op2_const_{const}_d")

        return ctk.writeSource(extracted_entities)


Scheme.register(CppCuda)


class CppHip(CppCuda):
    target = Target.get("hip")

    loop_host_templates = [Path("cpp/hip/loop_host.h.jinja")]
    master_kernel_templates = [Path("cpp/hip/master_kernel.cpp.jinja")]


Scheme.register(CppHip)
