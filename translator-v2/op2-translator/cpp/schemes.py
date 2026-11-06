from pathlib import Path
from typing import Any, Dict, List, Set

import op as OP
from language import Lang
from scheme import Scheme
from diagnostics import ParseError
from store import Application, Program
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
            raise ParseError(f"unable to find kernel: {loop.kernel}", loop.loc)

        extracted_entities = ctk.extractDependencies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)


Scheme.register(CppSeq)


class CppOpenMPLegacy(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("openmp_legacy")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/openmp_legacy/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/openmp_legacy/master_kernel.cpp.jinja")]

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

        if ctk.hasRuntimeDim(loop):
            raise OP.OpError(f"{loop.kernel}: runtime map indices and dims are not supported by {self.target.name}", loop.loc)

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}", loop.loc)

        extracted_entities = ctk.extractDependencies(kernel_entities, app)
        return ctk.writeSource(extracted_entities)


Scheme.register(CppOpenMPLegacy)


class CppCuda(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("cuda")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/cuda/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("cpp/cuda/master_kernel.cu.jinja")]

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

        if ctk.hasRuntimeDim(loop):
            raise OP.OpError(f"{loop.kernel}: runtime map indices and dims are not supported by {self.target.name}", loop.loc)
        if any(isinstance(arg, OP.ArgInfo) for arg in loop.args):
            raise OP.OpError(f"{loop.kernel}: op_arg_info is not supported by {self.target.name}", loop.loc)

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}", loop.loc)

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

        if ctk.hasRuntimeDim(loop):
            raise OP.OpError(f"{loop.kernel}: runtime map indices and dims are not supported by {self.target.name}", loop.loc)
        if any(isinstance(arg, OP.ArgInfo) for arg in loop.args):
            raise OP.OpError(f"{loop.kernel}: op_arg_info is not supported by {self.target.name}", loop.loc)

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}", loop.loc)

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

        if ctk.hasRuntimeDim(loop):
            raise OP.OpError(f"{loop.kernel}: runtime map indices and dims are not supported by {self.target.name}", loop.loc)

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}", loop.loc)

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

        if ctk.hasRuntimeDim(loop):
            raise OP.OpError(f"{loop.kernel}: runtime map indices and dims are not supported by {self.target.name}", loop.loc)

        kernel_entities = app.findEntities(loop.kernel, program)
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel: {loop.kernel}", loop.loc)

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


class CppOpenMP(Scheme):
    lang = Lang.get("cpp")
    target = Target.get("openmp")

    fallback = None

    consts_template = None
    loop_host_templates = [Path("cpp/openmp/loop_host.h.jinja")]
    master_kernel_templates = [Path("cpp/openmp/master_kernel.cpp.jinja")]

    # chunk_colouring runs the plain body over its chunks.
    staged = frozenset({"chunk_atomics"})

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
            raise ParseError(f"unable to find kernel: {loop.kernel}", loop.loc)

        extracted_entities = ctk.extractDependencies(kernel_entities, app)

        # Every other dat, global and info argument is an op::rt::kernel::Ptr too.
        def ptr_arg(arg: OP.Arg) -> bool:
            return isinstance(arg, (OP.ArgDat, OP.ArgGbl, OP.ArgInfo)) and not OP.indirectInc(arg)

        # An indirect increment through the kernel's hook (OP.Loop.incHook).
        hooked: Set[str] = set()
        for entity, _ in filter(lambda e: e[0] in kernel_entities, extracted_entities):
            hooked |= ctk.insertIncrements(
                entity, loop, extracted_entities, OP.indirectInc, call="op2_inc::add", hook="op2_inc"
            )
            ctk.insertPtrs(entity, loop, extracted_entities, ptr_arg)

        # Kernels inlined into each body that calls them, as openmp_f2c's are.
        ctk.updateFunctionTypes(
            extracted_entities,
            lambda typ, func: f"{'template<typename op2_inc> ' if func.key in hooked else ''}"
            f"__attribute__((always_inline)){'' if ctk.declaredInline(func) else ' inline'} {typ}",
        )

        return ctk.writeSource(extracted_entities)


Scheme.register(CppOpenMP)
