import re
import io
from typing import Any, Dict, Tuple, Set
from pathlib import Path

import pcpp

import op as OP
from store import Application, Kernel
from util import find


def translateKernel(
    include_dirs: Set[Path], config: Dict[str, Any], kernel: Kernel, app: Application
) -> str:
    preprocessor = pcpp.Preprocessor()

    preprocessor.line_directive = None
    # preprocessor.compress = 2

    for dir in include_dirs:
        preprocessor.add_path(str(dir.resolve()))

    preprocessor.parse(kernel.path.read_text(), str(kernel.path.resolve()))

    source = io.StringIO()
    preprocessor.write(source)

    source.seek(0)
    source = source.read()

    source = re.sub(fr"void\b\s+\b{kernel.name}\b", f"__device__ void {kernel.name}_gpu", source)

    for const in app.consts():
        source = re.sub(fr"\b{const.ptr}\b", f"{const.ptr}_d", source)

    loop = find(app.loops(), lambda loop: loop.kernel == kernel.name)

    for arg_idx in range(len(loop.args)):
        if not isinstance(loop.args[arg_idx], OP.ArgDat):
            continue

        dat = find(app.dats(), lambda dat: dat.ptr == loop.args[arg_idx].dat_ptr)
        if not dat.soa:
            continue

        param = kernel.params[arg_idx][0]

        if loop.args[arg_idx].map_idx < 0:
            source = re.sub(
                fr"\b{param}(\[[^\]]\])\[([\s+\+\*A-Za-z0-9_]*)\]",
                fr"{param}\1[(\2) * op2_dat_{dat.ptr}_stride_d]",
                source,
            )
        else:
            source = re.sub(fr"\*\b{param}\b\s*(?!\[)", f"{param}[0]", source)
            source = re.sub(
                fr"\b{param}\[([\s\+\*A-Za-z0-9_]*)\]", fr"{param}[(\1) * op2_dat_{dat.ptr}_stride_d]", source
            )

    return source
