import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pcpp
from clang.cindex import Cursor, CursorKind, Index, SourceLocation, SourceRange

import op as OP
from store import Application, Kernel
from util import Location, Rewriter, Span, find


def extentToSpan(extent: SourceRange) -> Span:
    start = Location(extent.start.line, extent.start.column)
    end = Location(extent.end.line, extent.end.column)

    return Span(start, end)


def translateKernel(include_dirs: Set[Path], config: Dict[str, Any], kernel: Kernel, app: Application) -> str:
    args = [f"-I{dir}" for dir in include_dirs]
    translation_unit = Index.create().parse(kernel.path, args=args)

    nodes = translation_unit.cursor.get_children()

    kernel_ast = find(nodes, lambda n: n.kind == CursorKind.FUNCTION_DECL and n.spelling == kernel.name)
    kernel_ast = kernel_ast.get_definition()

    kernel_path = Path(kernel_ast.extent.start.file.name)
    rewriter = Rewriter(kernel_path.read_text())

    loop = find(app.loops(), lambda loop: loop.kernel == kernel.name)

    for arg_idx in range(len(loop.args)):
        if not isinstance(loop.args[arg_idx], OP.ArgDat):
            continue

        dat = find(app.dats(), lambda dat: dat.ptr == loop.args[arg_idx].dat_ptr)
        if not dat.soa:
            continue

        is_vec = loop.args[arg_idx].map_idx < -1
        insertStride(kernel.params[arg_idx][0], dat.ptr, is_vec, kernel_ast, rewriter)

    preprocessor = pcpp.Preprocessor()
    preprocessor.line_directive = None

    for dir in include_dirs:
        preprocessor.add_path(str(dir.resolve()))

    preprocessor.parse(rewriter.rewrite(), str(kernel_path.resolve()))

    source = io.StringIO()
    preprocessor.write(source)

    source.seek(0)
    return source.read()


def insertStride(param: str, dat_ptr: str, is_vec: bool, kernel_ast: Cursor, rewriter: Rewriter) -> None:
    for node in kernel_ast.walk_preorder():
        if node.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
            continue

        ident, subscript = node.get_children()
        if is_vec:
            if next(ident.get_children()).kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
                continue

            ident, _ = next(ident.get_children()).get_children()

        if ident.spelling == param:
            rewriter.update(extentToSpan(subscript.extent), lambda s: f"({s}) * op_dat_{dat_ptr}_stride")