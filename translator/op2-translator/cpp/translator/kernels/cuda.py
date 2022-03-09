import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pcpp
from clang.cindex import Cursor, CursorKind, Index, SourceLocation, SourceRange

import op as OP
from language import Lang
from store import Application, Kernel
from util import Location, Rewriter, Span, find


def extentToSpan(extent: SourceRange) -> Span:
    start = Location(extent.start.line, extent.start.column)
    end = Location(extent.end.line, extent.end.column)

    return Span(start, end)


def translateKernel(
    lang: Lang, include_dirs: Set[Path], config: Dict[str, Any], kernel: Kernel, app: Application
) -> str:
    translation_unit = lang.parseFile(kernel.path, frozenset(include_dirs))
    nodes = translation_unit.cursor.get_children()

    kernel_ast = find(nodes, lambda n: n.kind == CursorKind.FUNCTION_DECL and n.spelling == kernel.name)
    kernel_ast = kernel_ast.get_definition()

    kernel_path = Path(kernel_ast.extent.start.file.name)
    rewriter = Rewriter(kernel_path.read_text())

    function_type_span = extentToSpan(next(kernel_ast.get_tokens()).extent)
    rewriter.update(function_type_span, lambda s: f"__device__ {s}")

    for tok in kernel_ast.get_tokens():
        if tok.spelling == kernel.name:
            rewriter.update(extentToSpan(tok.extent), lambda s: f"{s}_gpu")
            break

    const_ptrs = set(map(lambda const: const.ptr, app.consts()))
    for node in kernel_ast.walk_preorder():
        if node.kind != CursorKind.DECL_REF_EXPR:
            continue

        if node.spelling in const_ptrs:
            rewriter.update(extentToSpan(node.extent), lambda s: f"{s}_d")

    loop = find(app.loops(), lambda loop: loop.kernel == kernel.name)

    for arg_idx in range(len(loop.args)):
        if not isinstance(loop.args[arg_idx], OP.ArgDat):
            continue

        if loop.args[arg_idx].access_type == OP.AccessType.INC and config["atomics"]:
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
            rewriter.update(extentToSpan(subscript.extent), lambda s: f"({s}) * op2_dat_{dat_ptr}_stride_d")
