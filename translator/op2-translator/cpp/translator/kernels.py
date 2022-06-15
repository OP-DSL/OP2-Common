from io import StringIO
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

import pcpp
from clang.cindex import Cursor, CursorKind, SourceRange

import op as OP
from language import Lang
from store import Application, Kernel
from util import Location, Rewriter, Span, find


def extentToSpan(extent: SourceRange) -> Span:
    start = Location(extent.start.line, extent.start.column)
    end = Location(extent.end.line, extent.end.column)

    return Span(start, end)


def findKernel(lang: Lang, kernel: Kernel, include_dirs: Set[Path], defines: List[str]) -> Tuple[Cursor, Rewriter]:
    translation_unit = lang.parseFile(kernel.path, frozenset(include_dirs), frozenset(defines))
    nodes = translation_unit.cursor.get_children()

    kernel_ast = find(nodes, lambda n: n.kind == CursorKind.FUNCTION_DECL and n.spelling == kernel.name)
    kernel_ast = kernel_ast.get_definition()

    kernel_path = Path(kernel_ast.extent.start.file.name)
    rewriter = Rewriter(kernel_path.read_text())

    return (kernel_ast, kernel_path, rewriter)


def updateFunctionType(kernel_ast: Cursor, rewriter: Rewriter, replacement: Callable[[str], str]) -> None:
    function_type_span = extentToSpan(next(kernel_ast.get_tokens()).extent)
    rewriter.update(function_type_span, replacement)


def renameKernel(kernel_ast: Cursor, rewriter: Rewriter, kernel: Kernel, replacement: Callable[[str], str]) -> None:
    for tok in kernel_ast.get_tokens():
        if tok.spelling == kernel.name:
            rewriter.update(extentToSpan(tok.extent), replacement)
            break


def renameConsts(kernel_ast: Cursor, rewriter: Rewriter, app: Application, replacement: Callable[[str], str]) -> None:
    const_ptrs = set(map(lambda const: const.ptr, app.consts()))

    for node in kernel_ast.walk_preorder():
        if node.kind != CursorKind.DECL_REF_EXPR:
            continue

        if node.spelling in const_ptrs:
            rewriter.update(extentToSpan(node.extent), replacement)


def insertStrides(
    kernel_ast: Cursor,
    rewriter: Rewriter,
    app: Application,
    kernel: Kernel,
    stride: Callable[[int], str],
    skip: Optional[Callable[[OP.ArgDat], bool]] = None,
) -> None:
    loop = find(app.loops(), lambda loop: loop.kernel == kernel.name)

    for arg_idx in range(len(loop.args)):
        if not isinstance(loop.args[arg_idx], OP.ArgDat):
            continue

        if skip is not None and skip(loop.args[arg_idx]):
            continue

        dat = loop.dats[loop.args[arg_idx].dat_id]
        if not dat.soa:
            continue

        is_vec = loop.args[arg_idx].map_idx < -1
        insertStride(kernel_ast, rewriter, kernel.params[arg_idx][0], dat.id, is_vec, stride)


def insertStride(
    kernel_ast: Cursor, rewriter: Rewriter, param: str, dat_id: int, is_vec: bool, stride: Callable[[int], str]
) -> None:
    for node in kernel_ast.walk_preorder():
        if node.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
            continue

        ident, subscript = node.get_children()
        if is_vec:
            if next(ident.get_children()).kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
                continue

            ident, _ = next(ident.get_children()).get_children()

        if ident.spelling == param:
            rewriter.update(extentToSpan(subscript.extent), lambda s: f"({s}) * {stride(dat_id)}")


def preprocess(source: str, path: Path, include_dirs: Set[Path], defines: List[str]) -> str:
    preprocessor = pcpp.Preprocessor()
    preprocessor.line_directive = None

    for dir in include_dirs:
        preprocessor.add_path(str(dir.resolve()))

    preprocessor.parse(source, str(path.resolve()))

    source = StringIO()
    preprocessor.write(source)

    source.seek(0)
    return source.read()
