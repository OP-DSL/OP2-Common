from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

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
    translation_unit, source = lang.parseFile(kernel.path, frozenset(include_dirs), frozenset(defines), preprocess=True)
    nodes = translation_unit.cursor.get_children()

    kernel_ast = find(nodes, lambda n: n.kind == CursorKind.FUNCTION_DECL and n.spelling == kernel.name)
    kernel_ast = kernel_ast.get_definition()

    kernel_path = Path(kernel_ast.extent.start.file.name)
    rewriter = Rewriter(source, [extentToSpan(kernel_ast.extent)])

    return (kernel_ast, kernel_path, rewriter)


def renameType(def_ast: Cursor, asts: List[Cursor], rewriter: Rewriter, name: str, replacement: str) -> None:
    renameTypeDefinition(def_ast, rewriter, replacement)

    for ast in asts:
        renameTypeRefs(ast, rewriter, name, replacement)


def renameFunction(def_ast: Cursor, asts: List[Cursor], rewriter: Rewriter, name: str, replacement: str) -> None:
    renameFunctionDefinition(def_ast, rewriter, replacement)

    for ast in asts:
        renameFunctionCalls(ast, rewriter, name, replacement)


def extractTypes(kernel_ast: Cursor, rewriter: Rewriter, exclude: List[str] = []) -> List[Tuple[str, Cursor]]:
    seen = []
    defs = []

    for node in kernel_ast.walk_preorder():
        if node.kind != CursorKind.TYPE_REF:
            continue

        definition = node.get_definition()

        if definition is not None and definition.spelling not in seen and definition.spelling not in exclude:
            seen.append(definition.spelling)
            defs.append(definition)

            span = extentToSpan(definition.extent)
            rewriter.extend(span)
            rewriter.update(Span(span.end, span.end), lambda s: ";\n\n")

    sub_types = []
    for type_def in defs:
        sub_types.extend(extractTypes(type_def, rewriter, exclude + seen))

    return sub_types + list(zip(seen, defs))


def renameTypeDefinition(ast: Cursor, rewriter: Rewriter, replacement: str) -> None:
    for tok in ast.get_tokens():
        if tok.spelling == ast.spelling:
            rewriter.update(extentToSpan(tok.extent), lambda _: replacement)
            break


def renameTypeRefs(ast: Cursor, rewriter: Rewriter, name: str, replacement: str) -> None:
    seen = []

    for node in ast.walk_preorder():
        if node.kind != CursorKind.TYPE_REF:
            continue

        if node.spelling != name and node.spelling != f"struct {name}" and node.spelling != f"class {name}":
            continue

        if extentToSpan(node.extent) in seen:
            continue

        seen.append(extentToSpan(node.extent))
        rewriter.update(extentToSpan(node.extent), lambda _: replacement)


def extractFunctions(ast: Cursor, rewriter: Rewriter, exclude: List[str] = []) -> List[Tuple[str, Cursor]]:
    seen = []
    defs = []

    for node in ast.walk_preorder():
        if node.kind != CursorKind.CALL_EXPR:
            continue

        # TODO: we seem to get type refs here without this check?
        if next(node.get_children(), None) is None:
            continue

        definition = node.get_definition()

        if definition is None or str(definition.extent.start.file) != str(ast.extent.start.file):
            continue

        if definition.spelling not in seen and definition.spelling not in exclude:
            seen.append(definition.spelling)
            defs.append(definition)

            span = extentToSpan(definition.extent)
            rewriter.extend(span)
            rewriter.update(Span(span.end, span.end), lambda s: "\n\n")

    sub_calls = []
    for func_def in defs:
        sub_calls.extend(extractFunctions(func_def, rewriter, exclude + seen))

    return sub_calls + list(zip(seen, defs))


def renameFunctionDefinition(function_ast: Cursor, rewriter: Rewriter, replacement: str) -> None:
    for tok in function_ast.get_tokens():
        if tok.spelling == function_ast.spelling:
            rewriter.update(extentToSpan(tok.extent), lambda _: replacement)
            break


def renameFunctionCalls(ast: Cursor, rewriter: Rewriter, name: str, replacement: str) -> None:
    for node in ast.walk_preorder():
        if node.kind != CursorKind.CALL_EXPR:
            continue

        if node.spelling != name:
            continue

        children = list(node.get_children())
        if len(children) < 1:
            continue

        rewriter.update(extentToSpan(children[0].extent), lambda _: replacement)


def updateFunctionType(kernel_ast: Cursor, rewriter: Rewriter, replacement: Callable[[str], str]) -> None:
    function_type_span = extentToSpan(next(kernel_ast.get_tokens()).extent)
    rewriter.update(function_type_span, replacement)


def renameKernel(kernel_ast: Cursor, rewriter: Rewriter, kernel: Kernel, replacement: Callable[[str], str]) -> None:
    for tok in kernel_ast.get_tokens():
        if tok.spelling == kernel.name:
            rewriter.update(extentToSpan(tok.extent), replacement)
            break


def renameConsts(ast: Cursor, rewriter: Rewriter, app: Application, replacement: Callable[[str], str]) -> None:
    const_ptrs = set(map(lambda const: const.ptr, app.consts()))

    for node in ast.walk_preorder():
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
