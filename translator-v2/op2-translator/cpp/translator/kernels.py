from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from clang.cindex import Cursor, CursorKind, SourceRange, Token, TypeKind

import op as OP
from store import Application, Entity, Function, Type
from util import Location, Rewriter, Span, find, safeFind


def extentToSpan(extent: SourceRange) -> Span:
    start = Location(extent.start.line, extent.start.column)
    end = Location(extent.end.line, extent.end.column)

    return Span(start, end)


def extractDependencies(entities: List[Entity], app: Application) -> List[Tuple[Entity, Rewriter]]:
    unprocessed_entities = list(entities)
    extracted_entities = []

    while len(unprocessed_entities) > 0:
        entity = unprocessed_entities.pop(0)

        if safeFind(extracted_entities, lambda e: e[0] == entity):
            continue

        for dependency in entity.depends:
            dependency_entities = app.findEntities(dependency, entity.program)
            unprocessed_entities.extend(dependency_entities)

        rewriter = Rewriter(entity.program.source, [extentToSpan(entity.ast.extent)])
        extracted_entities.insert(0, (entity, rewriter))

    return extracted_entities


def updateFunctionTypes(entities: List[Tuple[Entity, Rewriter]], replacement: Callable[[str, Entity], str]) -> None:
    for entity, rewriter in filter(lambda e: isinstance(e[0], Function), entities):
        function_type_span = extentToSpan(next(entity.ast.get_tokens()).extent)
        rewriter.update(function_type_span, lambda s, entity=entity: replacement(s, entity))


def renameConsts(
    entities: List[Tuple[Entity, Rewriter]], app: Application, replacement: Callable[[str, Entity], str]
) -> None:
    const_ptrs = app.constPtrs()

    for entity, rewriter in entities:
        for node in entity.ast.walk_preorder():
            if node.kind != CursorKind.DECL_REF_EXPR:
                continue

            if node.spelling in const_ptrs:
                rewriter.update(extentToSpan(node.extent), lambda s, entity=entity: replacement(s, entity))


def mapParam(
    func: Function,
    param_idx: int,
    funcs: List[Function],
    op: Callable[[Function, int], Optional[bool]],
    *args,
) -> None:
    called_list = findCalled(func, param_idx, funcs)

    for func2, param_idx2 in called_list:
        stop = op(func2, param_idx2, *args)
        if stop:
            break


def findCalled(func: Function, param_idx: int, funcs: List[Function]) -> List[Tuple[Function, int]]:
    checked: Dict[str, Set[int]] = {}
    stack = [(func, param_idx)]

    while len(stack) > 0:
        candidate_func, candidate_param_idx = stack.pop()

        if candidate_func.name not in checked:
            checked[candidate_func.name] = set()
        elif candidate_param_idx in checked[candidate_func.name]:
            continue

        checked[candidate_func.name].add(candidate_param_idx)

        for called_func, called_param_idx in findCalled2(candidate_func, candidate_param_idx, funcs):
            if called_func.name in checked and called_param_idx in checked[called_func.name]:
                continue

            stack.append((called_func, called_param_idx))

    called_list = []
    for func_name, indicies in checked.items():
        for index in indicies:
            called_list.append((find(funcs, lambda f: f.name == func_name), index))

    return called_list


def findCalled2(func: Function, param_idx: int, funcs: List[Function]) -> List[Tuple[Function, int]]:
    called: List[Tuple[Function, int]] = []

    if param_idx >= len(func.parameters):
        return called

    param = func.parameters[param_idx]

    for node in func.ast.walk_preorder():
        if node.kind != CursorKind.CALL_EXPR:
            continue

        callee = node.get_definition() or node.referenced
        if callee is None:
            continue

        callee_func = safeFind(funcs, lambda f: f.name == callee.spelling)
        if callee_func is None:
            continue

        for arg_idx, arg in enumerate(node.get_arguments()):
            if isDirectParamRef(arg, param):
                called.append((callee_func, arg_idx))

    return called


def isDirectParamRef(node: Cursor, param: str) -> bool:
    node = stripImplicit(node)
    return node.kind == CursorKind.DECL_REF_EXPR and node.spelling == param


def stripImplicit(node: Cursor) -> Cursor:
    implicit_kinds = []
    for kind_name in [
        "UNEXPOSED_EXPR",
        "IMPLICIT_CAST_EXPR",
        "PAREN_EXPR",
        "CSTYLE_CAST_EXPR",
        "CXX_STATIC_CAST_EXPR",
        "CXX_REINTERPRET_CAST_EXPR",
        "CXX_CONST_CAST_EXPR",
        "CXX_FUNCTIONAL_CAST_EXPR",
    ]:
        kind = getattr(CursorKind, kind_name, None)
        if kind is not None:
            implicit_kinds.append(kind)

    while node.kind in implicit_kinds:
        children = list(node.get_children())
        if len(children) != 1:
            break

        node = children[0]

    return node


def insertStrides(
    entity: Entity,
    rewriter: Rewriter,
    app: Application,
    loop: OP.Loop,
    stride: Callable[[int], str],
    skip: Optional[Callable[[OP.ArgDat], bool]] = None,
    entities: Optional[List[Tuple[Entity, Rewriter]]] = None,
) -> None:
    if not isinstance(entity, Function):
        return

    if entities is None:
        entities = [(entity, rewriter)]

    funcs = [e for e, _ in entities if isinstance(e, Function)]
    rewriters = {e.name: r for e, r in entities if isinstance(e, Function)}

    for arg_idx in range(len(loop.args)):
        arg = loop.args[arg_idx]
        if not isinstance(arg, OP.ArgDat):
            continue

        if skip is not None and skip(arg):
            continue

        dat = loop.dats[arg.dat_id]
        if not dat.soa:
            continue

        is_vec = arg.map_idx is not None and arg.map_idx < -1

        def applyStride(func: Function, param_idx: int, dat_id: int, is_vec: bool) -> None:
            if param_idx >= len(func.parameters):
                return

            func_rewriter = rewriters.get(func.name)
            if func_rewriter is None:
                return

            insertStride(func.ast, func_rewriter, func.parameters[param_idx], dat_id, is_vec, stride)

        mapParam(entity, arg_idx, funcs, applyStride, dat.id, is_vec)

# Route each increment of a matched argument through call(target, value),
# by default op2_inc, which each strategy's generated code defines, and pass
# the argument as an op::kernel::Ptr, which carries its stride.  An increment is p[i] += e, p[i] -= e,
# ++p[i], p[i]--, p[i] = p[i] + e - f (a chain of + and - led by p[i]) or
# p[i] = e + p[i] (*p for p[0]) as a statement of its own; the argument may
# also be passed to a helper function under the same rules.  Any other use of
# it is an error.
def insertIncrements(
    entity: Entity,
    loop: OP.Loop,
    entities: List[Tuple[Entity, Rewriter]],
    match: Callable[[OP.Arg], bool],
    call: str = "op2_inc",
) -> None:
    funcs = [e for e, _ in entities if isinstance(e, Function)]
    rewriters = {e.name: r for e, r in entities if isinstance(e, Function)}

    targets = argumentTargets(entity, loop, funcs, match)
    for func in funcs:
        for param_idx in sorted(targets.get(func.name, set())):
            rewriteIncrements(func, rewriters[func.name], param_idx, targets, loop, call)

    # A helper that takes an increment target takes nothing else there.
    for func in funcs:
        params = parameters(func)
        for node in func.ast.walk_preorder():
            callee = node.referenced if node.kind == CursorKind.CALL_EXPR else None
            if callee is None or callee.spelling not in targets:
                continue

            for idx, arg in enumerate(node.get_arguments()):
                arg = stripImplicit(arg)
                if idx in targets[callee.spelling] and not (
                    arg.kind == CursorKind.DECL_REF_EXPR
                    and arg.referenced in params
                    and params.index(arg.referenced) in targets.get(func.name, set())
                ):
                    raise OP.OpError(
                        f"{func.name}: {callee.spelling} takes an OP_INC argument as its parameter {idx}, "
                        f"so it may take nothing else there: {tokenText(node)}",
                        loop.loc,
                    )


# Pass each matched argument as an op::kernel::Ptr, which carries its stride,
# through every helper it reaches.  A use is p[i], *p, or a pass to a helper
# whose parameter is a Ptr there, which may take a raw pointer (a local
# array, at stride 1) at another call; any other use of it is an error.
def insertPtrs(
    entity: Entity,
    loop: OP.Loop,
    entities: List[Tuple[Entity, Rewriter]],
    match: Callable[[OP.Arg], bool],
) -> None:
    funcs = [e for e, _ in entities if isinstance(e, Function)]
    rewriters = {e.name: r for e, r in entities if isinstance(e, Function)}

    targets = argumentTargets(entity, loop, funcs, match)
    for func in funcs:
        for param_idx in sorted(targets.get(func.name, set())):
            param = parameters(func)[param_idx]

            def fail(node: Cursor, what: str, name: str = param.spelling, func: Function = func) -> None:
                raise OP.OpError(
                    f"{func.name}: argument {name} may only be indexed, but {what}: {tokenText(node)}", loop.loc
                )

            for _, parents, use_index, _ in ptrUses(func, rewriters[func.name], param, targets, fail):
                # &p[i] and &*p would be raw pointers without the stride.
                outer = parents[enclosing(parents, use_index - 1)]
                if outer.kind == CursorKind.UNARY_OPERATOR and operatorToken(outer).spelling == "&":
                    fail(outer, "its element's address is taken")


# The parameters each matched argument reaches, by function.
def argumentTargets(
    entity: Entity, loop: OP.Loop, funcs: List[Function], match: Callable[[OP.Arg], bool]
) -> Dict[str, Set[int]]:
    targets: Dict[str, Set[int]] = {}
    for arg_idx, arg in enumerate(loop.args):
        if match(arg):
            mapParam(entity, arg_idx, funcs, lambda func, idx: targets.setdefault(func.name, set()).add(idx))

    return targets


# Retype param, a pointer, to an op::kernel::Ptr of its element type, and
# yield each use of it as an element, as (subscript or dereference, its
# ancestors, its index among them, whether a dereference).  A pass to a
# helper whose parameter is in targets is skipped; any other use fails, and
# so do references in skip.
def ptrUses(
    func: Function,
    rewriter: Rewriter,
    param: Cursor,
    targets: Dict[str, Set[int]],
    fail: Callable[[Cursor, str], None],
    skip: Set[int] = frozenset(),
) -> Iterator[Tuple[Cursor, Tuple[Cursor, ...], int, bool]]:
    typ = param.type.get_canonical()
    if typ.kind == TypeKind.POINTER:
        element = typ.get_pointee()
    elif typ.kind in [TypeKind.CONSTANTARRAY, TypeKind.INCOMPLETEARRAY]:
        element = typ.element_type
    else:
        fail(param, "it is not a pointer")

    name = param.spelling
    rewriter.update(extentToSpan(param.extent), lambda _: f"op::kernel::Ptr<{element.spelling}> {name}")

    for node, parents in walkWithParents(func.ast):
        if node.kind != CursorKind.DECL_REF_EXPR or node.hash in skip:
            continue
        if node.referenced is None or node.referenced != param:
            continue

        use_index = enclosing(parents, len(parents) - 1)
        target = parents[use_index]
        if target.kind == CursorKind.CALL_EXPR:
            callee = target.referenced
            passed = {idx for idx, arg in enumerate(target.get_arguments()) if stripParens(arg)[0] == node}
            if callee is None or not passed or not passed <= targets.get(callee.spelling, set()):
                fail(target, "a call takes it")
            continue

        if target.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR and stripParens(next(target.get_children()))[0] == node:
            yield target, parents, use_index, False
        elif target.kind == CursorKind.UNARY_OPERATOR and operatorToken(target).spelling == "*":
            yield target, parents, use_index, True
        else:
            fail(target, "it is used as a pointer")


def parameters(func: Function) -> List[Cursor]:
    return [child for child in func.ast.get_children() if child.kind == CursorKind.PARM_DECL]


def tokenText(node: Cursor) -> str:
    return " ".join(token.spelling for token in node.get_tokens())


# Every node below node with its ancestors, outermost first.
def walkWithParents(node: Cursor, parents: Tuple[Cursor, ...] = ()) -> Iterator[Tuple[Cursor, Tuple[Cursor, ...]]]:
    for child in node.get_children():
        yield child, parents + (node,)
        yield from walkWithParents(child, parents + (node,))


# The index in parents of the nearest ancestor from index up that is more
# than an implicit conversion or parentheses.
def enclosing(parents: Tuple[Cursor, ...], index: int) -> int:
    while parents[index].kind in [CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR]:
        index -= 1

    return index


# node without implicit conversions and parentheses, and how many
# parentheses it had.
def stripParens(node: Cursor) -> Tuple[Cursor, int]:
    levels = 0
    while node.kind in [CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR]:
        children = list(node.get_children())
        if len(children) != 1:
            break

        levels += node.kind == CursorKind.PAREN_EXPR
        node = children[0]

    return node, levels


# The leftmost term of a chain of + and -: as written, without parentheses,
# the parentheses opened around the chain on the way to it, and the operator
# after it (none if node is no such chain).
def leftmostTerm(node: Cursor) -> Tuple[Cursor, Cursor, int, Optional[Token]]:
    levels, op = 0, None
    while True:
        stripped, inner = stripParens(node)
        if stripped.kind != CursorKind.BINARY_OPERATOR or operatorToken(stripped).spelling not in ["+", "-"]:
            return node, stripped, levels, op

        levels += inner
        op = operatorToken(stripped)
        node = next(stripped.get_children())


def operatorToken(node: Cursor) -> Token:
    tokens = list(node.get_tokens())
    operand = next(node.get_children())

    if node.kind == CursorKind.UNARY_OPERATOR:
        return tokens[0] if tokens[0].extent.start.offset < operand.extent.start.offset else tokens[-1]

    return next(token for token in tokens if token.extent.start.offset >= operand.extent.end.offset)


# Whether statement discards the value of node, its child.
def discarded(node: Cursor, statement: Cursor) -> bool:
    children = list(statement.get_children())

    if statement.kind in [CursorKind.COMPOUND_STMT, CursorKind.DEFAULT_STMT, CursorKind.LABEL_STMT]:
        return True
    if statement.kind == CursorKind.IF_STMT:
        return node != children[0]
    if statement.kind in [CursorKind.FOR_STMT, CursorKind.WHILE_STMT, CursorKind.CASE_STMT]:
        return node == children[-1]
    if statement.kind == CursorKind.DO_STMT:
        return node == children[0]

    return False


def location(extent: SourceRange, end: bool = False) -> Location:
    point = extent.end if end else extent.start
    return Location(point.line, point.column)


def rewriteIncrements(
    func: Function, rewriter: Rewriter, param_idx: int, targets: Dict[str, Set[int]], loop: OP.Loop, call: str
) -> None:
    param = parameters(func)[param_idx]
    name = param.spelling

    def fail(node: Cursor, what: str) -> None:
        raise OP.OpError(
            f"{func.name}: OP_INC argument {name} may only be incremented, but {what}: {tokenText(node)}", loop.loc
        )

    consumed: Set[int] = set()
    for target, parents, use_index, deref in ptrUses(func, rewriter, param, targets, fail, consumed):
        index = enclosing(parents, use_index - 1)
        expr = parents[index]
        children = list(expr.get_children())
        operator = expr.kind in [
            CursorKind.BINARY_OPERATOR,
            CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
            CursorKind.UNARY_OPERATOR,
        ]
        op = operatorToken(expr) if operator else None
        assigned = expr.kind != CursorKind.UNARY_OPERATOR and operator and stripParens(children[0])[0] == target

        # Each edit: a span, its new text, and whether it replaces the span
        # outright (dropping any other edit within it).
        edits: List[Tuple[Span, str, bool]] = []
        start = location(expr.extent)
        insert = True

        if assigned and op.spelling in ["+=", "-="]:
            minus = op.spelling == "-="
            edits.append((extentToSpan(op.extent), ", -(" if minus else ",", False))
            suffix = "))" if minus else ")"
        elif expr.kind == CursorKind.UNARY_OPERATOR and op.spelling in ["++", "--"]:
            step = ", -1)" if op.spelling == "--" else ", 1)"
            insert = op.extent.start.offset > target.extent.start.offset
            edits.append((extentToSpan(op.extent), step if insert else f"{call}(", False))
            suffix = None if insert else step
        elif assigned and op.spelling == "=":
            # x = x + a - b: drop "x +" (or just "x" before a "-"), keeping
            # the parentheses opened on the way to it.  x = a + x: drop "+ x".
            term, stripped, levels, term_op = leftmostTerm(children[1])
            rhs = stripParens(children[1])[0]
            rhs_op = operatorToken(rhs) if term_op is not None else None
            last = list(rhs.get_children())[-1] if term_op is not None else None

            if term_op is not None and tokenText(stripped) == tokenText(target):
                upto = term_op if term_op.spelling == "+" else term
                replaced = Span(location(op.extent), location(upto.extent, end=True))
                edits.append((replaced, ", " + "(" * levels, True))
                suffix = ")"
                duplicate = stripped
            elif rhs_op is not None and rhs_op.spelling == "+" and tokenText(stripParens(last)[0]) == tokenText(target):
                edits.append((extentToSpan(op.extent), ",", False))
                replaced = Span(location(rhs_op.extent), location(last.extent, end=True))
                edits.append((replaced, ")", True))
                suffix = None
                duplicate = stripParens(last)[0]
            else:
                fail(expr, "it is assigned")

            for ref in duplicate.walk_preorder():
                if ref.kind == CursorKind.DECL_REF_EXPR and ref.referenced is not None and ref.referenced == param:
                    consumed.add(ref.hash)
        elif assigned:
            fail(expr, "it is assigned")
        else:
            fail(expr, "it is read")

        statement = index - 1
        while parents[statement].kind == CursorKind.UNEXPOSED_EXPR:
            statement -= 1
        if not discarded(parents[statement + 1] if statement + 1 < index else expr, parents[statement]):
            fail(expr, "the value of its increment is used")

        if insert and deref and location(target.extent) == start:
            edits.append((extentToSpan(target.extent), f"{call}({name}[0]", False))
        else:
            if insert:
                edits.append((Span(start, start), f"{call}(", False))
            if deref:
                edits.append((extentToSpan(target.extent), f"{name}[0]", False))

        if suffix is not None:
            end = location(expr.extent, end=True)
            edits.append((Span(end, end), suffix, False))

        for span, text, replace in edits:
            if replace:
                rewriter.replace(span, text)
            else:
                rewriter.update(span, lambda _, text=text: text)


def insertStride(
    ast: Cursor, rewriter: Rewriter, param: str, id: int, is_vec: bool, stride: Callable[[int], str]
) -> None:
    if not hasattr(rewriter, "updated_nodes"):
        rewriter.updated_nodes = set()

    for node in ast.walk_preorder():
        if node.kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
            continue

        ident, subscript = node.get_children()
        if is_vec:
            if next(ident.get_children()).kind != CursorKind.ARRAY_SUBSCRIPT_EXPR:
                continue

            ident, _ = next(ident.get_children()).get_children()

        if ident.spelling == param:
            node_id = node.hash if hasattr(node, "hash") else id(node)
            if node_id in rewriter.updated_nodes:
                continue

            rewriter.updated_nodes.add(node_id)
            span = extentToSpan(subscript.extent)

            rewriter.update(span, lambda s: f"({s}) * {stride(id)}")

def writeSource(entities: List[Tuple[Entity, Rewriter]]) -> str:
    source = ""
    while len(entities) > 0:
        for i in range(len(entities)):
            entity, rewriter = entities[i]
            resolved = True

            for dependency in entity.depends:
                if safeFind(entities, lambda e: e[0].name == dependency):
                    resolved = False
                    break

            if resolved:
                entities.pop(i)
                if source == "":
                    source = rewriter.rewrite()
                else:
                    source = source + "\n\n" + rewriter.rewrite()

                if isinstance(entity, Type):
                    source = source + ";"

                break

    return source
