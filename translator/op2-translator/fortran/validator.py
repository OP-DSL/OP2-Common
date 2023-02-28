from typing import Any, List, Optional, Union, Callable, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

from sympy.parsing.sympy_parser import parse_expr
from sympy import simplify

import fortran.translator.kernels as ftk
import fortran.util as fu
import op as OP

from op import OpError
from store import Application, Entity, Function, Program
from util import find, safeFind


def validateLoop(loop: OP.Loop, program: Program, app: Application) -> None:
    kernel_entities = app.findEntities(loop.kernel, program, [])

    if len(kernel_entities) == 0:
        raise OpError(f"unable to find kernel function for {loop.kernel}")
    elif len(kernel_entities) > 1:
        raise OpError(f"ambiguous kernel function for {loop.kernel}")

    dependencies, unknown_dependencies = ftk.extractDependencies(kernel_entities, app, [])
    entities = kernel_entities + list(filter(lambda e: isinstance(e, Function), dependencies))

    if len(unknown_dependencies) > 0:
        printViolations(loop, "unknown subroutine/function references", list(set(unknown_dependencies)))
        loop.fallback = True

    seen_entity_names = []
    for entity in entities:
        if entity.name in seen_entity_names:
            raise OpError(f"ambiguous function {entity.name} used in kernel {loop.kernel}")

        seen_entity_names.append(entity.name)

    if len(loop.args) != len(kernel_entities[0].parameters):
        raise OpError(
            f"op_par_loop argument list length ({len(loop.args)}) mismatch "
            f"(expected: {len(kernel_entities[0].parameters)}, kernel function: {loop.kernel})",
            loop.loc,
        )
        return

    # Check for disallowed statements (IO, exit, ...)
    # for entity in entities:
    #     violations = []
    #     checkStatements(entity, violations)

    #     if len(violations) > 0:
    #         printViolations(loop, "invalid statements", violations)
    #         loop.fallback = True

    # Check for args marked OP_READ or arg_idx but appear to be written
    for idx, arg in enumerate(loop.args):
        if isinstance(arg, OP.ArgInfo) or (hasattr(arg, "access_type") and arg.access_type != OP.AccessType.READ):
            continue

        violations = []
        fu.mapParam(kernel_entities[0], idx, entities, checkRead, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]

            if isinstance(arg, OP.ArgIdx):
                msg = "is an op_arg_idx but was written"
            else:
                msg = "marked OP_READ but was written"

            printViolations(loop, msg, violations, (idx, param_name))

            loop.fallback = True

    # Check for OP_INC args that don't appear to be incremented
    for idx, arg in enumerate(loop.args):
        if isinstance(arg, OP.ArgIdx) or isinstance(arg, OP.ArgInfo) or arg.access_type != OP.AccessType.INC:
            continue

        violations = []
        fu.mapParam(kernel_entities[0], idx, entities, checkInc, entities, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]
            printViolations(loop, "marked OP_INC but not incremented", violations, (idx, param_name))

            loop.fallback = True


def printViolations(loop: OP.Loop, warning: str, violations: List[str], arg: Optional[Tuple[int, str]] = None) -> None:
    if arg is not None:
        print(f"{loop.loc}: Warning: arg {arg[0] + 1} ({arg[1]}) of {loop.kernel} {warning}:")
    else:
        print(f"{loop.loc}: Warning: {loop.kernel} {warning}:")

    for violation in violations[:5]:
        print(f"    {violation}")

    if len(violations) > 5:
        print(f"    ({len(violations) - 5} more)")

    print()


def checkStatements(func: Function, violations: List[str]) -> None:
    for node in fpu.walk(
        func.ast,
        (
            f2003.Allocate_Stmt,
            f2003.Backspace_Stmt,
            f2003.Close_Stmt,
            f2003.Deallocate_Stmt,
            f2003.Endfile_Stmt,
            f2003.Exit_Stmt,
            f2003.Flush_Stmt,
            f2003.Inquire_Stmt,
            f2003.Open_Stmt,
            f2003.Print_Stmt,
            f2003.Read_Stmt,
            f2003.Rewind_Stmt,
            f2003.Stop_Stmt,
            f2003.Wait_Stmt,
            f2003.Write_Stmt,
        ),
    ):
        violations.append(f"In {func.name}: {fu.getItem(node).line}")


def checkRead(func: Function, param_idx: int, violations: List[str]) -> None:
    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    def msg(s: str) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {func.parameters[param_idx]}): {s}"

    for node in fpu.walk(execution_part, f2003.Assignment_Stmt):
        if fu.isRef(node.items[0], func.parameters[param_idx]):
            violations.append(msg(f"{fu.getItem(node).line}"))


def checkInc(func: Function, param_idx: int, funcs: List[Function], violations: List[str]) -> None:
    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    def msg(s: str) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {func.parameters[param_idx]}): {s}"

    assignment_lhs_refs = []
    other_refs = []

    # Sort all the Name node refs into assignment LHS or something else
    for node in fu.walkRefs(execution_part, func.parameters[param_idx]):
        if getattr(node, "parent", None) is None:
            continue

        if isinstance(node.parent, f2003.Assignment_Stmt):
            if id(node.parent.items[0]) == id(node):
                assignment_lhs_refs.append(node)
                continue

        if (
            isinstance(node.parent, f2003.Part_Ref)
            and getattr(node.parent, "parent", None) is not None
            and isinstance(node.parent.parent, f2003.Assignment_Stmt)
        ):
            if id(node.parent.parent.items[0]) == id(node.parent):
                assignment_lhs_refs.append(node)
                continue

        other_refs.append(node)

    # Remove the RHS refs from other_refs
    for node in assignment_lhs_refs:
        assignment_node = node.parent

        if isinstance(assignment_node, f2003.Part_Ref):
            assignment_node = assignment_node.parent

        for node2 in fu.walkRefs(assignment_node.items[2], func.parameters[param_idx]):
            other_refs = list(filter(lambda r: id(r) != id(node2), other_refs))

    # Everything left in other_refs must be either passed as a param to a function or a violation
    for node in other_refs:
        call = fu.getCall(node, funcs)

        if call is not None:
            continue

        violations.append(msg(f"invalid context: {fu.getItem(node).line}"))

    # Finally check the assignments in assignment_lhs_refs
    for node in assignment_lhs_refs:
        assignment_node = fu.walkOut(node, f2003.Assignment_Stmt)

        rhs_refs = fu.walkRefs(assignment_node.items[2], node.string)

        if len(rhs_refs) > 1:
            violations.append(msg(f"multi-ref: {fu.getItem(node).line}"))
            continue

        if len(rhs_refs) == 0:
            violations.append(msg(f"no-ref: {fu.getItem(node).line}"))
            continue

        rhs_ref = rhs_refs[0]

        if isinstance(assignment_node.items[0], f2003.Part_Ref):
            rhs_ref = rhs_ref.parent

        if repr(rhs_ref) != repr(assignment_node.items[0]):
            violations.append(msg(f"index mismatch: {fu.getItem(node).line}"))
            continue

        try:
            count = [0]
            expr = simplifyLevel2(assignment_node.items[2], assignment_node.items[0], node.string, count)
        except OpError as e:
            violations.append(msg(f"invalid usage: {fu.getItem(node).line}"))
            continue

        sym_expr = parse_expr(f"({expr}) - (x + {expr.replace('x', '0')})", evaluate=False)
        if simplify(sym_expr) != 0:
            violations.append(msg(f"non increment: {fu.getItem(node).line}"))


def simplifyLevel2(node: f2003.Base, ref: Union[f2003.Name, f2003.Part_Ref], ref_name: str, count: List[int]) -> str:
    def incSym() -> str:
        count[0] += 1
        return f"y{count[0]}"

    if isinstance(node, f2003.Parenthesis):
        return f"({simplifyLevel2(node.children[0], ref, ref_name, count)})"

    if isinstance(node, f2003.Level_2_Expr):
        if len(fu.walkRefs(node.items[0], ref_name)) > 0:
            return f"{simplifyLevel2(node.items[0], ref, ref_name, count)} {node.items[1]} {incSym()}"

        if len(fu.walkRefs(node.items[2], ref_name)) > 0:
            return f"{incSym()} {node.items[1]} {simplifyLevel2(node.items[2], ref, ref_name, count)}"

        assert False

    if isinstance(node, f2003.Name):
        if node.string.lower() == ref.string.lower():
            return "x"

        return incSym()

    if isinstance(node, f2003.Part_Ref):
        if node.items[0].string.lower() == ref.items[0].string.lower():
            return "x"

        return incSym()

    if len(fu.walkRefs(node, ref_name)) > 0:
        raise OpError(str())

    return incSym()
