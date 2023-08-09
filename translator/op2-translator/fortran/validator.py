from typing import Any, Callable, List, Optional, Tuple, Union, Set

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr

import fortran.translator.kernels as ftk
import fortran.util as fu
import op as OP
from op import OpError
from store import Application, Entity, Function, Program
from util import find, safeFind


def validateLoop(loop: OP.Loop, program: Program, app: Application) -> None:
    kernel_entities = app.findEntities(loop.kernel, program, [])

    if len(kernel_entities) == 0:
        raise OpError(f"unable to find kernel subroutine for {loop.kernel}")
    elif len(kernel_entities) > 1:
        raise OpError(f"ambiguous kernel subroutine for {loop.kernel}")

    dependencies, unknown_dependencies = ftk.extractDependencies(kernel_entities, app, [])
    entities = kernel_entities + list(filter(lambda e: isinstance(e, Function), dependencies))

    if len(unknown_dependencies) > 0:
        printViolations(loop, "unknown subroutine/function references", list(set(unknown_dependencies)))
        loop.fallback = True

    seen_entity_names = []
    for entity in entities:
        if entity.name in seen_entity_names:
            raise OpError(f"ambiguous subroutine/function {entity.name} used in kernel {loop.kernel}")

        seen_entity_names.append(entity.name)

    if len(loop.args) != len(kernel_entities[0].parameters):
        raise OpError(
            f"op_par_loop argument list length ({len(loop.args)}) mismatch "
            f"(expected: {len(kernel_entities[0].parameters)}, kernel subroutine: {loop.kernel})",
            loop.loc,
        )
        return

    for arg in loop.args:
        if isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.WRITE:
            print(f"{loop.loc}: Warning: {loop.kernel} indirect OP_WRITE\n")
            # loop.fallback = True
            break

    # Check parameter/const conflict and const writes
    const_ptrs = app.constPtrs()

    violations = []
    read_violations = []

    for entity in entities:
        const_param_aliases = set()

        for idx, param in enumerate(entity.parameters):
            if param not in const_ptrs:
                continue

            const_param_aliases.add(param)
            violations.append(f"In {entity.name}: parameter {idx + 1} ({param})")

        for const_ptr in const_ptrs:
            if const_ptr in const_param_aliases:
                continue

            checkConstRead(entity, const_ptr, read_violations)

    if len(violations) > 0:
        printViolations(loop, "subroutine/function parameter and const conflict", violations)

    if len(read_violations) > 0:
        printViolations(loop, "const written", read_violations)
        loop.fallback = True

    # Check for disallowed statements (IO, exit, ...)
    # for entity in entities:
    #     violations = []
    #     checkStatements(entity, violations)

    #     if len(violations) > 0:
    #         printViolations(loop, "invalid statements", violations)
    #         loop.fallback = True

    # Check for slice expressions for args with stride insertion (gbl reductions, dats)
    for idx, arg in enumerate(loop.args):
        if not (
            isinstance(arg, OP.ArgGbl) and arg.access_type in [OP.AccessType.MIN, OP.AccessType.MAX, OP.AccessType.INC]
        ) and not isinstance(arg, OP.ArgDat):
            continue

        if (isinstance(arg, OP.ArgGbl) and arg.dim == 1) or (isinstance(arg, OP.ArgDat) and loop.dat(arg).dim == 1):
            continue

        violations = []
        fu.mapParam(kernel_entities[0], idx, entities, checkSlice, entities, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]
            printViolations(
                loop, "element-wise access incompatible with stride insertion", violations, (idx, param_name)
            )

            loop.fallback = True

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
        if not isinstance(arg, OP.ArgDat) or arg.access_type != OP.AccessType.INC:
            continue

        violations = []
        fu.mapParam(kernel_entities[0], idx, entities, checkInc, entities, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]
            printViolations(loop, "marked OP_INC but not incremented", violations, (idx, param_name))

            loop.fallback = True

    # Check for runtime-dimension stack arrays (very slow for GPU)
    violations = []
    for entity in entities:
        checkRuntimeDimensionArrays(entity, app.constPtrs(), violations)

    if len(violations) > 0:
        printViolations(loop, "runtime dimension local arrays", violations)


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


def checkRuntimeDimensionArrays(func: Function, consts: Set[str], violations: List[str]) -> None:
    spec = fpu.get_child(func.ast, f2003.Specification_Part)

    if spec is None:
        return

    blacklist = set(consts)
    blacklist.update(func.parameters)

    for type_decl in fpu.walk(spec, f2003.Type_Declaration_Stmt):
        for entity_decl in fpu.walk(type_decl, f2003.Entity_Decl):
            name = fpu.get_child(entity_decl, f2003.Name).string
            if name.lower() in func.parameters:
                continue

            shape_spec = fpu.get_child(entity_decl, f2003.Explicit_Shape_Spec_List)
            if shape_spec is None:
                dimension_spec = fpu.walk(type_decl, f2003.Dimension_Attr_Spec)

                if len(dimension_spec) > 0:
                    shape_spec = fpu.get_child(dimension_spec[0], f2003.Explicit_Shape_Spec_List)

            if shape_spec is None:
                continue

            for ref_node in fpu.walk(shape_spec, f2003.Name):
                if ref_node.string.lower() in blacklist:
                    violations.append(f"In {func.name}: variable {name}, dimension {ref_node.string.lower()}")


def checkSlice(func: Function, param_idx: int, funcs: List[Function], violations: List[str]) -> None:
    dims = fu.parseDimensions(func, func.parameters[param_idx])

    if dims is None:
        return

    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    def msg(s: str) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {func.parameters[param_idx]}): {s}"

    for node in fpu.walk(execution_part, f2003.Name):
        if node.string.lower() != func.parameters[param_idx]:
            continue

        if getattr(node, "parent", None) and isinstance(node.parent, f2003.Part_Ref):
            subscript_list = fpu.get_child(node.parent, f2003.Section_Subscript_List)

            for subscript in subscript_list.children:
                if isinstance(subscript, (f2003.Subscript_Triplet, f2003.Vector_Subscript)):
                    violations.append(msg(f"{fu.getItem(node).line}"))

            continue

        if isinstance(node.parent, f2003.Actual_Arg_Spec_List):
            continue

        if isinstance(node.parent, f2003.Section_Subscript_List):
            func_name_node = fpu.get_child(node.parent.parent, f2003.Name)
            if func_name_node is not None:
                func_ref = safeFind(funcs, lambda f: f.name == func_name_node.string.lower())

                if func_ref is not None:
                    continue

        violations.append(msg(f"{fu.getItem(node).line}"))


def checkConstRead(func: Function, const_ptr: str, violations: List[str]) -> None:
    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    def msg(s: str) -> str:
        return f"In {func.name} (const {const_ptr}): {s}"

    for node in fpu.walk(execution_part, f2003.Assignment_Stmt):
        if fu.isRef(node.items[0], const_ptr):
            violations.append(msg(f"{fu.getItem(node).line}"))


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

        if simplify(parse_expr(f"{expr.replace('x', '0')}", evaluate=False)) == 0:
            violations.append(msg(f"no-op: {fu.getItem(node).line}"))
        elif simplify(parse_expr(f"({expr}) - (x + {expr.replace('x', '0')})", evaluate=False)) != 0:
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
