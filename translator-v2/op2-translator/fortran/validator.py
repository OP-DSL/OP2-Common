import logging
from collections import Counter
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set

import fparser.two.Fortran2003 as f2003
import fparser.two.utils as fpu

import fortran.translator.kernels as ftk
import fortran.util as fu
import op as OP
from op import OpError
from store import Application, Entity, Function, Program
from util import find, safeFind

logger = logging.getLogger(__name__)


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

    # for arg in loop.args:
    #     if isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.WRITE:
    #         print(f"{loop.loc}: Warning: {loop.kernel} indirect OP_WRITE\n")
    #         # loop.fallback = True
    #         break

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

        checkConstRead(entity, list(filter(lambda c: c not in const_param_aliases, const_ptrs)), read_violations)

    if len(violations) > 0:
        printViolations(loop, "subroutine/function parameter and const conflict", violations)

    if len(read_violations) > 0:
        printViolations(loop, "const written", read_violations)
        loop.fallback = True

    # Add used consts to the loop
    for entity in entities:
        for name in fpu.walk(entity.ast, f2003.Name):
            if name.string.lower() in const_ptrs and name.string.lower() not in entity.parameters:
                loop.addConst(name.string.lower())

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
            isinstance(arg, OP.ArgGbl) and arg.access_type in [OP.AccessType.MIN, OP.AccessType.MAX, OP.AccessType.INC, OP.AccessType.WORK]
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

    # Check that kernel parameter shapes match op_arg dimensions.
    #
    # For a dim=1 op_arg the dispatch loop passes a single-element indexing of
    # an assumed-shape array (e.g. dat0(1, n) or gbl3(1)).  Fortran only
    # accepts that when the dummy is a scalar - a `dimension(1)` dummy triggers
    # "Element of assumed-shape or pointer array passed to array dummy" at
    # compile time.  For a dim=N op_arg the dispatch passes N elements (e.g.
    # dat0(:, n)), which sequence association lets an explicit-shape dummy of
    # any rank take - dimension(5, 5) for dim=25 - so long as it has N
    # elements.  insertStrides and the C translation both index by the dummy's
    # own shape.
    for idx, arg in enumerate(loop.args):
        if not isinstance(arg, (OP.ArgDat, OP.ArgGbl)):
            continue

        if isinstance(arg, OP.ArgGbl):
            arg_dim = arg.dim
        else:
            arg_dim = loop.dat(arg).dim

        if arg_dim is None:
            continue  # runtime dim - can't check statically

        violations = []
        fu.mapParam(kernel_entities[0], idx, entities, checkParamShape, arg_dim, violations)

        if len(violations) > 0:
            param_name = kernel_entities[0].parameters[idx]
            expected = "scalar" if arg_dim == 1 else f"array of {arg_dim} elements"
            printViolations(loop, f"kernel parameter shape mismatch (op_arg dim={arg_dim}, expected {expected})",
                            violations, (idx, param_name))
            loop.fallback = True

    # Check for runtime-dimension stack arrays (very slow for GPU)
    violations = []
    for entity in entities:
        checkRuntimeDimensionArrays(entity, app.constPtrs(), violations)

    if len(violations) > 0:
        printViolations(loop, "runtime dimension local arrays", violations)


def printViolations(loop: OP.Loop, warning: str, violations: List[str], arg: Optional[Tuple[int, str]] = None) -> None:
    if arg is not None:
        header = f"{loop.loc}: Warning: arg {arg[0] + 1} ({arg[1]}) of {loop.kernel} {warning}:"
    else:
        header = f"{loop.loc}: Warning: {loop.kernel} {warning}:"

    lines = [header] + [f"    {v}" for v in violations[:5]]
    if len(violations) > 5:
        lines.append(f"    ({len(violations) - 5} more)")

    logger.warning("\n".join(lines))


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


def checkParamShape(func: Function, param_idx: int, arg_dim: int, violations: List[str]) -> None:
    """Verify a kernel parameter's declared shape matches the op_arg dimension.

    dim=1 op_args require the kernel parameter to be a scalar (no dimension
    spec).  dim>1 op_args require an explicit-shape array of any rank with
    exactly dim elements - dimension(25) or dimension(5, 5) for dim=25.  A
    shape whose size isn't known here (a bound that is not an integer
    literal, e.g. dimension(nvar)) passes, as do assumed-shape and
    assumed-size parameters, since parseDimensions returns None for those
    and for scalars alike - matching how the slice check already ignores
    them.
    """
    dims = fu.parseDimensions(func, func.parameters[param_idx])

    def msg(reason: str) -> str:
        return f"In {func.name} (arg {param_idx + 1}, {func.parameters[param_idx]}): {reason}"

    if arg_dim == 1:
        if dims is not None:
            violations.append(msg(f"declared with dimension {dims}, must be a scalar"))
    else:
        if dims is None:
            return  # scalar or assumed-shape - leave to other checks

        size = 1
        for lb, ub in dims:
            try:
                size *= int(ub) - int(lb) + 1
            except ValueError:
                return  # not an integer literal - size unknown, nothing to compare

        if size != arg_dim:
            violations.append(msg(f"declared with shape {dims} of {size} elements, must have {arg_dim}"))


def checkConstRead(func: Function, const_ptrs: List[str], violations: List[str]) -> None:
    execution_part = fpu.get_child(func.ast, f2003.Execution_Part)
    assert execution_part != None

    def msg(s: str) -> str:
        return f"In {func.name} (const {const_ptr}): {s}"

    for node in fpu.walk(execution_part, f2003.Assignment_Stmt):
        for const_ptr in const_ptrs:
            if fu.isRef(node.items[0], const_ptr):
                violations.append(msg(f"{fu.getItem(node).line}"))
                break


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

        if len(rhs_refs) == 0:
            violations.append(msg(f"no-ref: {fu.getItem(node).line}"))
            continue

        # The RHS may use the ref any number of times, but only ever as the
        # LHS itself - the same element, if it is indexed
        lhs = refKey(assignment_node.items[0])

        if any(refKey(refExpr(rhs_ref)) != lhs for rhs_ref in rhs_refs):
            violations.append(msg(f"index mismatch: {fu.getItem(node).line}"))
            continue

        try:
            rhs = incrementPoly(assignment_node.items[2], lhs, node.string)
        except OpError as e:
            violations.append(msg(f"invalid usage: {fu.getItem(node).line}"))
            continue

        # An increment is the ref plus terms without it, and those don't cancel
        delta = polyAdd(rhs, {((lhs, 1),): Fraction(1)}, -1)

        if any(var == lhs for monomial in delta for var, _ in monomial):
            violations.append(msg(f"non increment: {fu.getItem(node).line}"))
        elif len(delta) == 0:
            violations.append(msg(f"no-op: {fu.getItem(node).line}"))


def refKey(node: f2003.Base) -> str:
    return str(node).lower()


# The expression a ref is: the indexed array element when it is one
def refExpr(ref: f2003.Name) -> f2003.Base:
    if isinstance(ref.parent, f2003.Part_Ref) and ref.parent.items[0] is ref:
        return ref.parent

    return ref


# A polynomial in an increment's variables, as {monomial: coefficient}, with
# a monomial the sorted (variable, power) pairs of its variables.
Poly = Dict[Tuple[Tuple[str, int], ...], Fraction]


def polyAdd(p: Poly, q: Poly, scale: int = 1) -> Poly:
    total = dict(p)
    for monomial, coefficient in q.items():
        total[monomial] = total.get(monomial, 0) + scale * coefficient

    return {monomial: coefficient for monomial, coefficient in total.items() if coefficient != 0}


def polyMul(p: Poly, q: Poly) -> Poly:
    product: Poly = {}
    for monomial1, coefficient1 in p.items():
        for monomial2, coefficient2 in q.items():
            powers = Counter(dict(monomial1)) + Counter(dict(monomial2))
            monomial = tuple(sorted(powers.items()))
            product[monomial] = product.get(monomial, 0) + coefficient1 * coefficient2

    return {monomial: coefficient for monomial, coefficient in product.items() if coefficient != 0}


# Reads an increment's RHS as a polynomial in the ref (keyed lhs) and whatever
# else it adds, subtracts and multiplies. Anything other than those operations
# and numeric literals is a variable of its own - named by its source text, so
# only an identical term can cancel it - as long as it does not use the ref.
# If it does - a function or power of the ref, or division by or of it, which
# with integer operands truncates - this raises OpError.
def incrementPoly(node: f2003.Base, lhs: str, ref_name: str) -> Poly:
    if isinstance(node, f2003.Parenthesis):
        return incrementPoly(node.items[1], lhs, ref_name)

    if isinstance(node, f2003.Level_2_Unary_Expr):
        op, operand = node.items
        return polyAdd({}, incrementPoly(operand, lhs, ref_name), -1 if op == "-" else 1)

    if isinstance(node, f2003.Level_2_Expr):
        left, op, right = node.items
        return polyAdd(incrementPoly(left, lhs, ref_name), incrementPoly(right, lhs, ref_name), -1 if op == "-" else 1)

    if isinstance(node, f2003.Add_Operand) and node.items[1] == "*":
        return polyMul(incrementPoly(node.items[0], lhs, ref_name), incrementPoly(node.items[2], lhs, ref_name))

    if isinstance(node, (f2003.Int_Literal_Constant, f2003.Real_Literal_Constant)):
        value = Fraction(node.items[0].lower().replace("d", "e"))
        return {(): value} if value != 0 else {}

    key = refKey(node)

    if key != lhs and len(fu.walkRefs(node, ref_name)) > 0:
        raise OpError(f"unsupported use of {ref_name}: {node}")

    return {((key, 1),): Fraction(1)}
