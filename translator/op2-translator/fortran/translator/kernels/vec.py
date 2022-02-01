import re
from typing import Any, Dict, Tuple
from xml.etree.ElementTree import Element, dump

from store import Application, Kernel
from util import SourceBuffer, find, indexSplit


def translateKernel(config: Dict[str, Any], source: str, kernel: Kernel, app: Application) -> Tuple[str, int]:
    buffer = SourceBuffer(source)

    # Collect indirect increment identifiers TODO: Tidy
    loop = find(app.loops, lambda l: l.kernel == kernel.name)
    increments = []
    for param, arg in zip(kernel.params, loop.args):
        if arg.indirect and arg.acc == "OP_INC":
            increments.append(param[0])

    # Check if this kernel is for an indirect loop
    ind = 0
    loop = find(app.loops, lambda l: l.kernel == kernel.name)
    for arg in loop.args:
        if arg.indirect:
            ind = ind + arg.indirect
            print(arg.var, arg.typ, arg.dim)

    # modify only indirect loops
    if ind:
        # Ast traversal
        subroutine = kernel.ast.find("file/subroutine")
        body = subroutine.find("body")

        # Augment kernel subroutine name
        index = int(subroutine.attrib["line_begin"]) - 1
        buffer.apply(index, lambda line: line.replace(kernel.name, kernel.name + "_vec"))

        # Augment subroutine header with additional argument
        arguments = kernel.ast.find("file/subroutine/header/arguments")
        line_index = int(arguments.attrib["line_begin"]) - 1
        line = buffer.get(line_index).strip()
        continuations = 0
        while line.endswith("&"):
            continuations += 1
            line = line[:-1].strip() + buffer.get(line_index + continuations).strip()[1:].strip()

        # remove closing )
        i = line.find(")")
        line = line[:i]
        para = line[line.find("(") + 1 :].split(",")  # collect the kernel parameters -- for later
        para = [x.strip(" ") for x in para]

        buffer.update(line_index, line.strip() + ",idx)")

        # Remove old continuations
        for i in range(1, continuations + 1):
            buffer.remove(line_index + i)

        # remove Vec args from sepcification - note: assumes local vars are on separate lines
        spec = body.find("specification")
        s = int(spec.attrib["line_begin"])
        e = int(spec.attrib["line_end"])
        for i in range(s, e):
            line = buffer.get(i)
            Vars = re.split(",|::", line)
            Vars = [x.strip(" ") for x in Vars]
            for p in para:
                if p in Vars:
                    ui = i
            buffer.update(ui, "")

        # Add additional argument - idx
        spec = body.find("specification")
        indent = " " * int(spec.attrib["col_begin"])
        buffer.insert(int(spec.attrib["line_begin"]), indent + "INTEGER(kind=4) :: idx")

        source = buffer.translate()

    return source, ind
