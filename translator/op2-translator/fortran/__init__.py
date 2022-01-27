from types import MethodType

import op as OP
from fortran.parser import parseKernel, parseProgram
from fortran.translator.program import translateProgram
from language import Lang

lang = Lang(
    name="fortran",
    com_delim="!",
    zero_idx=False,
    source_exts=["F90", "F95"],
    include_ext="inc",
    types=["integer(4)", "real(8)"],
)


def formatType(self, typ: OP.Type) -> str:
    if isinstance(typ, OP.Int):
        if not typ.signed:
            raise NotImplementedError("Fortran does not support unsigned integers")

        return f"INTEGER(kind={int(typ.size / 8)})"
    elif isinstance(typ, OP.Float):
        return f"REAL(kind={int(typ.size / 8)})"
    elif isinstance(typ, OP.Bool):
        return f"LOGICAL"
    else:
        assert False


lang.parseProgram = MethodType(parseProgram, lang)  # type: ignore
lang.parseKernel = MethodType(parseKernel, lang)  # type: ignore
lang.translateProgram = MethodType(translateProgram, lang)  # type: ignore
lang.formatType = MethodType(formatType, lang)  # type: ignore
