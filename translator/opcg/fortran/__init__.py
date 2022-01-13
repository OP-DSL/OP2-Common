from types import MethodType

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


lang.parseProgram = MethodType(parseProgram, lang)  # type: ignore
lang.parseKernel = MethodType(parseKernel, lang)  # type: ignore
lang.translateProgram = MethodType(translateProgram, lang)  # type: ignore
