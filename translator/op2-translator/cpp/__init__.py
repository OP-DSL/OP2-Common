import os
from types import MethodType

import clang.cindex
from dotenv import load_dotenv

import op as OP
from cpp.parser import parseKernel, parseProgram
from cpp.translator.program import translateProgram
from language import Lang

# Load environment vairables set in .env and set libclang path
load_dotenv()
clang.cindex.Config.set_library_file(os.getenv("LIBCLANG_PATH"))


lang = Lang(
    name="c++",
    com_delim="//",
    source_exts=["cpp"],
    include_ext="h",
    types=["float", "double", "int", "uint", "ll", "ull", "bool"],
    kernel_dir=True,
)


def formatType(self, typ: OP.Type) -> str:
    int_types = {(True, 32): "int", (True, 64): "long long", (False, 32): "unsigned", (False, 64): "unsigned long long"}

    float_types = {32: "float", 64: "double"}

    if isinstance(typ, OP.Int):
        return int_types[(typ.signed, typ.size)]
    elif isinstance(typ, OP.Float):
        return float_types[typ.size]
    elif isinstance(typ, OP.Bool):
        return "bool"
    else:
        assert False


lang.parseProgram = MethodType(parseProgram, lang)  # type: ignore
lang.parseKernel = MethodType(parseKernel, lang)  # type: ignore
lang.translateProgram = MethodType(translateProgram, lang)  # type: ignore
lang.formatType = MethodType(formatType, lang)  # type: ignore
