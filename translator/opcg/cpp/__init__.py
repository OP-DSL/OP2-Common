
# Standard library imports
from types import MethodType
import os

# Third party imports
from dotenv import load_dotenv
import clang.cindex

# Application imports
from cpp.parser import parseProgram, parseKernel
from cpp.translator.program import translateProgram
from language import Lang


# Load environment vairables set in .env and set libclang path
load_dotenv()
clang.cindex.Config.set_library_file(os.getenv('LIBCLANG_PATH'))


lang = Lang(
  name='c++', 
  com_delim='//', 
  source_exts=['cpp'], 
  include_ext='h',
  types=['float', 'double', 'int', 'uint', 'll', 'ull', 'bool'], 
)


lang.parseProgram = MethodType(parseProgram, lang) # type: ignore
lang.parseKernel = MethodType(parseKernel, lang) # type: ignore
lang.translateProgram = MethodType(translateProgram, lang) # type: ignore

