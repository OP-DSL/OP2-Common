# OP2 Code Generation

## C
TODO
## FORTRAN
TODO
## OP-CG

### Getting Started

- Navigate to project directory `op-cg`
- Install python packages `pip3 install -r requirements.txt`
- Invoke the CLI with `python3 opcg`

### Requirements

- Python>=3.8
- Clang (including pyclang : sudo apt-get install libclang-dev)
- Java

### Brief Code Overview

- `__main__.py`
  - Entry point of OP-CG. Handles the arguments passed to OP-CG and calls the appropriate parser, validator and code generator.
- `jinja.py`
  - Configures the Jinja templating engine.
- `language.py`
  - Contains the `Lang` class used to hold information for the programming langauge that code is being generated for (currently C/C++ or FORTRAN).
- `op.py`
  - Holds classes representing OP2 sets, maps, dats, args and loops. The parser creates instances of these classes with the information needed by the code generator.
- `optimisation.py`
  - Contains the `Opt` class that is used to store information about the various optimisations that code can be generated for (sequential, OpenMP, CUDA).
- `scheme.py`
  - Contains the `Scheme` class that is used to generate code for each combination of language and optimisation. Each scheme is registered at the bottom of this file.
- `store.py`
  - Contains classes that are used during the parsing of the source files and store information about the overall program.
- `resources/templates`
  - The subdirectories within contain the Jinja templates used during the code generation.
