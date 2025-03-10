# OP2 Code Generation

### Requirements

- Python >= 3.8
- libclang (for Debian based systems: `sudo apt-get install libclang-dev`)

### Getting Started

- Navigate to project directory `cd translator`
- Install python packages `pip3 install -r requirements.txt`
- Invoke the CLI with `python3 op2-translator`

### Brief Code Overview

- `__main__.py`
  - Entry point of op2-translator. Handles the arguments passed to op2-translator and calls the appropriate parser, validator and code generator.
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
