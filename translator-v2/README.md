# OP2 Code Generation

The translator parses OP2 API calls in user source and generates per-target
kernel dispatch code. It's invoked automatically by the CMake app-build helper
`op2_add_app_variants()` and by the legacy GNU Make files under `makefiles/`.

### Dependencies

- Python >= 3.8
- Python packages: `jinja2`, `fparser`, `pcpp`, `sympy`, `libclang` (imports
  as `clang.cindex`). Install with `pip install -r requirements.txt`.

Python is a dependency you provide, the same way as MPI or HDF5 - CMake
finds a suitable interpreter (via `Python3_EXECUTABLE`), it doesn't create a
venv or run pip for you. If the picked interpreter doesn't have these
packages importable, configure reports `OP2_HAS_TRANSLATOR=FALSE` with a
message explaining what's missing.

### Building the translator

Nothing to build - the top-level OP2 `cmake -B build` step probes for a
Python 3.8+ interpreter with the packages above already installed, and
installs the translator's source alongside the runtime libs.

Once installed, the translator is available at
`<install-prefix>/bin/op2-translator` (on `$PATH`, invokes `python3` from
`PATH`) and as `${Python3_EXECUTABLE}
<install-prefix>/libexec/op2/translator/pkg` (invoked internally by
`op2_add_app_variants`).

### Manual invocation (development)

For running the translator directly against a single source file during
development:

```bash
python3 translator-v2/op2-translator \
    -t seq -o /tmp/out \
    -I translator-v2/op2-translator \
    -I op2/include \
    apps/c/airfoil/airfoil_plain/dp/airfoil.cpp
```

The `translator-v2/op2-translator.sh` wrapper does the same lookup automatically
and is what the legacy `makefiles/{c,f}_app.mk` invoke.

### Brief code overview

- `__main__.py` - CLI entry point. Handles arguments, drives parsing,
  validation, and code generation.
- `jinja.py` - Configures the Jinja templating engine (templates live under
  `../resources/templates/`).
- `language.py` - Abstract `Lang` base with concrete `Cpp` / `Fortran`
  subclasses in `cpp/` and `fortran/`.
- `op.py` - Data classes for OP2 primitives (`Set`, `Map`, `Dat`, `Arg*`,
  `Loop`, `AccessType`, `Type`).
- `scheme.py` - Binds a `Lang` to a `Target` (backend); each combination
  gets a registered scheme that produces host + kernel code.
- `store.py` - Parses source, populates `Application` / `Program`
  containers, tracks entity dependencies.
- `../resources/templates/` - Jinja templates for each language/target
  combination.
