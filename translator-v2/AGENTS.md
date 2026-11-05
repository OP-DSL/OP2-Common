# translator-v2: the source-to-source translator

Parses an application's `op_par_loop`s (C++ through libclang and pcpp,
Fortran through fparser and fpp), rewrites each loop's kernel for a target,
and renders the loop hosts and master files from `resources/templates`.

- Per-loop data is computed in Python, once (as `strategy.py` computes the
  chunk groups); per-element execution is C++ in the runtime's kernel
  skeletons (`op2/rt/kernel`); Jinja only wires the two.  Templates render
  under `StrictUndefined`.
- A mistake in the user's sources is a `TranslatorError` (`ParseError`,
  `OpError`; `diagnostics.py`) located in those sources (`Program.locate`,
  `sourceLocation`) and printed as `file:line:col`.  Anything else is a
  translator bug and keeps its traceback; a target falls back to its seq
  fallback on a `TranslatorError` only.
- One user kernel, and one JIT string of it, per loop host file: a kernel
  that takes an indirect `OP_INC` argument is a template over its increment
  hook (`OP.Loop.incHook`), in both languages.
- C++ kernel functions and types are known by their USR (`Entity.key`),
  never by their spelling.
- Kernel text is compiled by the host compiler, nvcc, NVRTC, hipcc and
  hipRTC alike: it is preprocessed with what C++20 fixes (`__cplusplus`),
  never one compiler's macros.
- Schemes stay compositional: share helpers between them, not a base layer
  over `translateKernel`.
- Check a change by its generated code: translate the apps and tests before
  and after, byte for byte where nothing should change, and run
  `ctest -L translator` and the corpus (`tests/functional/corpus*`).  A
  change to device code needs the register checks in
  `op2/include/op2/rt/kernel/AGENTS.md`.
- Hydra translates through the regex program translator with
  `--user-consts-module` and `OP_AUTO_SOA`; keep that path working
  (`gbl_regex_fortran` builds through it).
