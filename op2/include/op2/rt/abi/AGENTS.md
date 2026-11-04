# abi: the contract with the generated code

Plain types, constants and entry points the translator's templates and the
library both compile against: loop descriptors and their parameter rows,
families, launch records, element types.  Nothing here has behaviour or
knows a library internal.

- A change here changes the ABI: change the C++ and Fortran templates with
  it, regenerate, and check the generated code builds and its kernels'
  registers do not rise.
- Launch records are X-macro field lists (`OP_ABI_GPU_*_FIELDS`): a field is
  added there, once.
- `abi/gpu` and `abi/cpu` are each backend's part; the top-level headers are
  neutral.
- Descriptors are static in the generated code and outlive every loop.
