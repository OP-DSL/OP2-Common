# kernel: compiled with the generated kernels

Header-only skeletons (range and chunk, and on the GPU the wavefront's) with
the hooks the generated code fills, the increment hooks and the prelude.
Compiled only into op2_kernels and into JIT builds (`gpu/jit_embed.h`), never
into a library.

- Compiles under nvcc, hipcc, NVRTC and hipRTC, and the host compiler for
  `cpu/`: what the JIT embeds uses no host-only headers.
- Register pressure first: never raise a kernel's registers or local memory
  for tidiness; compare per-kernel registers on sm_90 and gfx942, offline and
  JIT, for every kernel a change reaches.  Device encodings may keep counts
  to shorten live ranges.
- Includes only `abi/` and `kernel/` from op2/rt; device headers only under
  device compilers.

## Register traps found so far
- Measure registers on whole-program compiles (`-cubin`, as the build and
  NVRTC do), never `--device-c`.
- Kernel parameters are separate parameters (the X-macro field lists), not a
  struct by value; `__grid_constant__` is worse still.
- HIP: read the block size through `__builtin_amdgcn_workgroup_size_x()`
  (`gpu/device.h`), as unsigned: `blockDim.x` in a callee, or an int stride,
  costs VGPRs.
- HIP: the skeleton derives per-chunk pointers from its own parameters and
  passes them to the hooks; a hook reading the record's pointer arrays gets
  generic pointers.
- A value held across the user kernel costs registers: keep counts in an
  encoding rather than derive them from offsets read early.  Where a pointer
  is computed (chunk start or flush) moves sm_90 counts: measure both.
