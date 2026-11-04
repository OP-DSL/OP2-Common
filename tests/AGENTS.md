# Tests

- Test through the real interfaces: never add an accessor, virtual or API only
  for a test; rework the test instead.
- `unit/` host tests; `cuda/` GPU tests; `functional/` small programs
  translated and built for every variant (`op2_add_functional_tests`, over
  `op2_add_app_variants`);
  `translator/` translator runs and expected failures; `apps/` the
  applications (`OP2_ENABLE_APP_TESTS`).  `rt_layering` runs
  `scripts/check_rt_layering.sh`.
- `loop_parts.h` builds a loop's parts from a descriptor and the run's
  context.  A test that changes variables reads them again: `op::rt::init()`
  replaces the context, destroying its loops, then `init_cpu`/`init_gpu` or
  the test's own family declarations add the rest; an option struct can be
  read on its own (`jit::read_options()`).
