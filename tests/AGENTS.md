# Tests

- Test through the real interfaces: never add an accessor, virtual or API only
  for a test; rework the test instead.
- `unit/` host tests; `cuda/` GPU tests; `functional/` small programs
  translated and built for every variant (`op2_add_app_variants`);
  `translator/` translator runs and expected failures; `apps/` the
  applications (`OP2_ENABLE_APP_TESTS`).  `rt_layering` runs
  `scripts/check_rt_layering.sh`.
- `loop_parts.h` builds a loop's parts from a descriptor and the run's
  context.  A test that changes variables re-runs `op::rt::init()`, which
  replaces the context and destroys its loops.
