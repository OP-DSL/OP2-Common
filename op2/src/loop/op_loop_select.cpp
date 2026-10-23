#include <loop/select.h>

#include <loop/jit.h>

namespace op::loop {

KernelKind choose_kernel(const Strategy& strategy, const Configuration& configuration) {
    (void)configuration;
    const auto& policy = jit_policy();
    auto kernels = strategy.kernels();
    if (!kernels.jit || !policy.enabled())
        return KernelKind::offline;
    if (!kernels.offline || policy.force ||
        kernels.offline_registers > policy.min_registers)
        return KernelKind::jit;
    return KernelKind::offline;
}

} // namespace op::loop
