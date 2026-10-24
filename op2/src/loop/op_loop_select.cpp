#include <loop/select.h>

#include <loop/jit.h>

namespace op::loop {

KernelKind choose_kernel(const Strategy& strategy, const Configuration& configuration) {
    const auto& policy = jit_policy();
    if (!policy.enabled())
        return KernelKind::offline;
    auto jit = strategy.estimate(configuration, KernelKind::jit);
    if (!jit.available)
        return KernelKind::offline;
    auto offline = strategy.estimate(configuration, KernelKind::offline);
    if (!offline.available || policy.force)
        return KernelKind::jit;
    if (jit.time && offline.time && *jit.time < *offline.time)
        return KernelKind::jit;
    return KernelKind::offline;
}

} // namespace op::loop
