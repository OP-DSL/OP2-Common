#pragma once

// Plans shared by every strategy, of any loop, that would build the same
// one.  A planner keys a plan by what it plans from (PlanKey), a backend
// keeps the live plans of each kind (SharedPlans), and each strategy holds
// the plans of the configurations it runs (PlanSlots): a plan lives while a
// strategy holds it.

#include <extern/rapidhash.h>
#include <loop/strategy.h>
#include <plan/staging.h>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <unordered_map>

namespace op::plan {

// What a planner plans from, hashed: the planner's name, the call's shape
// (loop::Call::key, which names sets and maps by their indices, unique in a
// run), each active argument's bytes, which the shape leaves out and staging
// lays regions out by, and the planner's other inputs, as it adds them.
class PlanKey {
private:
    std::uint64_t m_hash;

    void mix(const void *data, std::size_t bytes) {
        m_hash = rapidhash_withSeed(data, bytes, m_hash);
    }

public:
    PlanKey(std::string_view planner, const loop::Call& call) : m_hash{call.key()} {
        mix(planner.data(), planner.size());
        for (const auto& arg : call.args) {
            int bytes = arg.opt != 0 ? arg.size : 0;
            mix(&bytes, sizeof(bytes));
        }
    }

    template<std::integral T>
    PlanKey& add(T value) {
        std::int64_t word = value;
        mix(&word, sizeof(word));
        return *this;
    }

    PlanKey& add(double value) {
        std::int64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return add(bits);
    }

    PlanKey& add(const ArgGroups& groups) {
        add(groups.groups);
        for (const auto& group : groups.args)
            add(std::int64_t{group.arg} << 32 | static_cast<std::uint32_t>(group.group));
        return *this;
    }

    std::uint64_t value() const { return m_hash; }
};

// The live plans of one kind, by key; it holds none of them.
template<typename Plan>
class SharedPlans {
private:
    std::unordered_map<std::uint64_t, std::weak_ptr<const Plan>> m_plans;
    // Keys whose plans have gone are swept out once the map reaches this.
    std::size_t m_sweep_at = 64;

public:
    // The live plan under key, else the one make() returns, a
    // std::shared_ptr to it.
    template<typename Make>
    std::shared_ptr<const Plan> get(const PlanKey& key, Make&& make) {
        auto& slot = m_plans[key.value()];
        if (auto plan = slot.lock())
            return plan;
        std::shared_ptr<const Plan> plan = make();
        slot = plan;
        if (m_plans.size() >= m_sweep_at) {
            std::erase_if(m_plans, [](const auto& item) { return item.second.expired(); });
            m_sweep_at = std::max<std::size_t>(64, 2 * m_plans.size());
        }
        return plan;
    }

    // The plans some strategy still holds.
    std::size_t live() const {
        std::size_t count = 0;
        for (const auto& [key, plan] : m_plans)
            count += !plan.expired();
        return count;
    }
};

// A strategy's plans by configuration (loop::Call::key): a lookup per call,
// and the loop's shared plans when a configuration is new.
template<typename Plan>
class PlanSlots {
private:
    std::unordered_map<std::uint64_t, std::shared_ptr<const Plan>> m_slots;

public:
    // The configuration's plan, else the one make() returns, a
    // std::shared_ptr to it.
    template<typename Make>
    const Plan& get(const loop::Call& call, Make&& make) {
        auto key = call.key();
        auto found = m_slots.find(key);
        if (found == m_slots.end())
            found = m_slots.emplace(key, make()).first;
        return *found->second;
    }

    // Drop a configuration's plan; it goes once no strategy holds it.
    void forget(std::uint64_t call) { m_slots.erase(call); }
    void clear() { m_slots.clear(); }
    std::size_t size() const { return m_slots.size(); }
};

} // namespace op::plan
