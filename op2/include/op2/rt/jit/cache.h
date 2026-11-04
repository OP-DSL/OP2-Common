#pragma once

// Each strategy's cache of built kernels, one per key (the hash of the
// values each is specialised on), and when each is built; the backend
// supplies the job that builds one, and the run's build pool runs it.

#include <op2/rt/jit/options.h>
#include <op2/rt/jit/pool.h>
#include <op2/rt/loop/strategy.h>
#include <op_lib_core.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace op::rt::jit {

// A strategy's built kernels, one per key (the hash of the values each is
// specialised on), and when each is built.
template<typename Kernel>
class Cache {
public:
    struct Acquired {
        loop::Status status = loop::Status::unavailable;
        // The built kernel; none to run the offline one.
        Kernel *kernel = nullptr;
        // Why it is not ready: "compiling" or "build_failed".
        std::string_view reason;
    };

private:
    // Where a key's build stands.
    enum class State {
        seen,  // asked for, its build not started
        building,
        ready,
        failed,
    };

    // A build on the pool: its kernel, none when it failed, once its state
    // leaves building.
    struct Build {
        std::unique_ptr<Kernel> kernel;
        std::atomic<State> state{State::building};
    };

    struct Entry {
        // Times asked for while its build had not started.
        std::size_t calls = 0;
        // When it was last asked for, on m_clock.
        std::uint64_t used = 0;
        State state = State::seen;
        // Its build while on the pool.
        std::shared_ptr<Build> build;
        std::unique_ptr<Kernel> kernel;
    };

    // Past this many entries the least recently asked-for half of those not
    // started go, so values that change every call stay bounded.
    static constexpr std::size_t max_entries = 256;

    std::string m_name;
    const Options& m_options;
    BuildPool& m_pool;
    std::unordered_map<std::uint64_t, Entry> m_entries;
    std::uint64_t m_clock = 0;

    // The entry for key, marked as asked for now.
    Entry& entry_for(std::uint64_t key) {
        auto [found, inserted] = m_entries.try_emplace(key);
        found->second.used = ++m_clock;
        if (inserted && m_entries.size() > max_entries)
            evict();
        return found->second;
    }

    // Drop the least recently asked-for half of the entries not started; the
    // newest is never among them.
    void evict() {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> idle;
        for (const auto& [key, entry] : m_entries)
            if (entry.state == State::seen)
                idle.emplace_back(entry.used, key);
        auto middle = idle.begin() + static_cast<std::ptrdiff_t>(idle.size() / 2);
        std::nth_element(idle.begin(), middle, idle.end());
        for (auto item = idle.begin(); item != middle; ++item)
            m_entries.erase(item->second);
    }

    static void finish(Entry& entry, std::unique_ptr<Kernel> kernel) {
        entry.kernel = std::move(kernel);
        entry.state = entry.kernel != nullptr ? State::ready : State::failed;
    }

    // Take entry's build from the pool once it is published, waiting for it
    // under wait.
    static void collect(Entry& entry, bool wait) {
        auto& build = *entry.build;
        if (wait)
            build.state.wait(State::building, std::memory_order_acquire);
        if (build.state.load(std::memory_order_acquire) == State::building)
            return;
        finish(entry, std::move(build.kernel));
        entry.build.reset();
    }

public:
    // name, the kernel's, for messages; options and pool, which must outlive
    // it.
    Cache(std::string_view name, const Options& options, BuildPool& pool)
        : m_name{name}, m_options{options}, m_pool{pool} {}
    Cache(const Cache&) = delete;

    // The kernel for key once built.  Its build starts once key is asked for
    // OP_JIT_MIN_CALLS times, at once for Urgency::now and wait: here for
    // wait or under OP_JIT_SEQ_COMPILE, else on the pool while it has room.
    // Without the kernel: pending until it is built, unavailable once its
    // build failed.  make_job() is called when a build starts, and returns
    // the job the build runs: it returns the kernel, or none when the build
    // failed.
    template<typename MakeJob>
    Acquired acquire(std::uint64_t key, loop::Urgency urgency, MakeJob&& make_job) {
        auto found = m_entries.find(key);
        if (found != m_entries.end() && found->second.state == State::ready)
            return {loop::Status::ready, found->second.kernel.get(), {}};

        auto& entry = entry_for(key);
        ++entry.calls;
        bool here = urgency == loop::Urgency::wait || m_options.seq_compile;
        bool due = urgency != loop::Urgency::later ||
                   entry.calls >= static_cast<std::size_t>(m_options.min_calls);
        if (due && entry.state == State::seen) {
            auto start = [&] {
                if (m_options.debug)
                    op_printf("compiling %s for hash %lx\n", m_name.c_str(),
                              static_cast<unsigned long>(key));
                return make_job();
            };
            if (here) {
                finish(entry, start()());
            } else if (m_pool.try_start([&] {
                           auto build = std::make_shared<Build>();
                           entry.build = build;
                           return BuildPool::Task{[build, job = start()]() mutable {
                               build->kernel = job();
                               return std::function<void()>{[build] {
                                   build->state.store(build->kernel != nullptr
                                                          ? State::ready
                                                          : State::failed,
                                                      std::memory_order_release);
                                   build->state.notify_all();
                               }};
                           }};
                       })) {
                entry.state = State::building;
            }
        }
        // Wait for the build here, or take it once it is published.
        if (entry.state == State::building)
            collect(entry, here);

        switch (entry.state) {
        case State::ready:
            return {loop::Status::ready, entry.kernel.get(), {}};
        case State::failed:
            return {loop::Status::unavailable, nullptr, "build_failed"};
        default:
            return {loop::Status::pending, nullptr, "compiling"};
        }
    }

    loop::BuildCounts counts() const {
        loop::BuildCounts out;
        for (const auto& [key, entry] : m_entries) {
            auto state = entry.state == State::building
                             ? entry.build->state.load(std::memory_order_acquire)
                             : entry.state;
            switch (state) {
            case State::ready:    ++out.built; break;
            case State::failed:   ++out.failed; break;
            case State::building: ++out.building; break;
            default:              ++out.waiting; break;
            }
        }
        return out;
    }
};

} // namespace op::rt::jit
