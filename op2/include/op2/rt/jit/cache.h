#pragma once

// Each strategy's cache of built kernels, one per key (the hash of the
// values each is specialised on), and when each is built; the backend
// supplies the job that builds one.

#include <op2/rt/jit/options.h>
#include <op2/rt/loop/strategy.h>
#include <op_lib_core.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace op::rt::jit {

// Take one of limit build slots, shared by every cache; whether one was free.
bool take_build_slot(int limit);
void release_build_slot();

// Where a key's build stands.
enum class BuildState {
    unseen,  // never asked for, or evicted before its build started
    seen,    // asked for, its build not started
    building,
    ready,
    failed,
};

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
    struct Entry {
        // Times asked for while its build had not started.
        std::size_t calls = 0;
        // When it was last asked for, on m_clock.
        std::uint64_t used = 0;
        std::atomic<BuildState> state{BuildState::seen};
        std::thread thread;
        std::unique_ptr<Kernel> kernel;
    };

    // Past this many entries the least recently asked-for half of those not
    // started go, so values that change every call stay bounded.
    static constexpr std::size_t max_entries = 256;

    std::string m_name;
    const Options& m_options;
    int m_threads;
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
            if (entry.state.load(std::memory_order_relaxed) == BuildState::seen)
                idle.emplace_back(entry.used, key);
        auto middle = idle.begin() + static_cast<std::ptrdiff_t>(idle.size() / 2);
        std::nth_element(idle.begin(), middle, idle.end());
        for (auto item = idle.begin(); item != middle; ++item)
            m_entries.erase(item->second);
    }

    static void finish(Entry& entry, std::unique_ptr<Kernel> kernel) {
        entry.kernel = std::move(kernel);
        entry.state.store(entry.kernel != nullptr ? BuildState::ready
                                                  : BuildState::failed,
                          std::memory_order_release);
    }

public:
    // name, the kernel's, for messages; options, which must outlive it;
    // threads, the mechanism's default for OP_JIT_MAX_THREADS.
    Cache(std::string_view name, const Options& options, int threads)
        : m_name{name}, m_options{options}, m_threads{threads} {}
    Cache(const Cache&) = delete;
    // Joins the builds still running.
    ~Cache() {
        for (auto& [key, entry] : m_entries)
            if (entry.thread.joinable())
                entry.thread.join();
    }

    // The kernel for key once built.  Its build starts once key is asked for
    // OP_JIT_MIN_CALLS times, at once for Urgency::now and wait, while a
    // build slot is free; it runs here for wait or under OP_JIT_SEQ_COMPILE,
    // else on a thread.  Without the kernel: pending until it is built,
    // unavailable once its build failed.  make_job() is called here when a
    // build starts, and returns the job the build runs: it returns the
    // kernel, or none when the build failed.
    template<typename MakeJob>
    Acquired acquire(std::uint64_t key, loop::Urgency urgency, MakeJob&& make_job) {
        auto found = m_entries.find(key);
        if (found != m_entries.end() &&
            found->second.state.load(std::memory_order_acquire) == BuildState::ready) {
            // Its build has finished: free its thread.
            if (found->second.thread.joinable())
                found->second.thread.join();
            return {loop::Status::ready, found->second.kernel.get(), {}};
        }

        auto& entry = entry_for(key);
        ++entry.calls;
        const auto& options = m_options;
        bool here = urgency == loop::Urgency::wait || options.seq_compile;
        bool due = urgency != loop::Urgency::later ||
                   entry.calls >= static_cast<std::size_t>(options.min_calls);
        if (due && entry.state.load(std::memory_order_relaxed) == BuildState::seen &&
            (here || take_build_slot(options.max_threads.value_or(m_threads)))) {
            if (options.debug)
                op_printf("compiling %s for hash %lx\n", m_name.c_str(),
                            static_cast<unsigned long>(key));
            auto job = make_job();
            if (here) {
                finish(entry, job());
            } else {
                entry.state.store(BuildState::building, std::memory_order_relaxed);
                entry.thread = std::thread([&entry, job = std::move(job)]() mutable {
                    finish(entry, job());
                    release_build_slot();
                });
            }
        }
        // Wait for the build here, or free its thread once it has finished.
        if (entry.thread.joinable() &&
            (here || entry.state.load(std::memory_order_acquire) != BuildState::building))
            entry.thread.join();

        switch (entry.state.load(std::memory_order_acquire)) {
        case BuildState::ready:
            return {loop::Status::ready, entry.kernel.get(), {}};
        case BuildState::failed:
            return {loop::Status::unavailable, nullptr, "build_failed"};
        default:
            return {loop::Status::pending, nullptr, "compiling"};
        }
    }

    // Where key's build stands; no side effects.
    BuildState state(std::uint64_t key) const {
        auto found = m_entries.find(key);
        return found == m_entries.end()
                   ? BuildState::unseen
                   : found->second.state.load(std::memory_order_acquire);
    }

    loop::BuildCounts counts() const {
        loop::BuildCounts out;
        for (const auto& [key, entry] : m_entries)
            switch (entry.state.load(std::memory_order_acquire)) {
            case BuildState::ready:    ++out.built; break;
            case BuildState::failed:   ++out.failed; break;
            case BuildState::building: ++out.building; break;
            default:                   ++out.waiting; break;
            }
        return out;
    }
};

} // namespace op::rt::jit
