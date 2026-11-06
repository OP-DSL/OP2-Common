#pragma once

// The threads specialised kernels build on, shared by every strategy's
// cache; the run's context holds it (op2/rt/context.h), and its backend sets
// the limit.  A build starts only while fewer than the limit are under way,
// so one that cannot start waits for a later call, and its result is
// published once its place is free, so a call that sees it can start
// another.  Destroying the pool waits for the builds under way.  Defined in
// src/rt/jit/op_jit_pool.cpp.

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace op::rt::jit {

class BuildPool {
public:
    // A build, run on a pool thread, which returns how to publish its result.
    using Task = std::function<std::function<void()>()>;

private:
    std::mutex m_mutex;
    std::condition_variable m_wake;
    std::deque<Task> m_tasks;
    std::vector<std::thread> m_threads;
    // At most this many builds under way; one until the backend sets it.
    int m_limit = 1;
    // Builds started and not yet finished, queued ones included; never more
    // than the threads.
    int m_under_way = 0;
    bool m_stopping = false;

    void work();

public:
    BuildPool() = default;
    BuildPool(const BuildPool&) = delete;
    ~BuildPool();

    void set_limit(int limit);

    // Start make()'s task while fewer than the limit are under way, on a
    // free thread or a new one; whether it started.  make is called only
    // when it does.
    template<typename Make>
    bool try_start(Make&& make) {
        std::lock_guard lock{m_mutex};
        if (m_under_way >= m_limit)
            return false;
        m_tasks.push_back(make());
        if (m_threads.size() < static_cast<std::size_t>(++m_under_way))
            m_threads.emplace_back([this] { work(); });
        else
            m_wake.notify_one();
        return true;
    }
};

} // namespace op::rt::jit
