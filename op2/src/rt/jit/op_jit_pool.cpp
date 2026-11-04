#include <op2/rt/jit/pool.h>

#include <utility>

namespace op::rt::jit {

BuildPool::~BuildPool() {
    {
        std::lock_guard lock{m_mutex};
        m_stopping = true;
    }
    m_wake.notify_all();
    for (auto& thread : m_threads)
        thread.join();
}

void BuildPool::set_limit(int limit) {
    std::lock_guard lock{m_mutex};
    m_limit = limit;
}

void BuildPool::work() {
    std::unique_lock lock{m_mutex};
    while (true) {
        m_wake.wait(lock, [&] { return m_stopping || !m_tasks.empty(); });
        if (m_tasks.empty())
            return;
        auto task = std::move(m_tasks.front());
        m_tasks.pop_front();
        lock.unlock();
        auto publish = task();
        lock.lock();
        --m_under_way;
        lock.unlock();
        publish();
        lock.lock();
    }
}

} // namespace op::rt::jit
