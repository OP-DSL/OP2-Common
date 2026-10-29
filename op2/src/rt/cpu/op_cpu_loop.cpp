#include <op2/rt/cpu/loop.h>

#include <op2/rt/cpu/strategy.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace op::rt::cpu {

// --- Per-thread globals ------------------------------------------------------------

namespace {

constexpr std::size_t line = 64;

bool per_thread(const op_arg& arg, const op_arg *args) {
    if (arg.argtype == OP_ARG_INFO)
        return args[arg.acc].opt != 0;
    return arg.opt != 0 && arg.argtype == OP_ARG_GBL &&
           (arg.acc == OP_INC || arg.acc == OP_MIN || arg.acc == OP_MAX ||
            arg.acc == OP_WORK);
}

} // namespace

template<typename T>
void Globals::combine(op_arg *args, int index) const {
    auto& arg = args[index];
    T *result = reinterpret_cast<T *>(arg.data);
    for (int thread = 0; thread < m_threads; ++thread) {
        const T *value = reinterpret_cast<const T *>(
            m_pointers[static_cast<std::size_t>(thread * m_nargs + index)]);
        for (int d = 0; d < arg.dim; ++d) {
            if (arg.acc == OP_INC) {
                result[d] += value[d];
                continue;
            }
            bool better = arg.acc == OP_MIN ? value[d] < result[d]
                                            : value[d] > result[d];
            if (!better)
                continue;
            result[d] = value[d];
            // The payloads travel with the thread whose value won.
            for (int i = 0; i < m_nargs; ++i)
                if (args[i].argtype == OP_ARG_INFO && args[i].acc == index &&
                    m_offsets[static_cast<std::size_t>(i)] != SIZE_MAX)
                    std::memcpy(args[i].data,
                                m_pointers[static_cast<std::size_t>(thread * m_nargs + i)],
                                static_cast<std::size_t>(args[i].size));
        }
    }
}

void Globals::prepare(op_arg *args, int nargs, int threads) {
    m_nargs = nargs;
    m_threads = threads;
    m_offsets.assign(static_cast<std::size_t>(nargs), SIZE_MAX);
    m_stride = 0;
    for (int i = 0; i < nargs; ++i)
        if (per_thread(args[i], args)) {
            m_offsets[static_cast<std::size_t>(i)] = m_stride;
            m_stride += (static_cast<std::size_t>(args[i].size) + line - 1) / line * line;
        }

    m_pointers.assign(static_cast<std::size_t>(threads * nargs), nullptr);
    if (m_stride == 0)
        return;

    std::size_t bytes = m_stride * static_cast<std::size_t>(threads) + line;
    if (m_capacity < bytes) {
        m_storage = std::make_unique<char[]>(bytes);
        m_capacity = bytes;
    }
    auto base = reinterpret_cast<std::uintptr_t>(m_storage.get());
    char *aligned = m_storage.get() + ((line - base % line) % line);

    for (int thread = 0; thread < threads; ++thread)
        for (int i = 0; i < nargs; ++i) {
            auto offset = m_offsets[static_cast<std::size_t>(i)];
            if (offset == SIZE_MAX)
                continue;
            char *copy = aligned + m_stride * static_cast<std::size_t>(thread) + offset;
            auto size = static_cast<std::size_t>(args[i].size);
            if (args[i].argtype == OP_ARG_GBL && args[i].acc == OP_INC)
                std::memset(copy, 0, size);
            else
                std::memcpy(copy, args[i].data, size);
            m_pointers[static_cast<std::size_t>(thread * nargs + i)] = copy;
        }
}

void Globals::combine(op_arg *args, const char *loop) const {
    for (int i = 0; i < m_nargs; ++i) {
        const auto& arg = args[i];
        if (m_offsets[static_cast<std::size_t>(i)] == SIZE_MAX ||
            arg.argtype != OP_ARG_GBL || arg.acc == OP_WORK)
            continue;
        auto type = op::rt::loop::element_type(arg.type);
        if (type == abi::ElementType::f64) {
            combine<double>(args, i);
        } else if (type == abi::ElementType::f32) {
            combine<float>(args, i);
        } else if (type == abi::ElementType::i32) {
            combine<int>(args, i);
        } else {
            std::fprintf(stderr, "error: unsupported reduction type '%s' (in %s)\n",
                         arg.type, loop);
            std::exit(1);
        }
    }
}

// --- The loop ---------------------------------------------------------------------

Loop::Loop(const abi::cpu::Loop& descriptor)
    : op::rt::loop::Loop{descriptor.name, descriptor.target, descriptor.variant,
                         descriptor.direct, false},
      m_max_threads{descriptor.max_threads}, m_globals{std::make_unique<Globals>()} {
    for (const auto& family : descriptor.families)
        m_ladder.add(family.name, [&](const op::rt::loop::Point& point) {
            return make_strategy(*this, family, point);
        });
}

Loop::~Loop() = default;

abi::cpu::Range Loop::range() const {
    abi::cpu::Range range;
    range.threads = m_threads;
    range.globals = m_globals->pointers();
    range.nargs = m_globals->nargs();
    return range;
}

abi::cpu::Chunk Loop::chunks() const {
    abi::cpu::Chunk chunks;
    chunks.threads = m_threads;
    chunks.globals = m_globals->pointers();
    chunks.nargs = m_globals->nargs();
    chunks.scratch = m_scratch_lines.data();
    return chunks;
}

void Loop::reserve_scratch(std::size_t bytes) {
    if (bytes <= m_scratch_capacity &&
        m_scratch.size() >= static_cast<std::size_t>(m_threads))
        return;
    m_scratch_capacity = std::max(bytes, m_scratch_capacity);
    m_scratch.clear();
    m_scratch_lines.clear();
    for (int thread = 0; thread < m_threads; ++thread) {
        m_scratch.push_back(std::make_unique<char[]>(m_scratch_capacity + line));
        char *base = m_scratch.back().get();
        auto address = reinterpret_cast<std::uintptr_t>(base);
        m_scratch_lines.push_back(base + ((line - address % line) % line));
    }
}

void Loop::begin_call(const op::rt::loop::Call&, const op::rt::tune::Autotuner::Decision&,
                      op_arg *args, int nargs) {
    m_threads = m_max_threads();
    m_globals->prepare(args, nargs, m_threads);
}

void Loop::combine_globals(op_arg *args, int) { m_globals->combine(args, m_name.c_str()); }

} // namespace op::rt::cpu

namespace op::rt::abi::cpu {

namespace {

// The loops made, which live as long as the program.
std::vector<std::unique_ptr<op::rt::cpu::Loop>>& loops() {
    static std::vector<std::unique_ptr<op::rt::cpu::Loop>> made;
    return made;
}

} // namespace

void invoke(Loop& loop, op_set set, op_arg *args, int nargs) {
    auto *runtime = static_cast<op::rt::cpu::Loop *>(loop.runtime);
    if (runtime == nullptr) {
        runtime = loops().emplace_back(std::make_unique<op::rt::cpu::Loop>(loop)).get();
        loop.runtime = runtime;
    }
    runtime->invoke(set, args, nargs);
}

} // namespace op::rt::abi::cpu
