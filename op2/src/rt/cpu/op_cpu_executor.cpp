#include <op2/rt/cpu/executor.h>

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

bool per_thread(const op_arg& arg, std::span<const op_arg> args) {
    if (arg.argtype == OP_ARG_INFO)
        return args[arg.acc].opt != 0;
    return arg.opt != 0 && arg.argtype == OP_ARG_GBL &&
           (arg.acc == OP_INC || arg.acc == OP_MIN || arg.acc == OP_MAX ||
            arg.acc == OP_WORK);
}

} // namespace

template<typename T>
void Globals::combine(std::span<op_arg> args, int index) const {
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
            // Each payload's component d travels with the thread whose value won it.
            for (int i = 0; i < m_nargs; ++i)
                if (args[i].argtype == OP_ARG_INFO && args[i].acc == index &&
                    m_offsets[static_cast<std::size_t>(i)] != SIZE_MAX) {
                    auto bytes = static_cast<std::size_t>(args[i].size / args[i].dim);
                    std::memcpy(args[i].data + d * bytes,
                                m_pointers[static_cast<std::size_t>(thread * m_nargs + i)] +
                                    d * bytes,
                                bytes);
                }
        }
    }
}

void Globals::prepare(std::span<op_arg> args, int threads, const char *loop) {
    int nargs = static_cast<int>(args.size());
    m_nargs = nargs;
    m_threads = threads;
    m_offsets.assign(static_cast<std::size_t>(nargs), SIZE_MAX);
    m_types.assign(static_cast<std::size_t>(nargs), std::nullopt);
    m_stride = 0;
    for (int i = 0; i < nargs; ++i) {
        const auto& arg = args[i];
        if (!per_thread(arg, args))
            continue;
        m_offsets[static_cast<std::size_t>(i)] = m_stride;
        m_stride += (static_cast<std::size_t>(arg.size) + line - 1) / line * line;
        if (arg.argtype != OP_ARG_GBL || arg.acc == OP_WORK)
            continue;
        auto type = op::rt::loop::element_type(arg.type);
        if (type != abi::ElementType::f64 && type != abi::ElementType::f32 &&
            type != abi::ElementType::i32) {
            std::fprintf(stderr, "error: unsupported reduction type '%s' (in %s)\n", arg.type,
                         loop);
            std::exit(1);
        }
        m_types[static_cast<std::size_t>(i)] = type;
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

void Globals::combine(std::span<op_arg> args) const {
    for (int i = 0; i < m_nargs; ++i) {
        auto type = m_types[static_cast<std::size_t>(i)];
        if (type == abi::ElementType::f64)
            combine<double>(args, i);
        else if (type == abi::ElementType::f32)
            combine<float>(args, i);
        else if (type == abi::ElementType::i32)
            combine<int>(args, i);
    }
}

// --- The executor -----------------------------------------------------------------

Executor::Executor(const abi::cpu::Loop& descriptor, op::rt::jit::ParamSet&)
    : m_descriptor{descriptor} {}

void Executor::add_strategies(op::rt::loop::Ladder& ladder) {
    ladder.add_families(m_descriptor.families, strategy_families(), *this, "OpenMP");
}

abi::cpu::Range Executor::range() const {
    abi::cpu::Range range;
    range.threads = m_threads;
    range.globals = m_globals.pointers();
    range.nargs = m_globals.nargs();
    return range;
}

abi::cpu::Chunk Executor::chunks() const {
    abi::cpu::Chunk chunks;
    chunks.threads = m_threads;
    chunks.globals = m_globals.pointers();
    chunks.nargs = m_globals.nargs();
    chunks.scratch = m_scratch_lines.data();
    return chunks;
}

void Executor::reserve_scratch(std::size_t bytes) {
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

void Executor::begin_call(const op::rt::loop::Call& call) {
    m_threads = m_descriptor.max_threads();
    m_globals.prepare(call.args, m_threads, name());
}

void Executor::combine_globals(const op::rt::loop::Call& call) {
    m_globals.combine(call.args);
}

} // namespace op::rt::cpu

namespace op::rt::abi::cpu {

void invoke(Loop& loop, op_set set, op_arg *args, int nargs) {
    op::rt::loop::runtime<op::rt::cpu::Executor>(loop).invoke(
        set, {args, static_cast<std::size_t>(nargs)});
}

} // namespace op::rt::abi::cpu
