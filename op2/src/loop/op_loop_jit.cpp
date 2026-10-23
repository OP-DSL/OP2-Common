#include <loop/jit.h>

#include <op_rt_support.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace op::loop {

// --- Policy -------------------------------------------------------------------

namespace {

JitPolicy run_policy;
std::atomic_int builds_in_flight = 0;

// A comma-separated list's items.
std::vector<std::string> items(const char *variable) {
    std::vector<std::string> out;
    const char *text = std::getenv(variable);
    std::string_view rest = text != nullptr ? text : "";
    while (!rest.empty()) {
        auto comma = rest.find(',');
        auto item = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{}
                                               : rest.substr(comma + 1);
        if (!item.empty())
            out.emplace_back(item);
    }
    return out;
}

} // namespace

const JitPolicy& jit_policy() { return run_policy; }

void init_jit_policy() {
    run_policy = {};
    if (std::getenv("OP_JIT_ENABLE") != nullptr)
        op_printf("warning: OP_JIT_ENABLE is no longer read; OP_JIT_LEVEL=0 turns "
                  "the JIT off\n");
    int level = run_policy.level;
    if (env::parse_number("OP_JIT_LEVEL", level)) {
        if (level <= 2)
            run_policy.level = level;
        else
            op_printf("warning: ignoring OP_JIT_LEVEL=%d, expected 0, 1 or 2\n", level);
    }
    if (!run_policy.enabled())
        op_printf("Disabling JIT compilation\n");
    env::parse_bool("OP_JIT_DEBUG", run_policy.debug);
    if (run_policy.debug)
        op_printf("Enabling JIT debug\n");
    env::parse_bool("OP_JIT_SEQ_COMPILE", run_policy.seq_compile);
    env::parse_bool("OP_JIT_FORCE", run_policy.force);
    if (run_policy.force)
        op_printf("Forcing JIT compilation regardless of register count\n");
    env::parse_number("OP_JIT_MIN_CALLS", run_policy.min_calls);
    int threads = 0;
    if (env::parse_number("OP_JIT_MAX_THREADS", threads))
        run_policy.max_threads = threads;
    env::parse_number("OP_JIT_VARYING", run_policy.varying);
    env::parse_number("OP_JIT_MAX_GBL_BYTES", run_policy.max_gbl_bytes);
    env::parse_number("OP_JIT_MAX_CONST_BYTES", run_policy.max_const_bytes);
    run_policy.runtime_params = items("OP_JIT_RUNTIME_PARAMS");
    run_policy.constant_params = items("OP_JIT_CONSTANT_PARAMS");
}

bool take_build_slot(int limit) {
    int count = builds_in_flight.load();
    while (count < limit)
        if (builds_in_flight.compare_exchange_weak(count, count + 1))
            return true;
    return false;
}

void release_build_slot() { --builds_in_flight; }

// --- Parameters ---------------------------------------------------------------

void Param::settle(std::string_view loop, bool runtime, const JitPolicy& policy) {
    auto named = [&](const std::vector<std::string>& names) {
        return std::any_of(names.begin(), names.end(), [&](std::string_view item) {
            if (item.size() > loop.size() && item.substr(0, loop.size()) == loop &&
                item[loop.size()] == ':')
                item.remove_prefix(loop.size() + 1);
            return item == m_name;
        });
    };

    m_fixed = true;
    m_initial = ParamMode::baked;
    if (named(policy.constant_params)) {
    } else if (named(policy.runtime_params)) {
        if (runtime)
            m_initial = ParamMode::runtime;
        else
            op_printf("warning: %s's kernels cannot have %s at run time, from "
                      "OP_JIT_RUNTIME_PARAMS\n",
                      std::string{loop}.c_str(), m_name.c_str());
    } else if (runtime && m_class == ParamClass::constant) {
        if (bytes() > static_cast<std::size_t>(policy.max_const_bytes))
            m_initial = ParamMode::runtime;
        else
            m_fixed = false;
    } else if (runtime && m_class == ParamClass::global) {
        // An array_arg's size is each call's, so its cap is checked per call.
        m_initial = ParamMode::runtime;
        m_fixed = policy.level < 2 ||
                  (m_source == ParamSource::scalar_arg &&
                   bytes() > static_cast<std::size_t>(policy.max_gbl_bytes));
    }
    m_mode = m_initial;
}

void Param::update(op_arg *args, int nargs) {
    switch (m_source) {
    case ParamSource::scalar_arg:
        assert(m_index >= 0 && m_index < nargs);
        // An inactive optional global's kernel never reads it.
        if (args[m_index].opt == 0)
            return;
        assert(args[m_index].data != nullptr);
        std::memcpy(m_data, args[m_index].data, m_elem_size);
        return;
    case ParamSource::array_arg:
        assert(m_index >= 0 && m_index < nargs);
        if (args[m_index].opt == 0) {
            m_data = nullptr;
            m_elems = 0;
            return;
        }
        m_data = args[m_index].data;
        m_elems = static_cast<std::size_t>(args[m_index].dim);
        return;
    case ParamSource::dat_stride: {
        assert(m_index >= 0 && m_index < nargs);
        assert(m_elem_size == sizeof(int) && m_elems == 1);
        int size = getSetSizeFromOpArg(&args[m_index]);
        *static_cast<int *>(m_data) = (size + 31) & ~31;
        return;
    }
    case ParamSource::dat_dim:
        assert(m_index >= 0 && m_index < nargs);
        assert(m_elem_size == sizeof(int) && m_elems == 1);
        *static_cast<int *>(m_data) = args[m_index].dim;
        return;
    default:
        (void)nargs;
        return;
    }
}

void ParamSet::update(op_arg *args, int nargs) {
    ++m_calls;
    for (auto& param : m_params)
        param.update(args, nargs);
}

std::vector<ParamSet::History> *ParamSet::histories(std::uint64_t configuration) {
    if (m_history != nullptr && configuration == m_history_key)
        return m_history;
    auto found = m_histories.find(configuration);
    if (found == m_histories.end()) {
        if (m_histories.size() >= max_histories)
            return nullptr;
        found = m_histories.emplace(configuration, std::vector<History>(m_params.size()))
                    .first;
    }
    m_history_key = configuration;
    m_history = &found->second;
    return m_history;
}

void ParamSet::observe(const Configuration& configuration) {
    const auto& run = jit_policy();
    for (auto& param : m_params) {
        if (param.source() == ParamSource::strategy)
            continue;
        if (param.sized_out(run)) {
            param.set_mode(ParamMode::runtime);
            continue;
        }
        param.hash();
        if (param.fixed())
            param.set_mode(param.initial());
    }
    if (!m_moving)
        return;

    auto *history = histories(configuration.key());
    for (std::size_t i = 0; i < m_params.size(); ++i) {
        auto& param = m_params[i];
        if (param.fixed() || param.sized_out(run))
            continue;
        if (history == nullptr) {
            param.set_mode(param.initial());
            continue;
        }
        auto& entry = (*history)[i];
        if (entry.observe(param, run) && run.debug)
            std::printf("jit: %s %s %s for %s\n", m_loop.c_str(),
                        entry.mode == ParamMode::baked ? "bakes in" : "moves to run time",
                        param.name().c_str(), configuration.describe().c_str());
        param.set_mode(entry.mode);
    }
}

bool ParamSet::History::observe(const Param& param, const JitPolicy& run) {
    std::uint64_t hash = param.last_hash();
    if (!seen) {
        seen = true;
        last = hash;
        candidate = hash;
        hits = 1;
        total = 1;
        mode = param.initial();
        return false;
    }

    changes = (changes << 1) | (hash != last ? 1u : 0u);
    last = hash;
    int changed = std::popcount(changes);
    if (mode == ParamMode::baked) {
        if (changed <= run.varying)
            return false;
        mode = ParamMode::runtime;
        demoted = true;
        return true;
    }

    if (demoted || promoted || param.param_class() != ParamClass::global)
        return false;
    if (hash == candidate)
        ++hits;
    else if (2 * hits <= total) {
        candidate = hash;
        hits = 1;
        total = 0;
    }
    ++total;
    if (total < 32 || 5 * hits < 4 * total || changed > run.varying)
        return false;
    mode = ParamMode::baked;
    promoted = true;
    changes = 0;
    return true;
}

const char *cxx_type(ParamType type) {
    switch (type) {
    case ParamType::i32:     return "int";
    case ParamType::i64:     return "int64_t";
    case ParamType::f32:     return "float";
    case ParamType::f64:     return "double";
    case ParamType::logical: return "bool";
    }
    __builtin_unreachable();
}

namespace {

// A finite value in hexfloat, exactly; a NaN or infinity, which hexfloat
// writes as a word no compiler reads, as its bits.
template<typename Bits, typename T>
void format_float(std::ostream& os, T value, const char *type, const char *suffix) {
    if (std::isfinite(value)) {
        os << std::hexfloat << value;
        return;
    }
    Bits bits;
    std::memcpy(&bits, &value, sizeof(bits));
    os << "__builtin_bit_cast(" << type << ", 0x" << std::hex << bits << std::dec
       << suffix << ")";
}

std::string format_value(const Param& param) {
    std::ostringstream os;
    if (param.array())
        os << "{ ";
    std::size_t elem_size = param.bytes() / param.elems();
    for (std::size_t i = 0; i < param.elems(); ++i) {
        const char *elem = static_cast<const char *>(param.data()) + elem_size * i;
        switch (param.type()) {
        case ParamType::i32:     os << *((const int *)elem); break;
        case ParamType::i64:     os << *((const int64_t *)elem); break;
        case ParamType::f32:
            format_float<uint32_t>(os, *((const float *)elem), "float", "u");
            break;
        case ParamType::f64:
            format_float<uint64_t>(os, *((const double *)elem), "double", "ull");
            break;
        case ParamType::logical: os << std::boolalpha << *((const bool *)elem); break;
        }
        if (param.array() && i < param.elems() - 1)
            os << ", ";
    }
    if (param.array())
        os << " }";
    return os.str();
}

} // namespace

std::string format_cxx(const ParamSet& params) {
    std::string out;
    for (std::size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        if (param.mode() != ParamMode::baked)
            continue;
        out += "static constexpr ";
        out += cxx_type(param.type());
        out += " " + param.name();
        if (param.array())
            out += "[" + std::to_string(param.elems()) + "]";
        out += " = " + format_value(param) + ";\n";
    }
    return out;
}

} // namespace op::loop
