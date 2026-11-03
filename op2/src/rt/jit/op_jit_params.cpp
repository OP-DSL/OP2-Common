#include <op2/rt/jit/params.h>

#include <op2/env.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace op::rt::jit {

void Param::settle(std::string_view loop, bool runtime, const Options& options) {
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
    if (named(options.constant_params)) {
    } else if (named(options.runtime_params)) {
        if (runtime)
            m_initial = ParamMode::runtime;
        else
            env::warn("warning: " + std::string{loop} + "'s kernels cannot have " + m_name +
                      " at run time, from OP_JIT_RUNTIME_PARAMS\n");
    } else if (runtime && m_source == abi::ParamSource::external) {
        if (bytes() > static_cast<std::size_t>(options.max_const_bytes))
            m_initial = ParamMode::runtime;
        else
            m_fixed = false;
    } else if (runtime && global()) {
        // An array_arg's size is each call's, so its cap is checked per call.
        m_initial = ParamMode::runtime;
        m_fixed = options.level < 2 ||
                  (m_source == abi::ParamSource::scalar_arg &&
                   bytes() > static_cast<std::size_t>(options.max_gbl_bytes));
    }
    m_mode = m_initial;
}

void Param::update(std::span<const op_arg> args, DatStride dat_stride) {
    switch (m_source) {
    case abi::ParamSource::scalar_arg:
        assert(m_index >= 0 && static_cast<std::size_t>(m_index) < args.size());
        // An inactive optional global's kernel never reads it.
        if (args[m_index].opt == 0)
            return;
        assert(args[m_index].data != nullptr);
        std::memcpy(m_data, args[m_index].data, m_elem_size);
        return;
    case abi::ParamSource::array_arg:
        assert(m_index >= 0 && static_cast<std::size_t>(m_index) < args.size());
        if (args[m_index].opt == 0) {
            m_data = nullptr;
            m_elems = 0;
            return;
        }
        m_data = args[m_index].data;
        m_elems = static_cast<std::size_t>(args[m_index].dim);
        return;
    case abi::ParamSource::dat_stride:
        assert(m_index >= 0 && static_cast<std::size_t>(m_index) < args.size());
        assert(m_elem_size == sizeof(int) && m_elems == 1 && dat_stride != nullptr);
        *static_cast<int *>(m_data) = dat_stride(args[m_index]);
        return;
    case abi::ParamSource::dat_dim:
        assert(m_index >= 0 && static_cast<std::size_t>(m_index) < args.size());
        assert(m_elem_size == sizeof(int) && m_elems == 1);
        *static_cast<int *>(m_data) = args[m_index].dim;
        return;
    default:
        return;
    }
}

void ParamSet::update(std::span<const op_arg> args) {
    ++m_calls;
    for (auto& param : m_params)
        param.update(args, m_dat_stride);
}

ParamSet::Histories *ParamSet::histories(const loop::Call& call) {
    auto key = call.key();
    if (m_history != nullptr && key == m_history_key)
        return m_history;
    auto found = m_histories.find(key);
    if (found == m_histories.end()) {
        if (m_histories.size() >= max_histories)
            return nullptr;
        found = m_histories
                    .emplace(key, Histories{std::vector<History>(m_params.size()), 0,
                                            call.describe()})
                    .first;
    }
    m_history_key = key;
    m_history = &found->second;
    return m_history;
}

bool ParamSet::settled(const loop::Call& call) const {
    if (!m_moving || !jit::options().enabled())
        return true;
    auto found = m_histories.find(call.key());
    // Past max_histories a configuration keeps the initial modes.
    if (found == m_histories.end())
        return m_histories.size() >= max_histories;
    return found->second.calls >= statistics_window;
}

std::vector<ParamSet::ConfigurationModes> ParamSet::configurations() const {
    std::vector<ConfigurationModes> out;
    for (const auto& [key, histories] : m_histories) {
        ConfigurationModes configuration{histories.description, histories.calls, {}};
        for (std::size_t i = 0; i < m_params.size(); ++i) {
            const auto& history = histories.params[i];
            if (!m_params[i].fixed() && history.seen)
                configuration.params.push_back(
                    {&m_params[i], history.mode, history.promoted || history.demoted});
        }
        out.push_back(std::move(configuration));
    }
    return out;
}

void ParamSet::observe(const loop::Call& call) {
    const auto& options = jit::options();
    auto *history = m_moving ? histories(call) : nullptr;
    if (history != nullptr)
        ++history->calls;
    for (std::size_t i = 0; i < m_params.size(); ++i) {
        auto& param = m_params[i];
        if (param.source() == abi::ParamSource::strategy)
            continue;
        if (param.sized_out(options)) {
            param.set_mode(ParamMode::runtime);
            continue;
        }
        param.hash();
        if (param.fixed() || history == nullptr) {
            param.set_mode(param.initial());
            continue;
        }
        auto& entry = history->params[i];
        if (entry.observe(param, options) && options.debug)
            op_printf("jit: %s %s %s for %s\n", m_loop.c_str(),
                        entry.mode == ParamMode::baked ? "bakes in" : "moves to run time",
                        param.name().c_str(), call.describe().c_str());
        param.set_mode(entry.mode);
    }
}

bool ParamSet::History::observe(const Param& param, const Options& options) {
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

    unsigned change = hash != last ? 1u : 0u;
    changed += static_cast<int>(change) - static_cast<int>(changes >> 31);
    changes = (changes << 1) | change;
    last = hash;
    if (mode == ParamMode::baked) {
        if (changed <= options.varying)
            return false;
        mode = ParamMode::runtime;
        demoted = true;
        return true;
    }

    if (demoted || promoted || !param.global())
        return false;
    if (hash == candidate)
        ++hits;
    else if (2 * hits <= total) {
        candidate = hash;
        hits = 1;
        total = 0;
    }
    ++total;
    if (total < statistics_window || 5 * hits < 4 * total || changed > options.varying)
        return false;
    mode = ParamMode::baked;
    promoted = true;
    changes = 0;
    changed = 0;
    return true;
}

const char *cxx_type(abi::ElementType type) {
    switch (type) {
    case abi::ElementType::i32:     return "int";
    case abi::ElementType::i64:     return "int64_t";
    case abi::ElementType::f32:     return "float";
    case abi::ElementType::f64:     return "double";
    case abi::ElementType::logical: return "bool";
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
        case abi::ElementType::i32:     os << *((const int *)elem); break;
        case abi::ElementType::i64:     os << *((const int64_t *)elem); break;
        case abi::ElementType::f32:
            format_float<uint32_t>(os, *((const float *)elem), "float", "u");
            break;
        case abi::ElementType::f64:
            format_float<uint64_t>(os, *((const double *)elem), "double", "ull");
            break;
        case abi::ElementType::logical: os << std::boolalpha << *((const bool *)elem); break;
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

} // namespace op::rt::jit
