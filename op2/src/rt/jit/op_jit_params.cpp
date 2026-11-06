#include <op2/rt/jit/params.h>

#include <op2/env.h>
#include <op_lib_core.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>

namespace op::rt::jit {

void Param::settle(std::string_view loop, const Options& options) {
    bool runtime = m_row.runtime;
    auto named = [&](const std::vector<std::string>& names) {
        return std::any_of(names.begin(), names.end(), [&](std::string_view item) {
            auto scoped = env::scoped_item(item, loop);
            return scoped && scoped->name == name();
        });
    };

    m_fixed = true;
    m_initial = ParamMode::baked;
    if (named(options.constant_params))
        return;
    if (named(options.runtime_params)) {
        if (runtime)
            m_initial = ParamMode::runtime;
        else
            env::warn("warning: " + std::string{loop} + "'s kernels cannot have " + name() +
                      " at run time, from OP_JIT_RUNTIME_PARAMS\n");
    } else if (runtime && source() == abi::ParamSource::external) {
        if (bytes() > static_cast<std::size_t>(options.max_const_bytes))
            m_initial = ParamMode::runtime;
        else
            m_fixed = false;
    } else if (runtime && global()) {
        // An array_arg's size is each call's, so its cap is checked per call.
        m_initial = ParamMode::runtime;
        m_fixed = options.level < 2 ||
                  (source() == abi::ParamSource::scalar_arg &&
                   bytes() > static_cast<std::size_t>(options.max_gbl_bytes));
    }
}

void Param::update(std::span<const op_arg> args, DatStride dat_stride, const Options& options) {
    int index = m_row.index;
    switch (source()) {
    case abi::ParamSource::scalar_arg:
        assert(index >= 0 && static_cast<std::size_t>(index) < args.size());
        // An inactive optional global's kernel never reads it.
        if (args[index].opt != 0) {
            assert(args[index].data != nullptr);
            std::memcpy(m_data, args[index].data, elem_size());
        }
        break;
    case abi::ParamSource::array_arg: {
        assert(index >= 0 && static_cast<std::size_t>(index) < args.size());
        bool active = args[index].opt != 0;
        m_data = active ? args[index].data : nullptr;
        m_elems = active ? static_cast<std::size_t>(args[index].dim) : 0;
        break;
    }
    case abi::ParamSource::dat_stride:
        assert(index >= 0 && static_cast<std::size_t>(index) < args.size());
        assert(elem_size() == sizeof(int) && m_elems == 1 && dat_stride != nullptr);
        *static_cast<int *>(m_data) = dat_stride(args[index]);
        break;
    case abi::ParamSource::dat_dim:
        assert(index >= 0 && static_cast<std::size_t>(index) < args.size());
        assert(elem_size() == sizeof(int) && m_elems == 1);
        *static_cast<int *>(m_data) = args[index].dim;
        break;
    case abi::ParamSource::strategy:
    case abi::ParamSource::global_stride:
        // Its value comes with each kernel's key.
        return;
    default:
        break;
    }
    if (!sized_out(options))
        rehash();
}

void ParamSet::update(std::span<const op_arg> args) {
    for (auto& param : m_params)
        param.update(args, m_dat_stride, m_options);
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
    if (!m_moving || !m_options.enabled())
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

std::span<const ParamMode> ParamSet::observe(const loop::Call& call) {
    auto *history = m_moving ? histories(call) : nullptr;
    if (history != nullptr)
        ++history->calls;
    for (std::size_t i = 0; i < m_params.size(); ++i) {
        const auto& param = m_params[i];
        if (param.sized_out(m_options)) {
            m_modes[i] = ParamMode::runtime;
            continue;
        }
        if (param.fixed() || history == nullptr) {
            m_modes[i] = param.initial();
            continue;
        }
        auto& entry = history->params[i];
        if (entry.observe(param, m_options) && m_options.debug)
            op_printf("jit: %s %s %s for %s\n", m_loop.c_str(),
                        entry.mode == ParamMode::baked ? "bakes in" : "moves to run time",
                        param.name(), call.describe().c_str());
        m_modes[i] = entry.mode;
    }
    return m_modes;
}

bool ParamSet::History::observe(const Param& param, const Options& options) {
    std::uint64_t hash = param.hash();
    auto limit = static_cast<std::size_t>(options.max_values);
    if (!seen) {
        seen = true;
        last = hash;
        values.assign(1, hash);
        calls = 1;
        mode = param.initial();
        return false;
    }

    // One past the limit decides, so no more are recorded.
    if (hash != last && values.size() <= limit &&
        std::ranges::find(values, hash) == values.end())
        values.push_back(hash);
    last = hash;
    if (mode == ParamMode::baked) {
        if (values.size() <= limit)
            return false;
        mode = ParamMode::runtime;
        demoted = true;
        return true;
    }

    if (demoted || promoted || !param.global() || ++calls < statistics_window)
        return false;
    if (values.size() > limit) {
        values.assign(1, hash);
        calls = 0;
        return false;
    }
    mode = ParamMode::baked;
    promoted = true;
    values.assign(1, hash);
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

template<typename T>
T load(const char *bytes) {
    T value;
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

// An integer as a literal; its type's minimum, whose magnitude no literal of
// the type holds, as an expression.
template<typename T>
void format_int(std::ostream& os, T value) {
    if (value == std::numeric_limits<T>::min())
        os << "(" << value + 1 << " - 1)";
    else
        os << value;
}

// A finite value in hexfloat, exactly; a NaN or infinity, which hexfloat
// writes as a word no compiler reads, as its bits.
template<typename Bits, typename T>
void format_float(std::ostream& os, T value, const char *type, const char *suffix) {
    if (std::isfinite(value)) {
        os << std::hexfloat << value;
        return;
    }
    os << "__builtin_bit_cast(" << type << ", 0x" << std::hex << std::bit_cast<Bits>(value)
       << std::dec << suffix << ")";
}

std::string format_value(const Param& param) {
    std::ostringstream os;
    if (param.array())
        os << "{ ";
    for (std::size_t i = 0; i < param.elems(); ++i) {
        const char *elem = static_cast<const char *>(param.data()) + param.elem_size() * i;
        switch (param.type()) {
        case abi::ElementType::i32:     format_int(os, load<std::int32_t>(elem)); break;
        case abi::ElementType::i64:     format_int(os, load<std::int64_t>(elem)); break;
        case abi::ElementType::f32:
            format_float<std::uint32_t>(os, load<float>(elem), "float", "u");
            break;
        case abi::ElementType::f64:
            format_float<std::uint64_t>(os, load<double>(elem), "double", "ull");
            break;
        case abi::ElementType::logical: os << std::boolalpha << load<bool>(elem); break;
        }
        if (param.array() && i < param.elems() - 1)
            os << ", ";
    }
    if (param.array())
        os << " }";
    return os.str();
}

} // namespace

std::string format_cxx(const ParamSet& params, std::span<const ParamMode> modes,
                       std::span<const int> values) {
    std::string out;
    for (std::size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        if (modes[i] != ParamMode::baked)
            continue;
        out += "static constexpr ";
        out += cxx_type(param.type());
        out += " ";
        out += param.name();
        if (param.array())
            out += "[" + std::to_string(param.elems()) + "]";
        out += " = ";
        out += param.slot() >= 0
                   ? std::to_string(values[static_cast<std::size_t>(param.slot())])
                   : format_value(param);
        out += ";\n";
    }
    return out;
}

} // namespace op::rt::jit
