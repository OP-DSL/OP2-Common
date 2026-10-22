#include <loop/jit.h>

#include <op_rt_support.h>

#include <cmath>
#include <cstring>
#include <sstream>

namespace op::loop {

// --- Policy -------------------------------------------------------------------

namespace {

JitPolicy policy;
std::atomic_int builds_in_flight = 0;

} // namespace

const JitPolicy& jit_policy() { return policy; }

void init_jit_policy() {
    policy = {};
    env::parse_bool("OP_JIT_ENABLE", policy.enable);
    if (!policy.enable)
        op_printf("Disabling JIT compilation\n");
    env::parse_bool("OP_JIT_DEBUG", policy.debug);
    if (policy.debug)
        op_printf("Enabling JIT debug\n");
    env::parse_bool("OP_JIT_SEQ_COMPILE", policy.seq_compile);
    env::parse_bool("OP_JIT_FORCE", policy.force);
    if (policy.force)
        op_printf("Forcing JIT compilation regardless of register count\n");
    env::parse_number("OP_JIT_MIN_CALLS", policy.min_calls);
    int threads = 0;
    if (env::parse_number("OP_JIT_MAX_THREADS", threads))
        policy.max_threads = threads;
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
    for (auto& param : m_params)
        param.update(args, nargs);
}

namespace {

const char *format_type(ParamType type) {
    switch (type) {
    case ParamType::i32:     return "int";
    case ParamType::i64:     return "int64_t";
    case ParamType::f32:     return "float";
    case ParamType::f64:     return "double";
    case ParamType::logical: return "bool";
    }
    __builtin_unreachable();
}

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
        out += format_type(param.type());
        out += " " + param.name();
        if (param.array())
            out += "[" + std::to_string(param.elems()) + "]";
        out += " = " + format_value(param) + ";\n";
    }
    return out;
}

} // namespace op::loop
