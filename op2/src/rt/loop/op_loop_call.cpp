#include <op2/rt/loop/call.h>

#include <extern/rapidhash.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace op::rt::loop {

// --- One call -----------------------------------------------------------------

bool Call::global_reduction() const {
    for (const op_arg& arg : args) {
        if (arg.opt == 0 || arg.argtype != OP_ARG_GBL)
            continue;
        if (arg.acc == OP_INC || arg.acc == OP_MIN || arg.acc == OP_MAX)
            return true;
    }
    return false;
}

bool Call::writes_globals() const {
    for (const op_arg& arg : args) {
        if (arg.opt == 0 || arg.argtype != OP_ARG_GBL)
            continue;
        if (arg.acc == OP_INC || arg.acc == OP_MIN || arg.acc == OP_MAX ||
            arg.acc == OP_RW || arg.acc == OP_WRITE)
            return true;
    }
    return false;
}

std::uint64_t Call::key() const {
    if (m_keyed)
        return m_key;

    // The words that shape the call, hashed a buffer at a time: per
    // argument its activity, kind, access and map index in one word, then
    // the first argument on its dat, its map and dimension.
    constexpr std::size_t arg_words = 4;
    std::array<std::int32_t, 2 + 32 * arg_words> words;
    std::size_t count = 0;
    std::uint64_t hash = RAPID_SEED;
    words[count++] = set->index;
    words[count++] = run_size;
    auto first_on_dat = [&](std::size_t i) {
        for (std::size_t j = 0; j < i; ++j)
            if (args[j].opt != 0 && args[j].dat == args[i].dat)
                return static_cast<std::int32_t>(j);
        return static_cast<std::int32_t>(i);
    };
    for (std::size_t i = 0; i < args.size(); ++i) {
        const op_arg& arg = args[i];
        if (count + arg_words > words.size()) {
            hash = rapidhash_withSeed(words.data(), count * sizeof(words[0]), hash);
            count = 0;
        }
        if (arg.opt == 0) {
            words[count++] = 0;
            continue;
        }
        words[count++] = 1 | arg.argtype << 1 | arg.acc << 4 | (arg.idx & 0xffff) << 16;
        words[count++] = arg.dat != nullptr ? first_on_dat(i) : -1;
        words[count++] = arg.map != nullptr ? arg.map->index : -1;
        words[count++] = arg.dim;
    }

    m_key = rapidhash_withSeed(words.data(), count * sizeof(words[0]), hash);
    m_keyed = true;
    return m_key;
}

std::string Call::describe() const {
    // Fortran pads names with spaces.
    auto trimmed = [](const char *name) {
        std::string text = name != nullptr ? name : "?";
        text.erase(text.find_last_not_of(' ') + 1);
        return text;
    };
    std::string text = trimmed(set->name);
    text += " (" + std::to_string(set->size) + ")";

    std::string maps;
    std::vector<int> listed;
    std::string inactive;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const op_arg& arg = args[i];
        if (arg.opt == 0) {
            inactive += (inactive.empty() ? "" : " ") + std::to_string(i);
            continue;
        }
        if (arg.map == nullptr || arg.map->name == nullptr ||
            std::find(listed.begin(), listed.end(), arg.map->index) != listed.end())
            continue;
        listed.push_back(arg.map->index);
        maps += (maps.empty() ? "" : ", ") + trimmed(arg.map->name);
    }
    if (!maps.empty())
        text += " via " + maps;
    if (global_reduction())
        text += ", reduction";
    if (!inactive.empty())
        text += ", inactive args " + inactive;
    return text;
}

Sections Sections::direct(op_set set) {
    Sections sections;
    sections.m_sections[0] = {0, static_cast<int>(set->size)};
    sections.m_count = 1;
    return sections;
}

Sections Sections::indirect(op_set set, bool separate_owned, int run_size) {
    assert(run_size >= set->size && run_size <= set->size + set->exec_size);
    Sections sections;
    int size = static_cast<int>(set->size);
    int end = run_size;
    if (separate_owned) {
        sections.m_sections = {{{0, set->core_size},
                                {set->core_size, size},
                                {size, end}}};
        sections.m_count = 3;
    } else {
        sections.m_sections = {{{0, set->core_size}, {set->core_size, end}}};
        sections.m_count = 2;
    }
    sections.m_wait_before = 1;
    sections.m_globals_after = 1;
    return sections;
}

std::optional<abi::ElementType> element_type(const char *type) {
    auto is = [type](std::initializer_list<const char *> names) {
        return type != nullptr &&
               std::any_of(names.begin(), names.end(), [type](const char *name) {
                   return std::strcmp(type, name) == 0;
               });
    };
    if (is({"double", "r8", "real*8", "real(8)"}))
        return abi::ElementType::f64;
    if (is({"float", "r4", "real*4", "real(4)"}))
        return abi::ElementType::f32;
    if (is({"int", "i4", "integer*4", "integer(4)"}))
        return abi::ElementType::i32;
    if (is({"bool", "logical"}))
        return abi::ElementType::logical;
    return std::nullopt;
}

void reduce_mpi_globals(op_arg *args, int nargs, const char *loop) {
    for (int i = 0; i < nargs; ++i) {
        auto& arg = args[i];
        if (arg.opt == 0 || arg.argtype != OP_ARG_GBL ||
            (arg.acc != OP_INC && arg.acc != OP_MIN && arg.acc != OP_MAX))
            continue;

        auto type = element_type(arg.type);
        if (type == abi::ElementType::f64) {
            op_mpi_reduce_double(&arg, reinterpret_cast<double *>(arg.data));
        } else if (type == abi::ElementType::f32) {
            op_mpi_reduce_float(&arg, reinterpret_cast<float *>(arg.data));
        } else if (type == abi::ElementType::i32) {
            op_mpi_reduce_int(&arg, reinterpret_cast<int *>(arg.data));
        } else if (type == abi::ElementType::logical) {
            op_mpi_reduce_bool(&arg, reinterpret_cast<bool *>(arg.data));
        } else {
            std::fprintf(stderr, "error: unsupported MPI reduction type '%s' (in %s)\n",
                         arg.type == nullptr ? "<null>" : arg.type, loop);
            std::exit(1);
        }
    }
}

} // namespace op::rt::loop
