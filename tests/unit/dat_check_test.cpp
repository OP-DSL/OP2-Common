// A dat argument whose dim (dat_check_test dim) or element type
// (dat_check_test type) is not its dat's stops the run, after a loop with
// matching arguments has run.  Each loop is its own call site, whose types
// its first call names.

#include <op2/rt/abi/cpu/descriptor.h>
#include <op2/rt/kernel/cpu/threads.h>
#include <op_lib_c.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
    if (argc != 2 || (std::strcmp(argv[1], "dim") != 0 && std::strcmp(argv[1], "type") != 0)) {
        std::fprintf(stderr, "usage: dat_check_test dim|type\n");
        return EXIT_FAILURE;
    }

    op_init(1, argv, 0);
    op_set nodes = op_decl_set(4, "nodes");
    double q[8] = {};
    op_dat q_dat = op_decl_dat_char(nodes, 2, "double", sizeof(double),
                                    reinterpret_cast<char *>(q), "q");

    static int calls = 0;
    const op::rt::abi::cpu::Family families[] = {
        {{"plain", {}}, [](const op::rt::abi::cpu::Range&, op_arg *) { ++calls; }, nullptr},
    };
    op::rt::abi::cpu::Loop valid_loop{{.name = "valid", .target = "openmp", .direct = true},
                                      families, &op::rt::kernel::cpu::max_threads};
    op::rt::abi::cpu::Loop loop{{.name = "check", .target = "openmp", .direct = true},
                                families, &op::rt::kernel::cpu::max_threads};

    op_arg valid = op_arg_dat(q_dat, -1, nullptr, 2, "double", OP_READ);
    op::rt::abi::cpu::invoke(valid_loop, nodes, &valid, 1);
    if (calls == 0) {
        std::fprintf(stderr, "the valid call did not run\n");
        return EXIT_FAILURE;
    }

    bool dim = std::strcmp(argv[1], "dim") == 0;
    op_arg wrong = op_arg_dat(q_dat, -1, nullptr, dim ? 1 : 2, dim ? "double" : "float", OP_READ);
    op::rt::abi::cpu::invoke(loop, nodes, &wrong, 1);
    std::fprintf(stderr, "the mismatched call ran\n");
    return EXIT_FAILURE;
}
