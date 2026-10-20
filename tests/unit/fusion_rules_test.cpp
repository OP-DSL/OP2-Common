// When queued calls may share a kernel: element by element in queue order,
// or all at once.

#include <loop/fusion.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusion = op::loop::fusion;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

// Two dats on the nodes, a map from the edges to them, and one from the
// nodes to themselves.
struct Mesh {
    op_set_core nodes{};
    op_set_core edges{};
    op_dat_core x{};
    op_dat_core y{};
    op_map_core edge_nodes{};
    op_map_core next_node{};

    Mesh() {
        for (auto *dat : {&x, &y}) {
            dat->set = &nodes;
            dat->dim = 1;
            dat->size = sizeof(double);
            dat->type = "double";
        }
        edge_nodes.from = &edges;
        edge_nodes.to = &nodes;
        next_node.from = &nodes;
        next_node.to = &nodes;
    }
};

op_arg arg(op_dat dat, op_map map, op_access access, int opt = 1) {
    op_arg out{};
    out.opt = opt;
    out.argtype = OP_ARG_DAT;
    out.dat = dat;
    out.map = map;
    out.idx = map != nullptr ? 0 : -1;
    out.acc = access;
    return out;
}

using Args = std::vector<op_arg>;

bool element_wise(const Args& a, bool a_atomic, const Args& b, bool b_atomic) {
    return fusion::element_wise(fusion::uses(a), a_atomic, fusion::uses(b), b_atomic);
}

bool independent(const Args& a, bool a_atomic, const Args& b, bool b_atomic) {
    return fusion::independent(fusion::uses(a), a_atomic, fusion::uses(b), b_atomic);
}

void test_rules() {
    Mesh mesh;
    auto *x = &mesh.x;
    auto *y = &mesh.y;
    auto *edges = &mesh.edge_nodes;
    auto *next = &mesh.next_node;

    // A direct write read directly: the same element, in order.
    Args write_x{arg(x, nullptr, OP_WRITE)};
    Args read_x{arg(x, nullptr, OP_READ), arg(y, nullptr, OP_WRITE)};
    CHECK(element_wise(write_x, false, read_x, false));
    CHECK(!independent(write_x, false, read_x, false));

    // Reads alone never conflict.
    Args read_only{arg(x, nullptr, OP_READ)};
    Args read_through{arg(x, next, OP_READ)};
    CHECK(element_wise(read_only, false, read_through, false));
    CHECK(independent(read_only, false, read_through, false));

    // A direct write read through a map to the same set: other elements.
    Args write_y{arg(y, nullptr, OP_WRITE)};
    Args read_y_next{arg(y, next, OP_READ), arg(x, nullptr, OP_WRITE)};
    CHECK(!element_wise(write_y, false, read_y_next, false));
    CHECK(!independent(write_y, false, read_y_next, false));

    // Atomic indirect increments commute, either way.
    Args inc_x{arg(x, edges, OP_INC)};
    CHECK(element_wise(inc_x, true, inc_x, true));
    CHECK(independent(inc_x, true, inc_x, true));
    CHECK(!element_wise(inc_x, false, inc_x, true));
    CHECK(!independent(inc_x, true, inc_x, false));

    // Increments then a read of the result: every increment first.
    Args read_x_direct{arg(x, nullptr, OP_READ)};
    CHECK(!element_wise(inc_x, true, read_x_direct, false));
    CHECK(!independent(inc_x, true, read_x_direct, false));

    // A direct increment is one element's own, but no atomic.
    Args inc_x_direct{arg(x, nullptr, OP_INC)};
    CHECK(element_wise(inc_x_direct, false, inc_x_direct, false));
    CHECK(!independent(inc_x_direct, false, inc_x_direct, false));
    CHECK(element_wise(inc_x_direct, false, inc_x, true));
    CHECK(!independent(inc_x_direct, false, inc_x, true));

    // An indirect write is no increment.
    Args write_x_indirect{arg(x, edges, OP_WRITE)};
    CHECK(!element_wise(write_x_indirect, true, inc_x, true));
    CHECK(!independent(write_x_indirect, true, inc_x, true));

    // An inactive argument is no use.
    Args inactive{arg(x, edges, OP_WRITE, 0)};
    CHECK(fusion::uses(inactive).empty());
    CHECK(independent(inactive, false, write_x, false));

    // Two arguments on one dat make one use of both.
    Args both{arg(x, nullptr, OP_READ), arg(x, edges, OP_INC)};
    auto use = fusion::uses(both);
    CHECK(use.size() == 1 && use[0].read && use[0].write && !use[0].only_inc &&
          !use[0].all_direct && use[0].any_direct && use[0].indirect_inc);
}

} // namespace

int main() {
    try {
        test_rules();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("fusion rules tests passed\n");
    return EXIT_SUCCESS;
}
