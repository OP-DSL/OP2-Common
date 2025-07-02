#include <op_mpi_unified_exchanges.h>
#include <op_mpi_cuda_unified_kernels.h>

#include <op_lib_mpi.h>
#include <op_timing2.h>

#include <optional>
#include <vector>
#include <map>
#include <cassert>

using namespace op::mpi::unified;

struct ExchangeSpec {
    op_dat dat;
    std::optional<op_map> map = std::nullopt;

    ExchangeSpec(op_dat dat) : dat{dat} {}
    bool is_partial() const { return map.has_value(); }
};

void extract_gathers(const ExchangeSpec &exchange,
                     std::map<int, std::vector<GatherSpec>> &gathers) {
    auto dat = DatAccessor(exchange.dat);

    if (exchange.is_partial()) {
        auto nonexec_list = OP_export_nonexec_permap[(*exchange.map)->index];

        for (int i = 0; i < nonexec_list->ranks_size; ++i) {
            auto list = export_nonexec_list_partial_d[(*exchange.map)->index] + nonexec_list->disps[i];
            auto gather_spec = GatherSpec(nonexec_list->sizes[i], list, dat);
            gathers[nonexec_list->ranks[i]].push_back(gather_spec);
        }

        return;
    }

    auto exec_list = OP_export_exec_list[exchange.dat->set->index];
    auto nonexec_list = OP_export_nonexec_list[exchange.dat->set->index];

    for (int i = 0; i < exec_list->ranks_size; ++i) {
        auto list = export_exec_list_d[exchange.dat->set->index] + exec_list->disps[i];
        auto gather_spec = GatherSpec(exec_list->sizes[i], list, dat);
        gathers[exec_list->ranks[i]].push_back(gather_spec);
    }

    for (int i = 0; i < nonexec_list->ranks_size; ++i) {
        auto list = export_nonexec_list_d[exchange.dat->set->index] + nonexec_list->disps[i];
        auto gather_spec = GatherSpec(nonexec_list->sizes[i], list, dat);
        gathers[nonexec_list->ranks[i]].push_back(gather_spec);
    }
}

void extract_scatters(const ExchangeSpec &exchange,
                      std::map<int, std::vector<ScatterSpec>> &scatters) {
    auto dat = DatAccessor(exchange.dat);

    if (exchange.is_partial()) {
        auto nonexec_list = OP_import_nonexec_permap[(*exchange.map)->index];

        for (int i = 0; i < nonexec_list->ranks_size; ++i) {
            auto list = import_nonexec_list_partial_d[(*exchange.map)->index] + nonexec_list->disps[i];
            auto scatter_spec = ScatterSpec(nonexec_list->sizes[i], list, dat);
            scatters[nonexec_list->ranks[i]].push_back(scatter_spec);
        }

        return;
    }

    auto exec_list = OP_import_exec_list[exchange.dat->set->index];
    auto nonexec_list = OP_import_nonexec_list[exchange.dat->set->index];

    auto exec_offset = exchange.dat->set->size;
    auto nonexec_offset = exchange.dat->set->size + OP_import_exec_list[exchange.dat->set->index]->size;

    for (int i = 0; i < exec_list->ranks_size; ++i) {
        auto scatter_spec = ScatterSpec(exec_list->sizes[i], exec_offset + exec_list->disps[i], dat);
        scatters[exec_list->ranks[i]].push_back(scatter_spec);
    }

    for (int i = 0; i < nonexec_list->ranks_size; ++i) {
        auto scatter_spec = ScatterSpec(nonexec_list->sizes[i], nonexec_offset + nonexec_list->disps[i], dat);
        scatters[nonexec_list->ranks[i]].push_back(scatter_spec);
    }
}

struct Block {
    void *data;
    size_t size;

    void send(int neighbour, MPI_Request *request) {
        int err = MPI_Isend(data, size, MPI_CHAR, neighbour, 400, OP_MPI_WORLD, request);
        assert(err == MPI_SUCCESS);
    }

    void recv(int neighbour, MPI_Request *request) {
        int err = MPI_Irecv(data, size, MPI_CHAR, neighbour, 400, OP_MPI_WORLD, request);
        assert(err == MPI_SUCCESS);
    }
};

struct ExchangeContext {
    bool exec;

    std::vector<ExchangeSpec> exchanges;

    std::map<int, std::vector<GatherSpec>> gathers_for_neighbour;
    std::map<int, std::vector<ScatterSpec>> scatters_for_neighbour;

    std::map<int, Block> send_blocks;
    std::map<int, Block> recv_blocks;

    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs;

    void reset() {
        exchanges.clear();

        gathers_for_neighbour.clear();
        scatters_for_neighbour.clear();

        send_blocks.clear();
        recv_blocks.clear();
    }

    void add(const op_arg& arg) {
        if (!arg.opt) return;
        if (arg.argtype != OP_ARG_DAT) return;
        if (arg.acc != OP_READ && arg.acc != OP_RW) return;
        if (arg.dat->dirtybit != 1) return;
        if (!exec && arg.map == OP_ID) return;

        for (auto& exchange : exchanges) {
            if (arg.dat->index == exchange.dat->index) {
                // Fallback to full exchange if map mismatch
                if (exchange.is_partial() && (arg.map == OP_ID || (*exchange.map)->index != arg.map->index)) {
                    exchange.map = std::nullopt;
                }

                // Already doing full exchange - nothing to add
                return;
            }
        }

        auto& exchange = exchanges.emplace_back(arg.dat);

        // Do partial exchange if available
        if (arg.map != OP_ID && OP_map_partial_exchange[arg.map->index]) {
            exchange.map = arg.map;
        }
    }

    void construct_lists() {
        for (auto& exchange : exchanges) {
            extract_gathers(exchange, gathers_for_neighbour);
            extract_scatters(exchange, scatters_for_neighbour);
        }
    }

    void alloc_buffers() {
        size_t gather_size = 0;
        for (auto &[neighbour, gathers] : gathers_for_neighbour) {
            for (auto &gather : gathers) {
                gather_size += gather.gather_size();
            }
        }

        size_t scatter_size = 0;
        for (auto &[neighbour, scatters] : scatters_for_neighbour) {
            for (auto &scatter : scatters) {
                scatter_size += scatter.scatter_size();
            }
        }

        auto [gather_buf, scatter_buf] = alloc_exchange_buffers(gather_size, scatter_size);

        size_t gather_offset = 0;
        for (auto &[neighbour, gathers] : gathers_for_neighbour) {
            auto block_start = (void *) ((char * ) gather_buf + gather_offset);
            auto block_start_offset = gather_offset;

            for (auto &gather : gathers) {
                gather.target = (void *) ((char * ) gather_buf + gather_offset);
                gather_offset += gather.gather_size();
            }

            send_blocks[neighbour] = Block{block_start, gather_offset - block_start_offset};
        }

        size_t scatter_offset = 0;
        for (auto &[neighbour, scatters] : scatters_for_neighbour) {
            auto block_start = (void *) ((char * ) scatter_buf + scatter_offset);
            auto block_start_offset = scatter_offset;

            for (auto& scatter : scatters) {
                scatter.source = (void *) ((char * ) scatter_buf + scatter_offset);
                scatter_offset += scatter.scatter_size();
            }

            recv_blocks[neighbour] = Block{block_start, scatter_offset - block_start_offset};
        }
    }

    void initiate_gathers() {
        if (gathers_for_neighbour.size() > 0) {
            ::initiate_gathers(gathers_for_neighbour);
        }

        auto recv_index = 0;
        recv_reqs.resize(recv_blocks.size());
        for (auto [neighbour, block] : recv_blocks) {
            block.recv(neighbour, &recv_reqs[recv_index]);
            ++recv_index;
        }
    }

    void wait_gathers() {
        if (gathers_for_neighbour.size() > 0) {
            ::wait_gathers();
        }
    }

    void exchange_buffers() {
        send_reqs.resize(send_blocks.size());

        auto send_index = 0;
        for (auto [neighbour, block] : send_blocks) {
            block.send(neighbour, &send_reqs[send_index]);
            ++send_index;
        }

        if (recv_reqs.size() > 0) {
            MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
            recv_reqs.clear();
        }
    }

    void initiate_scatters() {
        if (scatters_for_neighbour.size() > 0) {
            ::initiate_scatters(scatters_for_neighbour);
            op_scatter_sync();
        }

        if (send_reqs.size() > 0) {
            MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
            send_reqs.clear();
        }
    }

    void set_dirtybits() {
        for (auto& exchange : exchanges) {
            if (exchange.is_partial()) continue;

            exchange.dat->dirtybit = 0;
            exchange.dat->dirty_hd = 2;
        }
    }
};


ExchangeContext ctx;


int op_mpi_halo_exchanges_unified(op_set set, int nargs, op_arg *args) {
    op_timing2::instance().enter2("Halo Exchanges Unified", false);

    bool exec = false;
    int size = set->size;

    for (int n = 0; n < nargs; ++n) {
        if (!args[n].opt) continue;
        if (args[n].argtype != OP_ARG_DAT || args[n].idx == -1) continue;
        if (args[n].acc == OP_READ) continue;

        exec = true;
        size += set->exec_size;
        break;
    }

    ctx.reset();
    ctx.exec = exec;

    for (int n = 0; n < nargs; ++n) {
        ctx.add(args[n]);
    }

    if (ctx.exchanges.size() > 0) {
        ctx.construct_lists();
        ctx.alloc_buffers();
        ctx.initiate_gathers();
    }

    op_timing2::instance().exit2(false);
    return size;
}

void op_mpi_wait_all_unified(int nargs, op_arg *args) {
    op_timing2::instance().enter2("Halo Exchanges Wait Unified", false);

    if (ctx.exchanges.size() > 0) {
        ctx.wait_gathers();

        op_timing2::instance().enter2("Exchange buffers", false);
        ctx.exchange_buffers();
        op_timing2::instance().exit2(false);

        ctx.initiate_scatters();
        ctx.set_dirtybits();
    }

    op_timing2::instance().exit2(false);
}
