#include <op_mpi_unified_exchanges.h>
#include <op_mpi_unified_backend.h>

#include <op_lib_mpi.h>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <string>

namespace op::unified_exchanges {

Backend &backend_for(int device) {
    Backend *backend = (device == 2) ? device_backend() : host_backend();

    if (backend == nullptr) {
        std::printf("op_mpi_halo_exchanges_unified: no unified exchange backend "
                    "for device %d in this library\n", device);
        std::exit(-1);
    }

    return *backend;
}

struct ExchangeSpec {
    op_dat dat;

    // Null unless this dat is going out as a partial exchange over that map.
    op_map map = nullptr;

    ExchangeSpec(op_dat dat) : dat{dat} {}
    bool is_partial() const { return map != nullptr; }
};

void extract_gathers(const DatPlacement &placement, const ExchangeSpec &exchange,
                     SpecsByNeighbour<GatherSpec> &gathers) {
    auto &dat = placement.dat;

    if (exchange.is_partial()) {
        auto nonexec_list = OP_export_nonexec_permap[exchange.map->index];

        for (int i = 0; i < nonexec_list->ranks_size; ++i) {
            auto list = placement.nonexec_export_permap + nonexec_list->disps[i];
            gathers[nonexec_list->ranks[i]].emplace_back(nonexec_list->sizes[i], list, dat);
        }

        return;
    }

    auto exec_list = OP_export_exec_list[exchange.dat->set->index];
    auto nonexec_list = OP_export_nonexec_list[exchange.dat->set->index];

    for (int i = 0; i < exec_list->ranks_size; ++i) {
        auto list = placement.exec_export + exec_list->disps[i];
        gathers[exec_list->ranks[i]].emplace_back(exec_list->sizes[i], list, dat);
    }

    for (int i = 0; i < nonexec_list->ranks_size; ++i) {
        auto list = placement.nonexec_export + nonexec_list->disps[i];
        gathers[nonexec_list->ranks[i]].emplace_back(nonexec_list->sizes[i], list, dat);
    }
}

void extract_scatters(const DatPlacement &placement, const ExchangeSpec &exchange,
                      SpecsByNeighbour<ScatterSpec> &scatters) {
    auto &dat = placement.dat;

    if (exchange.is_partial()) {
        auto nonexec_list = OP_import_nonexec_permap[exchange.map->index];

        for (int i = 0; i < nonexec_list->ranks_size; ++i) {
            auto list = placement.nonexec_import_permap + nonexec_list->disps[i];
            scatters[nonexec_list->ranks[i]].emplace_back(nonexec_list->sizes[i], list, dat);
        }

        return;
    }

    auto exec_list = OP_import_exec_list[exchange.dat->set->index];
    auto nonexec_list = OP_import_nonexec_list[exchange.dat->set->index];

    auto exec_offset = exchange.dat->set->size;
    auto nonexec_offset = exchange.dat->set->size + OP_import_exec_list[exchange.dat->set->index]->size;

    for (int i = 0; i < exec_list->ranks_size; ++i) {
        scatters[exec_list->ranks[i]].emplace_back(exec_list->sizes[i],
                                                   (int) (exec_offset + exec_list->disps[i]), dat);
    }

    for (int i = 0; i < nonexec_list->ranks_size; ++i) {
        scatters[nonexec_list->ranks[i]].emplace_back(nonexec_list->sizes[i],
                                                      (int) (nonexec_offset + nonexec_list->disps[i]), dat);
    }
}

struct Block {
    void *data;
    size_t size;

    void send(int neighbour, MPI_Request *request, int tag) {
        int err = MPI_Isend(data, size, MPI_CHAR, neighbour, tag, OP_MPI_WORLD, request);
        assert(err == MPI_SUCCESS);
    }

    void recv(int neighbour, MPI_Request *request, int tag) {
        int err = MPI_Irecv(data, size, MPI_CHAR, neighbour, tag, OP_MPI_WORLD, request);
        assert(err == MPI_SUCCESS);
    }
};

struct ExchangeContext {
    bool exec;

    // Where the scatter lands: 1 host, 2 device. Recorded because the copy that
    // receives the halo is the one that becomes current (see exchange_and_scatter).
    int device = 1;

    static constexpr int tag_ini = 0x7000;
    static constexpr int tag_max = 0x8000;
    int tag = tag_ini;

    // The set of the exchange that has not been waited for yet, null when there
    // is none. See unpaired().
    op_set unwaited = nullptr;

    // The dats of the outstanding exchange: filled by add(), emptied once
    // exchange_and_scatter() has completed it.
    std::vector<ExchangeSpec> exchanges;

    SpecsByNeighbour<GatherSpec> gathers_for_neighbour;
    SpecsByNeighbour<ScatterSpec> scatters_for_neighbour;

    // Built once and iterated once per exchange, so a vector beats a map:
    // clear() keeps the capacity instead of freeing a node per neighbour.
    std::vector<std::pair<int, Block>> send_blocks;
    std::vector<std::pair<int, Block>> recv_blocks;

    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs;

    Backend *backend = nullptr;

    // Spec counts, not map sizes: the maps keep their keys between exchanges so
    // a neighbour can be present with an empty vector.
    size_t n_gather_specs = 0;
    size_t n_scatter_specs = 0;

    void reset(int device, bool exec) {
        backend = &backend_for(device);

        this->device = device;
        this->exec = exec;

        tag++;
        if (tag >= ExchangeContext::tag_max) tag = ExchangeContext::tag_ini;

        exchanges.clear();

        // Keep the buckets and the inner vectors' capacity: clearing the maps
        // frees every node and vector, so each exchange would reallocate them.
        for (auto &[neighbour, gathers] : gathers_for_neighbour) gathers.clear();
        for (auto &[neighbour, scatters] : scatters_for_neighbour) scatters.clear();

        send_blocks.clear();
        recv_blocks.clear();

        n_gather_specs = 0;
        n_scatter_specs = 0;
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
                if (exchange.is_partial() && (arg.map == OP_ID || exchange.map->index != arg.map->index)) {
                    exchange.map = nullptr;
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

    void prepare_and_gather() {
        if (exchanges.size() == 0) {
            return;
        }

        // Normalise exchanges
        std::sort(exchanges.begin(), exchanges.end(), [](auto& a, auto& b) {
            return a.dat->index < b.dat->index;
        });

        for (auto& exchange : exchanges) {
            auto placement = backend->placement(exchange.dat, exchange.map);

            extract_gathers(placement, exchange, gathers_for_neighbour);
            extract_scatters(placement, exchange, scatters_for_neighbour);
        }

        size_t gather_size = 0;
        for (auto &[neighbour, gathers] : gathers_for_neighbour) {
            n_gather_specs += gathers.size();
            for (auto &gather : gathers) {
                gather_size += gather.gather_size();
            }
        }

        size_t scatter_size = 0;
        for (auto &[neighbour, scatters] : scatters_for_neighbour) {
            n_scatter_specs += scatters.size();
            for (auto &scatter : scatters) {
                scatter_size += scatter.scatter_size();
            }
        }

        // Wait for any previous sendreqs to complete
        if (send_reqs.size() > 0) {
            MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
            send_reqs.clear();
        }

        auto bufs = backend->alloc_buffers(gather_size, scatter_size);

        size_t gather_offset = 0;
        for (auto &[neighbour, gathers] : gathers_for_neighbour) {
            if (gathers.empty()) continue;
            auto block_start = (void *) ((char * ) bufs.gather_mpi + gather_offset);
            auto block_start_offset = gather_offset;

            for (auto &gather : gathers) {
                gather.target = (void *) ((char * ) bufs.gather + gather_offset);
                gather_offset += gather.gather_size();
            }

            send_blocks.emplace_back(neighbour, Block{block_start, gather_offset - block_start_offset});
        }

        size_t scatter_offset = 0;
        for (auto &[neighbour, scatters] : scatters_for_neighbour) {
            if (scatters.empty()) continue;
            auto block_start = (void *) ((char * ) bufs.scatter_mpi + scatter_offset);
            auto block_start_offset = scatter_offset;

            for (auto& scatter : scatters) {
                scatter.source = (void *) ((char * ) bufs.scatter + scatter_offset);
                scatter_offset += scatter.scatter_size();
            }

            recv_blocks.emplace_back(neighbour, Block{block_start, scatter_offset - block_start_offset});
        }

        // Initiate gathers
        if (n_gather_specs > 0) {
            backend->initiate_gathers(gathers_for_neighbour);
        }

        // A synchronous backend has already filled the send blocks, so the sends
        // can go out now rather than at wait-all.
        if (backend->synchronous) {
            post_sends();
        }

        // Wait for previous scatter kernels to complete before initiating MPI recvs
        backend->wait_scatters();

        post_recvs();
    }

    void post_sends() {
        send_reqs.resize(send_blocks.size());

        auto send_index = 0;
        for (auto [neighbour, block] : send_blocks) {
            block.send(neighbour, &send_reqs[send_index], tag);
            ++send_index;
        }
    }

    void post_recvs() {
        recv_reqs.resize(recv_blocks.size());

        auto recv_index = 0;
        for (auto [neighbour, block] : recv_blocks) {
            block.recv(neighbour, &recv_reqs[recv_index], tag);
            ++recv_index;
        }
    }

    void exchange_and_scatter() {
        if (exchanges.size() == 0) {
            return;
        }

        if (!backend->synchronous) {
            if (n_gather_specs > 0) {
                backend->wait_gathers();
            }

            post_sends();
        }

        if (recv_reqs.size() > 0) {
            MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
            recv_reqs.clear();
        }

        if (n_scatter_specs > 0) {
            backend->initiate_scatters(scatters_for_neighbour);
        }

        // Set dirtybits
        for (auto& exchange : exchanges) {
            if (exchange.is_partial()) continue;

            // The scatter wrote into this backend's copy, so that copy is now the
            // current one: dirty_hd 1 means the host copy is newer, 2 the device
            // copy. A constant 2 made a host exchange in an MPI+CUDA build claim
            // the device was current, and the next host loop then downloaded the
            // stale device copy over the halo it had just received.
            exchange.dat->dirtybit = 0;
            exchange.dat->dirty_hd = device;
        }

        exchanges.clear();
    }

    // Progress the outstanding requests without blocking, completing them if
    // they are done. The receives complete here only as far as MPI goes: the
    // scatter still waits for exchange_and_scatter().
    void test() {
        test_requests(recv_reqs);
        test_requests(send_reqs);
    }

    static void test_requests(std::vector<MPI_Request> &reqs) {
        if (reqs.empty()) return;

        int done = 0;
        MPI_Testall(reqs.size(), reqs.data(), &done, MPI_STATUSES_IGNORE);
        if (done) reqs.clear();
    }
};

ExchangeContext ctx;

// Every exchange is followed by exactly one wait before the next exchange, even
// when it moved nothing. A caller that skips the wait may already have read a
// stale halo, and nothing later can tell, so an unpaired call stops the job
// rather than being tidied up.
[[noreturn]] void unpaired(const std::string &problem) {
    std::fprintf(stderr, "OP2: %s. Every halo exchange (op_mpi_halo_exchanges*) must be followed by "
                         "exactly one wait (op_mpi_wait_all*) before the next exchange.\n", problem.c_str());
    MPI_Abort(OP_MPI_WORLD, 1);
    std::abort();
}

}  // namespace op::unified_exchanges

using namespace op::unified_exchanges;

int op_mpi_halo_exchanges_unified(op_set set, int nargs, op_arg *args, int device) {
    if (ctx.unwaited != nullptr)
        unpaired(std::string("the halo exchange on set '") + ctx.unwaited->name + "' was never waited for");
    ctx.unwaited = set;

    // Bring each dat into the space this loop runs in - every arg, not just the
    // ones that end up being exchanged, since a direct loop exchanges nothing but
    // still reads whichever copy is current.
    for (int n = 0; n < nargs; ++n) {
        if (!args[n].opt || args[n].argtype != OP_ARG_DAT) continue;

        if (device == 2 && args[n].dat->dirty_hd == 1) {
            op_upload_dat(args[n].dat);
            args[n].dat->dirty_hd = 0;
        }

        if (device == 1 && args[n].dat->dirty_hd == 2) {
            op_download_dat(args[n].dat);
            args[n].dat->dirty_hd = 0;
        }
    }

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

    ctx.reset(device, exec);

    for (int n = 0; n < nargs; ++n) {
        ctx.add(args[n]);
    }

    ctx.prepare_and_gather();
    return size;
}

void op_mpi_wait_all_unified(int, op_arg *) {
    if (ctx.unwaited == nullptr)
        unpaired("a wait with no halo exchange to wait for");

    ctx.exchange_and_scatter();
    ctx.unwaited = nullptr;
}

void op_mpi_test_all_unified(int, op_arg *) {
    ctx.test();
}

void op_mpi_unified_exit() {
    if (ctx.unwaited != nullptr)
        unpaired(std::string("op_exit with the halo exchange on set '") + ctx.unwaited->name + "' never waited for");
}
