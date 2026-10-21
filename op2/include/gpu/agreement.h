#pragma once

// Fusion decisions every rank shares.  A decision that depends on the rank
// (its kernel compiled yet, its timings) takes effect only once all ranks
// agree, and then on all of them at the same flush, so their plans stay
// alike.  Each fused group has a slot in a fixed table, by its structural
// hash; the table's maximum over the ranks is started at one collective
// flush (a global returned to the host) and finished at the next.  A slot
// reads: not ready (1 until the rank has the kernel, 2 when it never will),
// and the rank's fused and apart medians in microseconds once timed (-1
// before).  A rank without the group leaves its slot not ready, so a group
// some rank lacks is fused nowhere; timings agree on the slowest rank's.

#include <op_lib_core.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace op::gpu {

class Agreement {
public:
    static constexpr std::size_t slots = 1024;

    struct Slot {
        double not_ready = 1.0;
        double fused = -1.0;
        double apart = -1.0;
    };

private:
    std::vector<Slot> m_local = std::vector<Slot>(slots);
    std::vector<Slot> m_flight = std::vector<Slot>(slots);
    std::vector<Slot> m_agreed = std::vector<Slot>(slots);
    // The group using each slot here; a slot two groups share never fuses.
    std::vector<std::uint64_t> m_owner = std::vector<std::uint64_t>(slots, 0);
    std::vector<bool> m_shared = std::vector<bool>(slots, false);
    void *m_request = nullptr;
    bool m_in_flight = false;

    static std::size_t index(std::uint64_t hash) { return hash % slots; }

    // The group's slot here, claimed on first use; null once shared.
    Slot *claim(std::uint64_t hash) {
        auto i = index(hash);
        if (m_owner[i] == 0)
            m_owner[i] = hash;
        if (m_owner[i] != hash)
            m_shared[i] = true;
        if (m_shared[i]) {
            m_local[i].not_ready = 2.0;
            return nullptr;
        }
        return &m_local[i];
    }

public:
    // The group formed here: its slot is taken, or found shared.
    void seen(std::uint64_t hash) { claim(hash); }

    // This rank has the group's kernel, or never will.
    void ready(std::uint64_t hash, bool never) {
        if (auto *slot = claim(hash))
            slot->not_ready = never ? 2.0 : 0.0;
    }

    // This rank's medians for the group.
    void timed(std::uint64_t hash, double fused, double apart) {
        if (auto *slot = claim(hash)) {
            slot->fused = fused;
            slot->apart = apart;
        }
    }

    const Slot& agreed(std::uint64_t hash) const { return m_agreed[index(hash)]; }

    // At a collective flush: the round started at the last one becomes the
    // agreement, and a new one starts from this rank's slots.
    void round() {
        finish();
        m_flight = m_local;
        m_request = op_mpi_start_max(reinterpret_cast<double *>(m_flight.data()),
                                     static_cast<int>(3 * slots));
        m_in_flight = true;
    }

    // Finish the round in flight, if any.
    void finish() {
        if (!m_in_flight)
            return;
        op_mpi_finish_max(m_request);
        m_agreed = m_flight;
        m_in_flight = false;
    }
};

} // namespace op::gpu
