/*
 * sloan.hpp
 *
 * Sloan's profile-minimising node ordering.
 *
 * Header-only. Pure graph algorithm - no OP2 dependencies. Operates on a
 * CSR-format adjacency (row_offsets, col_indices) and produces a permutation
 * array with permutation[old_idx] = new_idx.
 *
 * Sloan (1986, 1989) targets profile (envelope) rather than maximum
 * bandwidth. Each vertex v in the priority queue has
 *     P(v) = W1 * d(v, e) - W2 * cdeg(v)
 * where d(v, e) is the BFS distance from v to the "end" vertex e and
 * cdeg(v) is 1 plus the number of v's not-yet-numbered neighbours (the
 * current contribution to the active front if v were numbered next). The
 * maximum-priority vertex is numbered next; cdeg updates propagate W2
 * increments to neighbours as each vertex becomes postactive.
 *
 * Weights (W1, W2) = (2, 1) follow Sloan's original recommendation and
 * work well on typical FE/FV meshes.
 *
 * Depends on rcm.hpp for bfs_levels and find_pseudo_peripheral.
 */

#ifndef SLOAN_HPP
#define SLOAN_HPP

#include "rcm.hpp"

#include <vector>
#include <algorithm>
#include <climits>
#include <queue>
#include <utility>

namespace op_renumber_impl {

// Sloan's ordering. `permutation[old_idx] = new_idx`.
inline void sloan_order(const std::vector<int> &row_offsets,
                        const std::vector<int> &col_indices,
                        int num_verts,
                        std::vector<int> &permutation) {
  permutation.assign(num_verts, -1);

  // Sloan weights: distance weight and current-degree weight.
  const int W1 = 2;
  const int W2 = 1;

  enum { INACTIVE = 0, PREACTIVE = 1, ACTIVE = 2, POSTACTIVE = 3 };

  std::vector<char> status(num_verts, (char)INACTIVE);
  std::vector<int> cdeg(num_verts, 0);
  std::vector<long long> priority(num_verts, 0);
  std::vector<int> d;

  int next_index = 0;

  // Outer loop handles disconnected components.
  while (next_index < num_verts) {
    // Pick a starting hint: minimum-degree still-unnumbered vertex.
    int hint = -1;
    int min_deg = INT_MAX;
    for (int i = 0; i < num_verts; i++) {
      if (status[i] != POSTACTIVE) {
        int deg = row_offsets[i + 1] - row_offsets[i];
        if (deg < min_deg) {
          min_deg = deg;
          hint = i;
        }
      }
    }
    if (hint == -1) break;

    // Find a pseudo-peripheral pair (s, e). One pseudo-peripheral pass
    // from the hint yields `e`; a second pass from `e` yields `s`.
    int e = find_pseudo_peripheral(row_offsets, col_indices, num_verts, hint);
    int s = find_pseudo_peripheral(row_offsets, col_indices, num_verts, e);

    // BFS distances from the end vertex. Vertices outside this component
    // have d[v] == -1 and will be picked up by the next outer iteration.
    bfs_levels(row_offsets, col_indices, num_verts, e, d);

    // Initialise cdeg for vertices in this component.
    for (int v = 0; v < num_verts; v++) {
      if (d[v] >= 0 && status[v] != POSTACTIVE)
        cdeg[v] = row_offsets[v + 1] - row_offsets[v] + 1;
    }

    // Lazy-deletion max-heap: we never remove; we push a fresh (priority,
    // vertex) entry whenever priority rises, and discard popped entries
    // whose priority no longer matches the current value, or whose vertex
    // is already postactive. Priority only increases for non-postactive
    // vertices (cdeg only decreases), so this is correct.
    std::priority_queue<std::pair<long long, int> > pq;

    status[s] = (char)PREACTIVE;
    priority[s] = (long long)W1 * d[s] - (long long)W2 * cdeg[s];
    pq.push(std::make_pair(priority[s], s));

    while (!pq.empty()) {
      std::pair<long long, int> top = pq.top();
      pq.pop();
      int v = top.second;

      if (status[v] == POSTACTIVE) continue;    // already numbered
      if (priority[v] != top.first) continue;   // stale entry

      // Number v.
      status[v] = (char)POSTACTIVE;
      permutation[v] = next_index++;

      // Walk v's neighbours.
      for (int e_idx = row_offsets[v]; e_idx < row_offsets[v + 1]; e_idx++) {
        int u = col_indices[e_idx];
        if (status[u] == POSTACTIVE) continue;

        char old_status = status[u];

        if (old_status == INACTIVE) {
          // Newly discovered: set baseline priority as if v were still
          // non-postactive. The +W2 increment below then brings it to the
          // correct value (v has just become postactive, so cdeg -= 1).
          status[u] = (char)PREACTIVE;
          priority[u] = (long long)W1 * d[u] - (long long)W2 * cdeg[u];
        }

        // v is becoming postactive - u has one less non-postactive neighbour.
        cdeg[u] -= 1;
        priority[u] += W2;
        pq.push(std::make_pair(priority[u], u));

        if (old_status == PREACTIVE) {
          // u was preactive and has now gained its first postactive
          // neighbour (v), so promote to active. Two-step expansion:
          // add u's still-inactive neighbours to the queue.
          status[u] = (char)ACTIVE;
          for (int e2 = row_offsets[u]; e2 < row_offsets[u + 1]; e2++) {
            int w = col_indices[e2];
            if (status[w] == INACTIVE) {
              status[w] = (char)PREACTIVE;
              // w has no postactive neighbour (else it would already be
              // in the queue) so cdeg[w] is unchanged = deg(w) + 1.
              priority[w] = (long long)W1 * d[w] - (long long)W2 * cdeg[w];
              pq.push(std::make_pair(priority[w], w));
            }
          }
        }
      }
    }
  }
}

} // namespace op_renumber_impl

#endif // SLOAN_HPP
