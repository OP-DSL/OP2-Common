/*
 * rcm.hpp
 *
 * Reverse Cuthill-McKee (RCM) node ordering.
 *
 * Header-only. Pure graph algorithm - no OP2 dependencies. Operates on a
 * CSR-format adjacency (row_offsets, col_indices) and produces a permutation
 * array with permutation[old_idx] = new_idx.
 *
 * Also exposes shared graph utilities used by the Sloan ordering:
 *   - bfs_levels
 *   - find_pseudo_peripheral
 * so sloan.hpp includes this header rather than duplicating them.
 */

#ifndef RCM_HPP
#define RCM_HPP

#include <vector>
#include <algorithm>
#include <climits>
#include <utility>

namespace op_renumber_impl {

// BFS from `start`, filling `level[]` with the BFS depth of each reached
// vertex (-1 for unreached vertices). Returns the eccentricity of `start`
// (the maximum level reached in `start`'s connected component).
inline int bfs_levels(const std::vector<int> &row_offsets,
                      const std::vector<int> &col_indices,
                      int num_verts,
                      int start,
                      std::vector<int> &level) {
  level.assign(num_verts, -1);
  std::vector<int> queue_vec;
  queue_vec.reserve(num_verts);
  queue_vec.push_back(start);
  level[start] = 0;
  int max_level = 0;
  size_t head = 0;
  while (head < queue_vec.size()) {
    int u = queue_vec[head++];
    int Lu = level[u];
    for (int e = row_offsets[u]; e < row_offsets[u + 1]; e++) {
      int v = col_indices[e];
      if (level[v] == -1) {
        level[v] = Lu + 1;
        if (Lu + 1 > max_level) max_level = Lu + 1;
        queue_vec.push_back(v);
      }
    }
  }
  return max_level;
}

// Pseudo-peripheral node finder (George-Liu style). Repeatedly performs
// rooted level structures; each iteration picks a minimum-degree node from
// the deepest level and BFSs again. Stops when the eccentricity no longer
// strictly increases, or after a small cap on iterations.
// Because `bfs_levels` only labels vertices reachable from `start`, the
// search is naturally restricted to `start_hint`'s connected component.
inline int find_pseudo_peripheral(const std::vector<int> &row_offsets,
                                  const std::vector<int> &col_indices,
                                  int num_verts,
                                  int start_hint) {
  int start = start_hint;
  std::vector<int> level;
  int depth = bfs_levels(row_offsets, col_indices, num_verts, start, level);

  const int max_iters = 10;
  for (int iter = 0; iter < max_iters; iter++) {
    int best = start;
    int best_deg = row_offsets[start + 1] - row_offsets[start];
    for (int i = 0; i < num_verts; i++) {
      if (level[i] == depth) {
        int deg = row_offsets[i + 1] - row_offsets[i];
        if (deg < best_deg) {
          best_deg = deg;
          best = i;
        }
      }
    }
    if (best == start) break;

    std::vector<int> new_level;
    int new_depth = bfs_levels(row_offsets, col_indices, num_verts, best, new_level);
    if (new_depth > depth) {
      start = best;
      depth = new_depth;
      level.swap(new_level);
    } else {
      break;
    }
  }
  return start;
}

// Cuthill-McKee BFS over the full graph. Handles disconnected components by
// repeating from a fresh minimum-degree hint in each unvisited component.
// `order[k]` is the k-th vertex in pre-reverse CM order.
inline void cuthill_mckee_order(const std::vector<int> &row_offsets,
                                const std::vector<int> &col_indices,
                                int num_verts,
                                std::vector<int> &order) {
  order.clear();
  order.reserve(num_verts);
  std::vector<char> visited(num_verts, 0);
  std::vector<std::pair<int, int> > neighbours;
  std::vector<int> queue_vec;

  while ((int)order.size() < num_verts) {
    int hint = -1;
    int min_deg = INT_MAX;
    for (int i = 0; i < num_verts; i++) {
      if (!visited[i]) {
        int deg = row_offsets[i + 1] - row_offsets[i];
        if (deg < min_deg) {
          min_deg = deg;
          hint = i;
        }
      }
    }
    if (hint == -1) break;

    int start = find_pseudo_peripheral(row_offsets, col_indices, num_verts, hint);

    queue_vec.clear();
    queue_vec.push_back(start);
    visited[start] = 1;
    size_t head = 0;
    while (head < queue_vec.size()) {
      int u = queue_vec[head++];
      order.push_back(u);
      neighbours.clear();
      for (int e = row_offsets[u]; e < row_offsets[u + 1]; e++) {
        int v = col_indices[e];
        if (!visited[v]) {
          visited[v] = 1;
          int deg = row_offsets[v + 1] - row_offsets[v];
          neighbours.push_back(std::make_pair(deg, v));
        }
      }
      std::sort(neighbours.begin(), neighbours.end());
      for (size_t k = 0; k < neighbours.size(); k++) {
        queue_vec.push_back(neighbours[k].second);
      }
    }
  }
}

// Reverse Cuthill-McKee. `permutation[old_idx] = new_idx`.
inline void rcm_order(const std::vector<int> &row_offsets,
                      const std::vector<int> &col_indices,
                      int num_verts,
                      std::vector<int> &permutation) {
  std::vector<int> order;
  cuthill_mckee_order(row_offsets, col_indices, num_verts, order);
  permutation.assign(num_verts, -1);
  int N = (int)order.size();
  for (int k = 0; k < N; k++) {
    permutation[order[k]] = N - 1 - k;
  }
}

} // namespace op_renumber_impl

#endif // RCM_HPP
