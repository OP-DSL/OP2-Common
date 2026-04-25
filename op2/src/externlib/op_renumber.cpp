/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * op_renumber.cpp
 *
 * Alternative renumbering entry point. Dispatches to one of several node
 * ordering algorithms based on the OP_REORDER environment variable:
 *
 *     OP_REORDER=none      - no reordering (default)
 *     OP_REORDER=random    - random permutation (benchmark baseline)
 *     OP_REORDER=rcm       - Reverse Cuthill-McKee
 *     OP_REORDER=sloan     - Sloan profile-minimising ordering
 *     OP_REORDER=hilbert   - Hilbert space-filling curve (needs coords)
 *
 * Once the primary set's permutation is computed, OP_REORDER_PROPAGATE
 * controls how that permutation is extended to other sets reachable via
 * maps (e.g., reordering edges after nodes have been reordered):
 *
 *     OP_REORDER_PROPAGATE=lex       - multi-key lex sort over all map
 *                                      dimensions (default; strict win
 *                                      over single-key for any dim>1 map)
 *     OP_REORDER_PROPAGATE=centroid  - Hilbert SFC of stencil centroid;
 *                                      centroids are averaged from the
 *                                      already-ordered parent set, so
 *                                      this requires a coordinates dat
 *                                      on the primary set
 *     OP_REORDER_PROPAGATE=single    - legacy single-dim sort by first
 *                                      map endpoint only (kept for
 *                                      benchmarking and regression)
 *
 * The adjacency-graph construction, permutation propagation and physical
 * re-application logic are kept identical in spirit to op_renumber.cpp.
 * The ordering algorithms themselves live in header-only modules:
 *
 *     op_renumber_rcm.hpp          - RCM + shared graph utilities
 *     op_renumber_sloan.hpp        - Sloan (depends on rcm.hpp)
 *     op_renumber_hilbert_sfc.hpp  - Hilbert SFC (standalone, geometric)
 *
 * No external library dependency is required; this translation unit does
 * not need HAVE_PTSCOTCH.
 */




#include <op_lib_core.h>
#include <op_lib_cpp.h>
#include <op_util.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <climits>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <random>
#include <op_lib_mpi.h>
#include <op_mpi_core.h>
#include <op_mpi_halo.h>

#include "renumber/rcm.hpp"
#include "renumber/sloan.hpp"
#include "renumber/hilbert_sfc.hpp"

using op::mpi::HaloList;

typedef struct {
  int a;
  int b;
} map2;

static int compare(const void *a, const void *b) {
  return ((*(map2 *)a).a - (*(map2 *)b).a);
}

static void check_permutation(int *perm, int size) {
  std::vector<int> flags(size, 0);
  for (int i = 0; i < size; i++)
    flags[perm[i]] = 1;
  int acc = 0;
  for (int i = 0; i < size; i++)
    acc += flags[i];
  if (acc != size) printf("Permutation map error\n");
}

//-----------------------------------------------------------------------------
// Permutation propagation and physical application (unchanged from the
// RCM-only version of this file; kept here to avoid spreading OP2 internals
// across multiple translation units).
//-----------------------------------------------------------------------------

// Propagation strategy for non-primary sets reached via a map from a set
// that has already been reordered. Selectable via OP_REORDER_PROPAGATE.
//
//  Single   - sort `to`-set elements by the new index of their first map
//             endpoint only (qsort, not stable). Original behaviour; kept
//             for benchmarking. Within each first-endpoint bucket the
//             remaining endpoints are in arbitrary order.
//  Lex      - sort `to`-set elements lexicographically by all map endpoint
//             new-indices. Within each first-endpoint bucket the second
//             endpoint is sorted, the third within that, etc. Improves
//             cache reuse and GPU coalescing on the second+ endpoint of
//             every dim>1 map. Default.
//  Centroid - compute a geometric centroid for each `to`-set element by
//             averaging the parent set's centroids (or coords, for the
//             primary set), then reorder via Hilbert SFC on those
//             centroids. Symmetric across all map dimensions; requires
//             a coordinates dat on the primary set.
enum class PropagateMethod {
  Single,
  Lex,
  Centroid,
};

// Per-call state carried through propagate_reordering recursion.
struct PropagationContext {
  PropagateMethod method;
  int coord_dim;  // 2 or 3 if centroids are available, 0 otherwise.
  // set_centroids[s] is a flat AoS array of length core_size * coord_dim
  // for set s, in original (pre-permutation) element order. Empty for
  // sets that have no centroid available (either coord_dim is 0, or the
  // set was reordered through a path that didn't propagate centroids).
  std::vector<std::vector<double> > set_centroids;
};

// Lex multi-key order: returns indices [0, n) sorted such that for k < k',
// the tuple (perm[map[order[k]*dim+0]], ..., perm[map[order[k]*dim+dim-1]])
// is lexicographically less than the corresponding tuple for order[k'].
// Ties are broken by original index for reproducibility.
static std::vector<int> compute_lex_order(op_map map,
                                          const std::vector<int> &fperm,
                                          int n_elems) {
  std::vector<int> order(n_elems);
  for (int i = 0; i < n_elems; i++) order[i] = i;

  const int dim = map->dim;
  const int *mp = map->map;
  std::sort(order.begin(), order.end(),
            [dim, &fperm, mp](int x, int y) {
              for (int d = 0; d < dim; d++) {
                int kx = fperm[mp[dim * x + d]];
                int ky = fperm[mp[dim * y + d]];
                if (kx != ky) return kx < ky;
              }
              return x < y;
            });
  return order;
}

// Compute centroids for the from-set of `map` by averaging the to-set's
// centroids over the map dimensions. Used by case 1 of propagate_reordering,
// where the to-set ('parent') is already ordered and has centroids.
//
// Returns true on success, false if parent centroids aren't available.
// Halo entries in the map (parent index >= parent core_size) are skipped:
// boundary elements average over fewer parents, which biases their centroid
// slightly inward. That's harmless for SFC ordering.
static bool compute_centroids_to_from(op_map map, PropagationContext &pctx) {
  if (pctx.coord_dim == 0) return false;
  op_set child = map->from;
  op_set parent = map->to;
  const std::vector<double> &pc = pctx.set_centroids[parent->index];
  if (pc.empty()) return false;

  const int cd = pctx.coord_dim;
  const int n = child->core_size;
  const int parent_n = (int)(pc.size() / cd);

  std::vector<double> &out = pctx.set_centroids[child->index];
  out.assign((size_t)n * cd, 0.0);

  for (int i = 0; i < n; i++) {
    int valid = 0;
    for (int d = 0; d < map->dim; d++) {
      int p = map->map[i * map->dim + d];
      if (p >= 0 && p < parent_n) {
        for (int c = 0; c < cd; c++)
          out[(size_t)i * cd + c] += pc[(size_t)p * cd + c];
        valid++;
      }
    }
    if (valid > 0) {
      for (int c = 0; c < cd; c++)
        out[(size_t)i * cd + c] /= (double)valid;
    }
  }
  return true;
}

// Compute centroids for the to-set of `map` by accumulating from-centroids
// from every from-element that references each to-element. Used by case 2
// of propagate_reordering. The to-set's actual ordering comes from the
// existing first-touch counter logic; centroids are computed alongside
// purely so that descendants of this to-set can use centroid mode.
static bool compute_centroids_from_to(op_map map, PropagationContext &pctx) {
  if (pctx.coord_dim == 0) return false;
  op_set parent = map->from;
  op_set child = map->to;
  const std::vector<double> &pc = pctx.set_centroids[parent->index];
  if (pc.empty()) return false;

  const int cd = pctx.coord_dim;
  const int n = child->core_size;
  const int parent_n = (int)(pc.size() / cd);

  std::vector<double> &out = pctx.set_centroids[child->index];
  out.assign((size_t)n * cd, 0.0);
  std::vector<int> touch(n, 0);

  for (int i = 0; i < parent->size && i < parent_n; i++) {
    for (int d = 0; d < map->dim; d++) {
      int t = map->map[i * map->dim + d];
      if (t >= 0 && t < n) {
        for (int c = 0; c < cd; c++)
          out[(size_t)t * cd + c] += pc[(size_t)i * cd + c];
        touch[t]++;
      }
    }
  }
  for (int i = 0; i < n; i++) {
    if (touch[i] > 0) {
      for (int c = 0; c < cd; c++)
        out[(size_t)i * cd + c] /= (double)touch[i];
    }
  }
  return true;
}

// Propagate renumbering based on a map that points to an already reordered set.
static void propagate_reordering(op_set from, op_set to,
                                 PropagationContext &pctx,
                                 std::vector<std::vector<int> > &set_permutations,
                                 std::vector<std::vector<int> > &set_ipermutations) {

  if (to->size == 0)
    return;

  // find a map that is (to)->(from), reorder (to)
  if (set_permutations[to->index].size() == 0) {
    for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
      op_map map = OP_map_list[mapidx];
      if (map->to == from && map->from == to) {
        const int n = to->core_size;
        const int total = to->size + to->exec_size + to->nonexec_size;
        set_permutations[to->index].resize(total);

        // --- Centroid mode (preferred when available). -----------------
        bool ordered_via_centroid = false;
        if (pctx.method == PropagateMethod::Centroid) {
          if (compute_centroids_to_from(map, pctx)) {
            std::vector<int> perm;
            op_renumber_impl::hilbert_sfc_order(
                pctx.set_centroids[to->index].data(), n, pctx.coord_dim, perm);
            for (int i = 0; i < n; i++)
              set_permutations[to->index][i] = perm[i];
            ordered_via_centroid = true;
          } else {
            // Centroid requested but parent has none; fall through to lex.
          }
        }

        // --- Lex mode (default) and Single mode (legacy benchmarking). -
        if (!ordered_via_centroid) {
          if (pctx.method == PropagateMethod::Single) {
            std::vector<map2> renum(n);
            for (int i = 0; i < n; i++) {
              renum[i].a = set_permutations[from->index][map->map[map->dim * i]];
              renum[i].b = i;
            }
            qsort(&renum[0], renum.size(), sizeof(map2), compare);
            for (int i = 0; i < n; i++)
              set_permutations[to->index][renum[i].b] = i;
          } else {
            // Lex (default), also fallback path for centroid-without-coords.
            std::vector<int> order = compute_lex_order(
                map, set_permutations[from->index], n);
            for (int i = 0; i < n; i++)
              set_permutations[to->index][order[i]] = i;
          }
          // Best-effort: still try to compute centroids for descendants
          // even if we used a non-centroid order at this level. Cheap and
          // makes deeper sets in the tree usable in centroid mode.
          if (pctx.method == PropagateMethod::Centroid && pctx.coord_dim > 0) {
            compute_centroids_to_from(map, pctx);
          }
        }

        for (int i = n; i < total; i++)
          set_permutations[to->index][i] = i;
        check_permutation(&set_permutations[to->index][0], total);
        break;
      }
    }
  }
  // find a map that is (from)->(to), reorder (to), if it's an onto map
  if (set_permutations[to->index].size() == 0) {
    for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
      op_map map = OP_map_list[mapidx];
      if (map->to == to && map->from == from) {
        int counter = 0;
        set_permutations[to->index].resize(to->size + to->exec_size + to->nonexec_size, -1);
        for (int i = 0; i < from->size; i++) {
          for (int d = 0; d < map->dim; d++) {
            int idx = map->map[set_ipermutations[from->index][i] * map->dim + d];
            if (idx < to->core_size && set_permutations[to->index][idx] == -1)
              set_permutations[to->index][idx] = counter++;
          }
        }
        int onto = 1;
        for (int i = 0; i < to->core_size; i++)
          if (set_permutations[to->index][i] == -1) { onto = 0; break; }
        if (!onto) {
          set_permutations[to->index].resize(0);
          continue;
        }
        for (int i = to->core_size; i < to->size + to->exec_size + to->nonexec_size; i++)
          set_permutations[to->index][i] = i;
        check_permutation(&set_permutations[to->index][0],
                          to->size + to->exec_size + to->nonexec_size);

        // Propagate centroids alongside the first-touch ordering so that
        // descendants of this set can still use centroid mode. The
        // ordering itself is unchanged from the legacy first-touch logic
        // - case 2's traversal already inherits locality from the
        // already-reordered `from` set.
        if (pctx.method == PropagateMethod::Centroid && pctx.coord_dim > 0) {
          compute_centroids_from_to(map, pctx);
        }
        break;
      }
    }
  }
  if (set_permutations[to->index].size() == 0) {
    return;
  }
  else {
    set_ipermutations[to->index].resize(to->size + to->exec_size + to->nonexec_size);
    for (int i = 0; i < to->size + to->exec_size + to->nonexec_size; i++) {
      set_ipermutations[to->index][set_permutations[to->index][i]] = i;
    }
  }

  // find any maps that is (*)->to, propagate reordering
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->to == to && set_permutations[map->from->index].size() == 0) {
      propagate_reordering(to, map->from, pctx, set_permutations, set_ipermutations);
    }
  }
  // find any maps that is to->(*), propagate reordering
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->from == to && set_permutations[map->to->index].size() == 0) {
      propagate_reordering(to, map->to, pctx, set_permutations, set_ipermutations);
    }
  }
}

static void reorder_set(op_set set, std::vector<std::vector<int> > &set_permutations) {

  if (set_permutations[set->index].size() == 0 && set->core_size > 0) {
    printf("No reordering for set %s, skipping...\n", set->name);
    return;
  }

  if (set->size == 0)
    return;

  // Reorder maps
  for (int mapidx = 0; mapidx < OP_map_index; mapidx++) {
    op_map map = OP_map_list[mapidx];
    if (map->from == set) {
      int *tempmap = (int *)malloc((set->size + set->exec_size) * sizeof(int) * map->dim);

      for (int i = 0; i < set->size + set->exec_size; i++)
        std::copy(map->map + map->dim * i, map->map + map->dim * (i + 1),
                  tempmap + map->dim * set_permutations[set->index][i]);
      free(map->map);
      map->map = tempmap;

    } else if (map->to == set) {
      for (int i = 0; i < (map->from->size + map->from->exec_size) * map->dim; i++)
        map->map[i] = set_permutations[set->index][map->map[i]];
    }
  }

  // Reorder datasets
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (dat->set == set && dat->data != NULL) {
      char *tempdata = (char *)malloc((size_t)(set->size + set->exec_size + set->nonexec_size) *
                                      (size_t)dat->size);
      for (unsigned long int i = 0;
           i < (unsigned long int)(set->size + set->exec_size + set->nonexec_size); i++)
        std::copy(dat->data + (unsigned long int)dat->size * i,
                  dat->data + (unsigned long int)dat->size * (i + 1),
                  tempdata +
                      (unsigned long int)dat->size *
                          (unsigned long int)set_permutations[set->index][i]);
      free(dat->data);
      dat->data = tempdata;
    }
  }

  // Renumber halos: this set's export lists, and the partial-exchange export
  // lists of every map onto it. The import lists on other ranks are refreshed
  // once every set is done (op_halo_refresh_imports).
  std::vector<HaloList *> exports = {&OP_set_halos[set->index].export_exec,
                                     &OP_set_halos[set->index].export_nonexec};
  for (int m = 0; m < (int)OP_map_halos.size(); m++)
    if (OP_map_list[m]->to == set)
      exports.push_back(&OP_map_halos[m].export_nonexec);
  for (HaloList *exp : exports)
    for (idx_l_t i = 0; i < exp->size(); i++)
      exp->list[i] = set_permutations[set->index][exp->list[i]];

  // Reorder mapping back to original (unpartitioned indexing)
  idx_g_t *new_g_index = (idx_g_t *)malloc(set->size * sizeof(idx_g_t));
  for (int i = 0; i < set->size; i++)
    new_g_index[set_permutations[set->index][i]] = OP_part_list[set->index]->g_index[i];
  free(OP_part_list[set->index]->g_index);
  OP_part_list[set->index]->g_index = new_g_index;
}

//-----------------------------------------------------------------------------
// OP_REORDER dispatch
//-----------------------------------------------------------------------------

enum class ReorderMethod {
  None,
  Random,
  RCM,
  Sloan,
  Hilbert
};

static int str_iequals(const char *a, const char *b) {
  while (*a && *b) {
    if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return 0;
    a++; b++;
  }
  return *a == 0 && *b == 0;
}

static const char *method_name(ReorderMethod m) {
  switch (m) {
    case ReorderMethod::None:    return "none";
    case ReorderMethod::Random:  return "random";
    case ReorderMethod::RCM:     return "RCM";
    case ReorderMethod::Sloan:   return "Sloan";
    case ReorderMethod::Hilbert: return "Hilbert SFC";
  }
  return "unknown";
}

static ReorderMethod get_reorder_method() {
  const char *env = getenv("OP_REORDER");
  if (env == NULL || env[0] == '\0') return ReorderMethod::None;
  if (str_iequals(env, "none"))    return ReorderMethod::None;
  if (str_iequals(env, "random"))  return ReorderMethod::Random;
  if (str_iequals(env, "rcm"))     return ReorderMethod::RCM;
  if (str_iequals(env, "sloan"))   return ReorderMethod::Sloan;
  if (str_iequals(env, "hilbert")) return ReorderMethod::Hilbert;
  op_printf("Warning: unknown OP_REORDER value '%s', defaulting to none\n", env);
  return ReorderMethod::None;
}

static const char *propagate_method_name(PropagateMethod m) {
  switch (m) {
    case PropagateMethod::Single:   return "single";
    case PropagateMethod::Lex:      return "lex";
    case PropagateMethod::Centroid: return "centroid";
  }
  return "unknown";
}

static PropagateMethod get_propagate_method() {
  const char *env = getenv("OP_REORDER_PROPAGATE");
  if (env == NULL || env[0] == '\0') return PropagateMethod::Lex;
  if (str_iequals(env, "single") ||
      str_iequals(env, "legacy"))      return PropagateMethod::Single;
  if (str_iequals(env, "lex") ||
      str_iequals(env, "multikey"))    return PropagateMethod::Lex;
  if (str_iequals(env, "centroid") ||
      str_iequals(env, "hilbert"))     return PropagateMethod::Centroid;
  op_printf("Warning: unknown OP_REORDER_PROPAGATE value '%s', defaulting to lex\n", env);
  return PropagateMethod::Lex;
}

// Fisher-Yates shuffle of the identity permutation. Fixed seed so that
// benchmark runs are reproducible across invocations.
static void random_order(int num_verts, std::vector<int> &permutation) {
  permutation.resize(num_verts);
  for (int i = 0; i < num_verts; i++) permutation[i] = i;
  std::mt19937 rng(42u);
  for (int i = num_verts - 1; i > 0; i--) {
    std::uniform_int_distribution<int> dist(0, i);
    int j = dist(rng);
    std::swap(permutation[i], permutation[j]);
  }
}

// Heuristic search for a coordinates dat on the given node set. Preference:
//   1. an op_dat on `node_set` whose name contains "coord", "pos", matches
//      "x"/"X", or starts with "p_x"/"p_X", with dim in {2,3} and type double.
//   2. any op_dat on `node_set` with dim in {2,3} and type double.
// Returns NULL if nothing plausible is found.
static op_dat find_coords_dat(op_set node_set) {
  op_dat_entry *item;

  // Pass 1: name-based match.
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (dat->set != node_set) continue;
    if (dat->dim != 2 && dat->dim != 3) continue;
    if (dat->data == NULL) continue;
    if (dat->type == NULL || strcmp(dat->type, "double") != 0) continue;
    const char *name = dat->name ? dat->name : "";
    if (strstr(name, "coord") || strstr(name, "Coord") ||
        strstr(name, "pos")   || strstr(name, "Pos")   ||
        strcmp(name, "x") == 0 || strcmp(name, "X") == 0 ||
        strncmp(name, "p_x", 3) == 0 || strncmp(name, "p_X", 3) == 0) {
      return dat;
    }
  }

  // Pass 2: any 2D/3D double dat on this set.
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (dat->set != node_set) continue;
    if (dat->dim != 2 && dat->dim != 3) continue;
    if (dat->data == NULL) continue;
    if (dat->type == NULL || strcmp(dat->type, "double") != 0) continue;
    return dat;
  }

  return NULL;
}

//-----------------------------------------------------------------------------
// Per-edge bandwidth / locality statistic: the maximum pairwise distance
// between endpoint indices of each edge, averaged and maxed over edges.
//-----------------------------------------------------------------------------
static void compute_edge_stats(op_map base, int &max_dist, long &avg_dist) {
  max_dist = 0;
  avg_dist = 0;
  if (base->from->size == 0) return;
  for (int i = 0; i < base->from->size; i++) {
    int dist = 0;
    for (int d1 = 0; d1 < base->dim; d1++)
      for (int d2 = 0; d2 < base->dim; d2++)
        dist = std::max(dist, std::abs(base->map[i * base->dim + d1] -
                                       base->map[i * base->dim + d2]));
    max_dist = std::max(max_dist, dist);
    avg_dist += dist;
  }
  avg_dist /= base->from->size;
}

//-----------------------------------------------------------------------------
// CSR adjacency build (same structure as op_renumber.cpp's local build,
// extracted into a helper since RCM and Sloan both need it).
//-----------------------------------------------------------------------------
static bool build_node_adjacency(op_map base,
                                 std::vector<int> &row_offsets,
                                 std::vector<int> &col_indices) {
  row_offsets.assign(base->to->core_size + 1, 0);
  col_indices.clear();

  if (base->to == base->from) {
    // Self-referencing map: adjacency is already in `base` directly; just
    // drop references that fall outside the core-size range.
    col_indices.resize(base->dim * base->from->size);
    row_offsets[0] = 0;
    for (int i = 0; i < base->from->size; i++) {
      int rowlen = 0;
      for (int j = 0; j < base->dim; j++)
        if (base->map[i * base->dim + j] < base->to->core_size)
          col_indices[row_offsets[i] + rowlen++] = base->map[i * base->dim + j];
      row_offsets[i + 1] = row_offsets[i] + rowlen;
    }
    col_indices.resize(row_offsets[base->to->core_size]);
    return true;
  }

  // Build self-referencing node->node map from an edge->node base map.
  col_indices.resize(base->from->size * (base->dim - 1) * (base->dim));

  std::vector<map2> loopback(base->from->size * base->dim);
  int sizectr = 0;
  for (int i = 0; i < base->from->size; i++) {
    for (int j = 0; j < base->dim; j++) {
      if (base->map[i * base->dim + j] < base->to->core_size) {
        loopback[sizectr].a = base->map[i * base->dim + j];
        loopback[sizectr].b = i;
        sizectr++;
      }
    }
  }

  loopback.resize(sizectr);
  qsort(&loopback[0], loopback.size(), sizeof(map2), compare);

  row_offsets[0] = 0;
  row_offsets[1] = 0;
  row_offsets[base->to->core_size] = 0;
  for (int i = 0; i < base->dim; i++) {
    if (base->map[base->dim * loopback[0].b + i] != 0 &&
        base->map[base->dim * loopback[0].b + i] < base->to->core_size)
      col_indices[row_offsets[1]++] =
          base->map[base->dim * loopback[0].b + i];
  }
  int nodectr = 0;
  for (int i = 1; i < (int)loopback.size(); i++) {
    if (loopback[i].a != loopback[i - 1].a) {
      nodectr++;
      row_offsets[nodectr + 1] = row_offsets[nodectr];
    }

    for (int d1 = 0; d1 < base->dim; d1++) {
      int id = base->map[base->dim * loopback[i].b + d1];
      int add = (id != nodectr && id < base->to->core_size);
      for (int d2 = row_offsets[nodectr];
           (d2 < row_offsets[nodectr + 1]) && add; d2++) {
        if (col_indices[d2] == id)
          add = 0;
      }
      if (add)
        col_indices[row_offsets[nodectr + 1]++] = id;
    }
  }
  if (row_offsets[base->to->core_size] == 0) {
    printf(
        "Map %s is not an onto map from %s to %s, or bad partitioning, aborting renumbering...\n",
        base->name, base->from->name, base->to->name);
    return false;
  }
  col_indices.resize(row_offsets[base->to->core_size]);
  if (OP_diags > 2)
    op_printf("Loopback map %s->%s constructed: %d, from set %s (%d)\n",
              base->to->name, base->to->name, (int)col_indices.size(),
              base->from->name, base->from->size);

  // Sanity check: graph rows and symmetry.
  for (int row = 0; row < (int)row_offsets.size() - 1; row++) {
    if (row_offsets[row] == row_offsets[row + 1]) printf("Zero length row\n");
    for (int col = row_offsets[row]; col < row_offsets[row + 1]; col++) {
      if (col_indices[col] < 0 || col_indices[col] >= (int)row_offsets.size() - 1)
        printf("Error col idx %d, but num rows is %lu\n", col_indices[col],
               row_offsets.size() - 1);
      else {
        int found = 0;
        for (int c2 = row_offsets[col_indices[col]];
             c2 < row_offsets[col_indices[col] + 1]; c2++) {
          if (col_indices[c2] == row) found = 1;
        }
        if (!found) printf("Error, symmetry broken at row %d col %d\n", row, col_indices[col]);
      }
    }
  }
  return true;
}

//-----------------------------------------------------------------------------
// Public entry point
//-----------------------------------------------------------------------------

/* Reorder this rank's core elements of base's target set, and of every set the
   ordering propagates to. It can give up on one rank alone (a core with no
   edges of its own), so nothing collective may happen in here. */
static void renumber_owned(op_map base, ReorderMethod method, PropagateMethod propagate) {
  op_printf("Renumbering (%s) using base map %s\n", method_name(method), base->name);

  int num_verts = base->to->core_size;
  if (num_verts == 0) {
    op_printf("op_renumber: core_size is zero on set %s, nothing to do\n",
              base->to->name);
    return;
  }

  // Locate a coordinates dat on the primary set - needed by Hilbert SFC for
  // the primary ordering, and by centroid propagation regardless of which
  // primary method was selected.
  op_dat coords_dat = find_coords_dat(base->to);

  //---------------------------------------------------------------------------
  // Compute the core-size permutation via the selected algorithm.
  //---------------------------------------------------------------------------
  std::vector<int> permutation;

  if (method == ReorderMethod::Random) {
    random_order(num_verts, permutation);

  } else if (method == ReorderMethod::Hilbert) {
    if (coords_dat == NULL) {
      op_printf("ERROR: Hilbert SFC requires a double-precision 2D/3D "
                "coordinates dat on set %s, but none was found. "
                "Aborting renumbering.\n", base->to->name);
      return;
    }
    op_printf("op_renumber: using dat '%s' (dim %d) as Hilbert SFC coordinates\n",
              coords_dat->name ? coords_dat->name : "<unnamed>", coords_dat->dim);

    op_renumber_impl::hilbert_sfc_stats hstats;
    op_renumber_impl::hilbert_sfc_order(
        reinterpret_cast<const double *>(coords_dat->data),
        num_verts, coords_dat->dim, permutation, &hstats);

    // Diagnostic log: grid resolution, quantisation collisions, per-axis
    // effective bits. A large gap between num_verts and distinct_indices
    // means many vertices collapsed into the same Hilbert bin and the
    // relative ordering within each bin was decided only by vertex id -
    // locality quality degrades correspondingly.
    double ratio = (hstats.num_verts > 0)
                       ? (double)hstats.distinct_indices /
                             (double)hstats.num_verts
                       : 1.0;
    op_printf("op_renumber: Hilbert SFC quantisation at %d bits/axis\n",
              hstats.nominal_bits);
    op_printf("  num_verts         = %d\n", hstats.num_verts);
    op_printf("  distinct_indices  = %d (%.4f of num_verts)\n",
              hstats.distinct_indices, ratio);
    if (hstats.dim == 2) {
      op_printf("  axis ranges       = [%.3g, %.3g]\n",
                hstats.axis_range[0], hstats.axis_range[1]);
      op_printf("  effective bits    = [%.2f, %.2f]\n",
                hstats.effective_bits[0], hstats.effective_bits[1]);
    } else {
      op_printf("  axis ranges       = [%.3g, %.3g, %.3g]\n",
                hstats.axis_range[0], hstats.axis_range[1],
                hstats.axis_range[2]);
      op_printf("  effective bits    = [%.2f, %.2f, %.2f]\n",
                hstats.effective_bits[0], hstats.effective_bits[1],
                hstats.effective_bits[2]);
    }
    if (ratio < 0.99) {
      op_printf("WARNING: Hilbert SFC has %.2f%% collision rate - %d of %d "
                "vertices share a bin with another. Grid resolution may be "
                "insufficient for this mesh's anisotropy or point density; "
                "consider increasing bits-per-axis in "
                "op_renumber_hilbert_sfc.hpp.\n",
                100.0 * (1.0 - ratio),
                hstats.num_verts - hstats.distinct_indices,
                hstats.num_verts);
    }

  } else {
    // RCM and Sloan both need the CSR adjacency.
    std::vector<int> row_offsets, col_indices;
    if (!build_node_adjacency(base, row_offsets, col_indices)) {
      return; // error already printed
    }

    if (method == ReorderMethod::RCM) {
      op_renumber_impl::rcm_order(row_offsets, col_indices, num_verts, permutation);
    } else { // Sloan
      op_renumber_impl::sloan_order(row_offsets, col_indices, num_verts, permutation);
    }
  }

  //---------------------------------------------------------------------------
  // Pre-reordering statistics (computed on the original map so the "before"
  // numbers correspond to the ordering coming in from the partitioner).
  //---------------------------------------------------------------------------
  int max_dist_before;
  long avg_dist_before;
  compute_edge_stats(base, max_dist_before, avg_dist_before);

  //---------------------------------------------------------------------------
  // Assemble the full per-set permutation with identity on the halo range.
  //---------------------------------------------------------------------------
  std::vector<std::vector<int> > set_permutations(OP_set_index);
  std::vector<std::vector<int> > set_ipermutations(OP_set_index);

  int to_total = base->to->size + base->to->exec_size + base->to->nonexec_size;
  set_permutations[base->to->index].resize(to_total);
  for (int i = 0; i < num_verts; i++)
    set_permutations[base->to->index][i] = permutation[i];
  for (int i = num_verts; i < to_total; i++)
    set_permutations[base->to->index][i] = i;
  check_permutation(&set_permutations[base->to->index][0], to_total);

  set_ipermutations[base->to->index].resize(to_total);
  for (int i = 0; i < to_total; i++) {
    set_ipermutations[base->to->index][set_permutations[base->to->index][i]] = i;
  }

  //---------------------------------------------------------------------------
  // Set up the propagation context. In centroid mode, seed the primary set's
  // centroids from the coords dat (in original/pre-permutation order - the
  // helpers in propagate_reordering index map entries against this layout).
  //---------------------------------------------------------------------------
  PropagationContext pctx;
  pctx.method = propagate;
  pctx.coord_dim = 0;
  pctx.set_centroids.resize(OP_set_index);

  if (propagate == PropagateMethod::Centroid) {
    if (coords_dat == NULL) {
      op_printf("WARNING: OP_REORDER_PROPAGATE=centroid requires a "
                "coordinates dat on set %s, but none was found. "
                "Falling back to lex multi-key sort.\n", base->to->name);
      pctx.method = PropagateMethod::Lex;
    } else {
      pctx.coord_dim = coords_dat->dim;
      const double *cd = reinterpret_cast<const double *>(coords_dat->data);
      pctx.set_centroids[base->to->index].assign(
          cd, cd + (size_t)num_verts * coords_dat->dim);
      op_printf("op_renumber: centroid propagation seeded from dat '%s' "
                "(dim %d) on set %s\n",
                coords_dat->name ? coords_dat->name : "<unnamed>",
                coords_dat->dim, base->to->name);
    }
  }

  //---------------------------------------------------------------------------
  // Propagate to connected sets and apply physically.
  //---------------------------------------------------------------------------
  propagate_reordering(base->to, base->to, pctx, set_permutations, set_ipermutations);
  for (int i = 0; i < OP_set_index; i++) {
    reorder_set(OP_set_list[i], set_permutations);
  }

  op_move_to_device();

  //---------------------------------------------------------------------------
  // Post-reordering statistics.
  //---------------------------------------------------------------------------
  int max_dist_after;
  long avg_dist_after;
  compute_edge_stats(base, max_dist_after, avg_dist_after);

  op_printf("Before renumbering: maximum bandwidth = %d average bandwidth = %ld\n",
            max_dist_before, avg_dist_before);
  op_printf("After  renumbering: maximum bandwidth = %d average bandwidth = %ld\n",
            max_dist_after, avg_dist_after);
}

void op_renumber(op_map base) {
  const ReorderMethod method = get_reorder_method();
  const PropagateMethod propagate = get_propagate_method();
  op_printf("op_renumber: OP_REORDER = %s, OP_REORDER_PROPAGATE = %s\n", method_name(method),
            propagate_method_name(propagate));
  if (method == ReorderMethod::None)
    return;

  renumber_owned(base, method, propagate);
  /* Every rank, reordered or not: other ranks' import lists name elements by
     their owners' numbering, which has just changed. */
  op_halo_refresh_imports();
}

extern "C" void op_renumber_ptr(int *ptr) {
  op_map item_map = op_search_map_ptr(ptr);

  if (item_map == NULL) {
    printf("ERROR in op_renumber: op_map not found for %p pointer\n", (void*)ptr);
    exit(-1);
  }

  op_renumber(item_map);
}
