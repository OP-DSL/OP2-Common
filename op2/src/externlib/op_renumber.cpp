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
 * op_renumber(base) reorders each rank's core elements of base->to, the primary
 * set, with the ordering the OP_REORDER environment variable names (the
 * single-node libraries have a stub that warns and reorders nothing):
 *
 *     OP_REORDER=none      - no reordering
 *     OP_REORDER=random    - random permutation (benchmark baseline)
 *     OP_REORDER=rcm       - Reverse Cuthill-McKee (default without geometry)
 *     OP_REORDER=sloan     - Sloan profile-minimising ordering
 *     OP_REORDER=hilbert   - Hilbert space-filling curve, on the primary set's
 *                            geometry (op_set_coords{,_derived}; default with it)
 *
 * Once the primary set's permutation is computed, OP_REORDER_PROPAGATE
 * controls how that permutation is extended to other sets reachable via
 * maps (e.g., reordering edges after nodes have been reordered):
 *
 *     OP_REORDER_PROPAGATE=lex       - multi-key lex sort over all map
 *                                      dimensions (default without
 *                                      geometry)
 *     OP_REORDER_PROPAGATE=centroid  - Hilbert SFC of stencil centroid;
 *                                      centroids are averaged from the
 *                                      already-ordered parent set, so
 *                                      this needs the primary set's
 *                                      geometry (default with it)
 *     OP_REORDER_PROPAGATE=single    - legacy single-dim sort by first
 *                                      map endpoint only (kept for
 *                                      benchmarking and regression)
 *
 * The ordering algorithms themselves live in header-only modules:
 *
 *     renumber/rcm.hpp          - RCM + shared graph utilities
 *     renumber/sloan.hpp        - Sloan (depends on rcm.hpp)
 *     renumber/hilbert_sfc.hpp  - Hilbert SFC (standalone, geometric)
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
#include <cstdio>
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
//             centroids. Symmetric across all map dimensions; needs the
//             primary set's geometry.
enum class PropagateMethod {
  Single,
  Lex,
  Centroid,
};

/* How many of a set's core elements a Hilbert curve ordered, at how many
   distinct positions, and with how many distinct keys: elements that share a key
   keep their previous relative order, so positions merged into one key mean the
   curve resolves the mesh too coarsely. */
struct HilbertKeys {
  int elements = 0, points = 0, keys = 0;
};

// Per-call state carried through propagate_reordering recursion.
struct PropagationContext {
  PropagateMethod method;
  int coord_dim;  // the primary set's geometry's dimension, 2 or 3 in centroid mode
  // set_centroids[s] is a flat AoS array of length size * coord_dim for
  // set s (every owned element, so that a core element's parents are all
  // there), in original (pre-permutation) element order. Empty for
  // sets that have no centroid available (either coord_dim is 0, or the
  // set was reordered through a path that didn't propagate centroids).
  std::vector<std::vector<double> > set_centroids;
  std::vector<HilbertKeys> keys; // by set, for every set a Hilbert curve ordered
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
// Halo entries in the map (parent index >= parent size) are skipped: only
// owned elements that are not core can reach them, and they average over
// fewer parents. Core elements reach owned parents only.
static bool compute_centroids_to_from(op_map map, PropagationContext &pctx) {
  op_set child = map->from;
  op_set parent = map->to;
  const std::vector<double> &pc = pctx.set_centroids[parent->index];
  if (pc.empty()) return false;

  const int cd = pctx.coord_dim;
  const int n = child->size;
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
  op_set parent = map->from;
  op_set child = map->to;
  const std::vector<double> &pc = pctx.set_centroids[parent->index];
  if (pc.empty()) return false;

  const int cd = pctx.coord_dim;
  const int n = child->size;
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

        // Centroids, unless the parent has none (it has no owned elements);
        // then lex.
        bool ordered_via_centroid = false;
        if (pctx.method == PropagateMethod::Centroid && compute_centroids_to_from(map, pctx)) {
          std::vector<int> perm;
          op_renumber_impl::hilbert_sfc_stats stats{};
          op_renumber_impl::hilbert_sfc_order(
              pctx.set_centroids[to->index].data(), n, pctx.coord_dim, perm, &stats);
          pctx.keys[to->index] = {stats.num_verts, stats.distinct_points, stats.distinct_indices};
          for (int i = 0; i < n; i++)
            set_permutations[to->index][i] = perm[i];
          ordered_via_centroid = true;
        }

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
            // Lex, also when the parent has no centroids.
            std::vector<int> order = compute_lex_order(
                map, set_permutations[from->index], n);
            for (int i = 0; i < n; i++)
              set_permutations[to->index][order[i]] = i;
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
        if (pctx.method == PropagateMethod::Centroid)
          compute_centroids_from_to(map, pctx);
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

/* OP_REORDER; unset, a Hilbert curve if the primary set has geometry, else RCM.
   A Hilbert curve asked for without geometry is RCM, with a warning. Geometry is
   registered on every rank alike, so every rank chooses the same. */
static ReorderMethod get_reorder_method(op_set set) {
  const bool geometry = set->coords != NULL;
  const ReorderMethod fallback = geometry ? ReorderMethod::Hilbert : ReorderMethod::RCM;
  const char *env = getenv("OP_REORDER");
  ReorderMethod m = fallback;
  if (env != NULL && env[0] != '\0') {
    if (str_iequals(env, "none"))         m = ReorderMethod::None;
    else if (str_iequals(env, "random"))  m = ReorderMethod::Random;
    else if (str_iequals(env, "rcm"))     m = ReorderMethod::RCM;
    else if (str_iequals(env, "sloan"))   m = ReorderMethod::Sloan;
    else if (str_iequals(env, "hilbert")) m = ReorderMethod::Hilbert;
    else
      op_printf("Warning: unknown OP_REORDER value '%s', using %s\n", env, method_name(fallback));
  }
  if (m == ReorderMethod::Hilbert && !geometry) {
    op_printf("Warning: OP_REORDER=hilbert needs the geometry of set %s, registered with op_set_coords or "
              "op_set_coords_derived; using RCM\n", set->name);
    m = ReorderMethod::RCM;
  }
  return m;
}

static const char *propagate_method_name(PropagateMethod m) {
  switch (m) {
    case PropagateMethod::Single:   return "single";
    case PropagateMethod::Lex:      return "lex";
    case PropagateMethod::Centroid: return "centroid";
  }
  return "unknown";
}

/* OP_REORDER_PROPAGATE; unset, centroids if the primary set has geometry, else lex.
   Centroids asked for without geometry are lex, with a warning. */
static PropagateMethod get_propagate_method(op_set set) {
  const bool geometry = set->coords != NULL;
  const PropagateMethod fallback = geometry ? PropagateMethod::Centroid : PropagateMethod::Lex;
  const char *env = getenv("OP_REORDER_PROPAGATE");
  PropagateMethod m = fallback;
  if (env != NULL && env[0] != '\0') {
    if (str_iequals(env, "single") || str_iequals(env, "legacy"))
      m = PropagateMethod::Single;
    else if (str_iequals(env, "lex") || str_iequals(env, "multikey"))
      m = PropagateMethod::Lex;
    else if (str_iequals(env, "centroid") || str_iequals(env, "hilbert"))
      m = PropagateMethod::Centroid;
    else
      op_printf("Warning: unknown OP_REORDER_PROPAGATE value '%s', using %s\n", env,
                propagate_method_name(fallback));
  }
  if (m == PropagateMethod::Centroid && !geometry) {
    op_printf("Warning: OP_REORDER_PROPAGATE=centroid needs the geometry of set %s, registered with op_set_coords "
              "or op_set_coords_derived; using lex\n", set->name);
    m = PropagateMethod::Lex;
  }
  return m;
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

/* The geometry of a set's owned elements, size * dim doubles in local order (the
   core elements first), from op_set_coords{,_derived}; empty if none was
   registered. Derived geometry averages over the elements the map reaches that
   this rank owns: halo creation has run, but nothing has exchanged the
   coordinates' halo yet. */
static std::vector<double> owned_coords(op_set set, int *dim) {
  const op_dat coords = set->coords;
  if (coords == NULL)
    return {};
  const int d = coords->dim, n = set->size;
  const double *xyz = reinterpret_cast<const double *>(coords->data);
  *dim = d;
  const op_map map = set->coords_map;
  if (map == NULL)
    return std::vector<double>(xyz, xyz + (size_t)n * d);

  std::vector<double> out((size_t)n * d, 0.0);
  int starved = 0;
  for (int e = 0; e < n; e++) {
    int owned = 0;
    for (int j = 0; j < map->dim; j++) {
      const int p = map->map[(size_t)e * map->dim + j];
      if (p >= coords->set->size)
        continue;
      for (int c = 0; c < d; c++)
        out[(size_t)e * d + c] += xyz[(size_t)p * d + c];
      owned++;
    }
    starved += owned == 0;
    for (int c = 0; c < d && owned > 0; c++)
      out[(size_t)e * d + c] /= owned;
  }
  if (starved > 0)
    op_printf("WARNING in op_renumber: %d of %d owned elements of set %s reach no owned element of %s through %s, "
              "so they sit at the origin\n", starved, n, set->name, coords->set->name, map->name);
  return out;
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

/* The graph RCM and Sloan order: the core elements of base->to, two of them
   adjacent when one element of base->from maps to both - or, for a map from a
   set to itself, when one maps to the other. Symmetric, with no self-loops or
   repeats; entries outside the core (halo elements, or negative) are left out.
   The neighbours of u are col_indices[row_offsets[u]] up to row_offsets[u + 1]. */
static void build_core_adjacency(op_map base, std::vector<int> &row_offsets, std::vector<int> &col_indices) {
  const int n = base->to->core_size, dim = base->dim;
  const bool self = base->from == base->to;
  auto in_core = [n](int v) { return v >= 0 && v < n; };
  // calls f(u, v) for every edge, once each way round
  auto each_edge = [&](auto &&f) {
    for (int e = 0; e < (self ? n : base->from->size); e++) {
      const int *row = base->map + (size_t)e * dim;
      for (int a = 0; a < dim; a++) {
        if (!in_core(row[a]))
          continue;
        if (self) {
          if (row[a] != e) {
            f(e, row[a]);
            f(row[a], e);
          }
        } else {
          for (int b = 0; b < dim; b++)
            if (b != a && in_core(row[b]) && row[b] != row[a])
              f(row[a], row[b]);
        }
      }
    }
  };

  row_offsets.assign(n + 1, 0);
  each_edge([&](int u, int) { row_offsets[u + 1]++; });
  for (int u = 0; u < n; u++)
    row_offsets[u + 1] += row_offsets[u];
  col_indices.resize(row_offsets[n]);
  std::vector<int> next(row_offsets.begin(), row_offsets.end() - 1);
  each_edge([&](int u, int v) { col_indices[next[u]++] = v; });

  // sort each row and drop its repeats, compacting in place
  int out = 0;
  for (int u = 0; u < n; u++) {
    const int begin = row_offsets[u], end = row_offsets[u + 1], start = out;
    std::sort(col_indices.begin() + begin, col_indices.begin() + end);
    for (int k = begin; k < end; k++)
      if (out == start || col_indices[out - 1] != col_indices[k])
        col_indices[out++] = col_indices[k];
    row_offsets[u] = start;
  }
  row_offsets[n] = out;
  col_indices.resize(out);
}

//-----------------------------------------------------------------------------
// Public entry point
//-----------------------------------------------------------------------------

/* Reorder this rank's core elements of base's target set, and of every set the
   ordering propagates to; returns, by set, what the Hilbert curves among them
   made of their keys. It can give up on one rank alone (a core with no edges of
   its own), so nothing collective may happen in here. */
static std::vector<HilbertKeys> renumber_owned(op_map base, ReorderMethod method, PropagateMethod propagate) {
  std::vector<HilbertKeys> keys(OP_set_index);
  int num_verts = base->to->core_size;
  if (num_verts == 0) {
    if (OP_diags > 2)
      printf("op_renumber: core_size is zero on set %s, nothing to do\n", base->to->name);
    return keys;
  }

  // The primary set's geometry: Hilbert orders by it, and centroid propagation
  // starts from it. op_renumber chose neither without it.
  int coord_dim = 0;
  const bool need_coords = method == ReorderMethod::Hilbert || propagate == PropagateMethod::Centroid;
  const std::vector<double> coords = need_coords ? owned_coords(base->to, &coord_dim) : std::vector<double>();

  //---------------------------------------------------------------------------
  // Compute the core-size permutation via the selected algorithm.
  //---------------------------------------------------------------------------
  std::vector<int> permutation;

  if (method == ReorderMethod::Random) {
    random_order(num_verts, permutation);

  } else if (method == ReorderMethod::Hilbert) {
    op_renumber_impl::hilbert_sfc_stats hstats{};
    op_renumber_impl::hilbert_sfc_order(coords.data(), num_verts, coord_dim, permutation, &hstats);
    keys[base->to->index] = {hstats.num_verts, hstats.distinct_points, hstats.distinct_indices};

    // Grid resolution and per-axis effective bits, on rank 0; report_hilbert_keys
    // warns about repeated keys on any rank.
    if (OP_diags > 2) {
      op_printf("op_renumber: Hilbert SFC on set %s at %d bits/axis: %d elements, %d positions, %d keys\n",
                base->to->name, hstats.nominal_bits, hstats.num_verts, hstats.distinct_points, hstats.distinct_indices);
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
    }

  } else {
    // RCM and Sloan both need the CSR adjacency.
    std::vector<int> row_offsets, col_indices;
    build_core_adjacency(base, row_offsets, col_indices);
    if (col_indices.empty()) {
      if (OP_diags > 2)
        printf("op_renumber: no two core elements of set %s share an element of %s, nothing to order\n",
               base->to->name, base->from->name);
      return keys;
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
  int max_dist_before = 0;
  long avg_dist_before = 0;
  if (OP_diags > 2)
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
  // centroids from its geometry (in original/pre-permutation order - the
  // helpers in propagate_reordering index map entries against this layout).
  //---------------------------------------------------------------------------
  PropagationContext pctx;
  pctx.method = propagate;
  pctx.coord_dim = coord_dim;
  pctx.set_centroids.resize(OP_set_index);
  pctx.keys = std::move(keys);
  if (propagate == PropagateMethod::Centroid)
    pctx.set_centroids[base->to->index] = coords;

  //---------------------------------------------------------------------------
  // Propagate to connected sets and apply physically.
  //---------------------------------------------------------------------------
  propagate_reordering(base->to, base->to, pctx, set_permutations, set_ipermutations);
  // a set the propagation did not reach keeps its order
  for (int s = 0; s < OP_set_index; s++)
    if (!set_permutations[s].empty())
      op::mpi::move_owned(OP_set_list[s], {set_permutations[s].data(), (std::size_t)OP_set_list[s]->size});

  op_move_to_device();

  //---------------------------------------------------------------------------
  // Post-reordering statistics.
  //---------------------------------------------------------------------------
  if (OP_diags > 2) {
    int max_dist_after;
    long avg_dist_after;
    compute_edge_stats(base, max_dist_after, avg_dist_after);
    op_printf("Before renumbering: maximum bandwidth = %d average bandwidth = %ld\n",
              max_dist_before, avg_dist_before);
    op_printf("After  renumbering: maximum bandwidth = %d average bandwidth = %ld\n",
              max_dist_after, avg_dist_after);
  }
  return pctx.keys;
}

/* Warn if, in some set, the Hilbert curve merged into one key the positions of
   more than 1% of the core elements it ordered - on any rank, naming the worst:
   those elements keep their previous relative order, so the curve resolves the
   mesh too coarsely there. Elements at one position (two boundary faces of a
   corner cell, given its centroid) cannot be separated and do not count.
   Collective over OP_MPI_WORLD. */
static void report_hilbert_keys(const std::vector<HilbertKeys> &keys) {
  struct Shared {
    double fraction;
    int rank; // MPI_DOUBLE_INT
  };
  std::vector<Shared> mine(OP_set_index), worst(OP_set_index);
  int rank;
  MPI_Comm_rank(OP_MPI_WORLD, &rank);
  for (int s = 0; s < OP_set_index; s++)
    mine[s] = {keys[s].elements > 0 ? (double)(keys[s].points - keys[s].keys) / keys[s].elements : 0.0, rank};
  MPI_Allreduce(mine.data(), worst.data(), OP_set_index, MPI_DOUBLE_INT, MPI_MAXLOC, OP_MPI_WORLD);
  for (int s = 0; s < OP_set_index; s++)
    if (worst[s].fraction > 0.01)
      op_printf("WARNING: op_renumber: on rank %d the Hilbert curve gives %.1f%% of the core elements of set %s the "
                "key of an element elsewhere, so they keep their previous order: the mesh is finer there than the "
                "curve resolves\n", worst[s].rank, 100 * worst[s].fraction, OP_set_list[s]->name);
}

void op_renumber(op_map base) {
  const ReorderMethod method = get_reorder_method(base->to);
  const PropagateMethod propagate = get_propagate_method(base->to);
  if (method == ReorderMethod::None) {
    op_printf("op_renumber: OP_REORDER=none, nothing is reordered\n");
    return;
  }
  op_printf("op_renumber: %s ordering of set %s through map %s, propagated by %s\n", method_name(method),
            base->to->name, base->name, propagate_method_name(propagate));

  const std::vector<HilbertKeys> keys = renumber_owned(base, method, propagate);
  /* Every rank, reordered or not: other ranks' import lists name elements by
     their owners' numbering, which has just changed. */
  op_halo_refresh_imports();
  report_hilbert_keys(keys);
}

extern "C" void op_renumber_ptr(int *ptr) {
  op_map item_map = op_search_map_ptr(ptr);

  if (item_map == NULL) {
    printf("ERROR in op_renumber: op_map not found for %p pointer\n", (void*)ptr);
    exit(-1);
  }

  op_renumber(item_map);
}
