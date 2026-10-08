/*
 * hilbert_sfc.hpp
 *
 * Hilbert space-filling-curve (SFC) node ordering.
 *
 * Header-only. Pure geometric algorithm - no OP2 dependencies. Operates on
 * a flat array of `num_verts * dim` coordinates (AoS, row-major: vertex v,
 * dimension d is at index v*dim + d) and produces a permutation with
 * permutation[old_idx] = new_idx such that vertices close together on the
 * Hilbert curve end up at adjacent new indices.
 *
 * Supports 2D and 3D meshes. Coordinates are rescaled into a shared
 * integer grid using an aspect-ratio-preserving normalisation - every axis
 * is divided by the SAME scalar (the largest per-axis range), so physical
 * distances are preserved across axes and a long, thin domain receives
 * more effective bits along its long axis. Each point's Hilbert index is
 * computed via Skilling's transpose-based algorithm (Skilling, 2004,
 * "Programming the Hilbert Curve") followed by bit-interleaving, then
 * vertices are sorted by Hilbert index.
 *
 * Grid resolution is 21 bits per coordinate - yielding a 63-bit Hilbert
 * index for 3D, well inside a uint64_t. The aspect-ratio-preserving
 * normalisation means short-range axes use fewer of those bits; a stats
 * struct reports the per-axis effective bit-count so the caller can
 * detect when the grid resolution has become marginal.
 */

#ifndef HILBERT_SFC_HPP
#define HILBERT_SFC_HPP

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <memory>
#include <utility>

namespace op_renumber_impl {

// Diagnostic statistics optionally filled in by hilbert_sfc_order. Allows
// the caller to detect under-resolved quantisation (many vertices sharing
// one Hilbert index) or strong anisotropy (one axis with far fewer
// effective bits than another).
struct hilbert_sfc_stats {
  int num_verts;             // total number of input vertices
  int distinct_indices;      // distinct Hilbert indices after quantisation
  int distinct_points;       // distinct positions, counted within each run of
                             // equal indices: vertices at one position no curve
                             // can separate, so distinct_points minus
                             // distinct_indices is what quantisation merged
  int dim;                   // spatial dimension (2 or 3)
  int nominal_bits;          // bits per axis used in Skilling (== max across axes)
  double axis_range[3];      // per-axis physical range (max - min)
  double effective_bits[3];  // per-axis effective bits after normalisation;
                             // == nominal_bits for the longest axis, less for
                             // shorter axes. May be negative for extremely
                             // short axes (quantised below grid resolution).
};

// The Hilbert indices of L points given as N coordinates of B bits each (B * N <= 64),
// lane l of X holding point l: Skilling's AxestoTranspose (Skilling, 2004,
// "Programming the Hilbert Curve"), then the transpose's bits interleaved, most
// significant first, X[0]'s bit leading each group of N. Branch-free and lane by
// lane, so the compiler can compute the L points side by side: one point's steps
// form a long dependent chain, L points' do not.
template <int N, int B, int L> inline void hilbert_index(std::uint32_t (&X)[N][L], std::uint64_t (&h)[L]) {
  // Inverse undo: at each level q, a set bit flips X[0]'s lower bits, a clear one
  // swaps them between X[0] and X[i]
  for (int q = B - 1; q > 0; q--) {
    const std::uint32_t P = (1u << q) - 1;
    for (int i = 0; i < N; i++)
      for (int l = 0; l < L; l++) {
        const std::uint32_t set = 0u - ((X[i][l] >> q) & 1u);
        const std::uint32_t t = (X[0][l] ^ X[i][l]) & P & ~set;
        X[0][l] ^= (P & set) | t;
        X[i][l] ^= t;
      }
  }
  // Gray encode. Bit j of t is the parity of X[N-1]'s bits above j, a suffix XOR
  // (Skilling's loop over the levels, done in five shifts).
  for (int i = 1; i < N; i++)
    for (int l = 0; l < L; l++)
      X[i][l] ^= X[i - 1][l];
  for (int l = 0; l < L; l++) {
    std::uint32_t y = X[N - 1][l];
    y ^= y >> 1;
    y ^= y >> 2;
    y ^= y >> 4;
    y ^= y >> 8;
    y ^= y >> 16;
    const std::uint32_t t = y >> 1;
    for (int i = 0; i < N; i++)
      X[i][l] ^= t;
  }

  // Interleave: spread each coordinate's bits N apart
  for (int l = 0; l < L; l++) {
    std::uint64_t key = 0;
    for (int i = 0; i < N; i++) {
      std::uint64_t x = X[i][l];
      if constexpr (N == 2) {
        x = (x | x << 16) & 0x0000ffff0000ffffULL;
        x = (x | x << 8) & 0x00ff00ff00ff00ffULL;
        x = (x | x << 4) & 0x0f0f0f0f0f0f0f0fULL;
        x = (x | x << 2) & 0x3333333333333333ULL;
        x = (x | x << 1) & 0x5555555555555555ULL;
      } else {
        static_assert(N == 3 && B <= 21, "2 or 3 dimensions, at most 64 bits");
        x &= 0x1fffffULL;
        x = (x | x << 32) & 0x1f00000000ffffULL;
        x = (x | x << 16) & 0x1f0000ff0000ffULL;
        x = (x | x << 8) & 0x100f00f00f00f00fULL;
        x = (x | x << 4) & 0x10c30c30c30c30c3ULL;
        x = (x | x << 2) & 0x1249249249249249ULL;
      }
      key |= x << (N - 1 - i);
    }
    h[l] = key;
  }
}

// A point's key and its index among the points.
struct hilbert_keyed {
  std::uint64_t key;
  int index;
};

// Compute a Hilbert-SFC permutation from geometric coordinates.
//  - coords:     flat AoS array of length num_verts * dim (coords[v*dim + d])
//  - num_verts:  number of vertices
//  - dim:        spatial dimension (2 or 3)
//  - permutation (out): permutation[old_idx] = new_idx (resized to num_verts)
//  - stats (out, optional): diagnostic statistics; pass NULL to skip
inline void hilbert_sfc_order(const double *coords,
                              int num_verts,
                              int dim,
                              std::vector<int> &permutation,
                              hilbert_sfc_stats *stats = nullptr) {
  permutation.assign(num_verts, -1);
  if (num_verts == 0) return;
  if (dim < 2 || dim > 3) return; // caller is expected to validate

  // 21 bits per axis. 2D index <= 42 bits, 3D index <= 63 bits. Both fit
  // comfortably in a uint64_t with room to spare.
  constexpr int bits = 21;
  const std::uint32_t scale = (1U << bits) - 1;

  // Bounding box per axis.
  double mn[3] = { DBL_MAX, DBL_MAX, DBL_MAX };
  double mx[3] = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
  for (int v = 0; v < num_verts; v++) {
    for (int d = 0; d < dim; d++) {
      double c = coords[v * dim + d];
      if (c < mn[d]) mn[d] = c;
      if (c > mx[d]) mx[d] = c;
    }
  }

  // Aspect-ratio-preserving normalisation: divide every axis by the SAME
  // scalar (max per-axis range) so physical distances are preserved. Axes
  // with smaller ranges then occupy a smaller portion of the integer grid
  // and effectively consume fewer bits.
  double range[3] = { 0.0, 0.0, 0.0 };
  double max_range = 0.0;
  for (int d = 0; d < dim; d++) {
    range[d] = mx[d] - mn[d];
    if (range[d] > max_range) max_range = range[d];
  }
  if (max_range <= 0.0) max_range = 1.0; // fully degenerate point cloud

  // Quantise coordinates into the integer grid and compute Hilbert index
  // for each vertex. We pair each index with its original vertex id so the
  // final sort yields the ordering directly. Ties in Hilbert index are
  // resolved by vertex id for reproducibility.
  auto indexed = std::make_unique_for_overwrite<hilbert_keyed[]>(num_verts);
  auto keys = [&]<int N>() {
    constexpr int L = 16;
    for (int v0 = 0; v0 < num_verts; v0 += L) {
      std::uint32_t X[N][L] = {};
      std::uint64_t h[L];
      const int lanes = std::min(L, num_verts - v0);
      for (int l = 0; l < lanes; l++)
        for (int d = 0; d < N; d++) {
          double norm = (coords[(size_t)(v0 + l) * N + d] - mn[d]) / max_range;
          if (norm < 0.0) norm = 0.0;
          if (norm > 1.0) norm = 1.0;
          X[d][l] = (std::uint32_t)(norm * (double)scale);
        }
      hilbert_index<N, bits, L>(X, h);
      for (int l = 0; l < lanes; l++)
        indexed[v0 + l] = {h[l], v0 + l};
    }
  };
  if (dim == 2)
    keys.template operator()<2>();
  else
    keys.template operator()<3>();

  std::sort(indexed.get(), indexed.get() + num_verts, [](const hilbert_keyed &a, const hilbert_keyed &b) {
    return a.key != b.key ? a.key < b.key : a.index < b.index;
  });

  for (int i = 0; i < num_verts; i++) {
    permutation[indexed[i].index] = i;
  }

  // Fill in diagnostic stats if requested.
  if (stats != nullptr) {
    stats->num_verts = num_verts;
    stats->dim = dim;
    stats->nominal_bits = bits;

    // Count distinct Hilbert indices post-quantisation by walking the
    // already-sorted array and counting transitions.
    int distinct = 1;
    for (int i = 1; i < num_verts; i++) {
      if (indexed[i].key != indexed[i - 1].key) distinct++;
    }
    stats->distinct_indices = distinct;

    // Distinct positions within each run of equal indices (runs are short).
    int points = 0;
    std::vector<int> run;
    auto before = [&](int a, int b) {
      return std::lexicographical_compare(coords + (size_t)a * dim, coords + (size_t)(a + 1) * dim,
                                          coords + (size_t)b * dim, coords + (size_t)(b + 1) * dim);
    };
    for (int i = 0; i < num_verts;) {
      int j = i + 1;
      while (j < num_verts && indexed[j].key == indexed[i].key) j++;
      run.clear();
      for (int k = i; k < j; k++) run.push_back(indexed[k].index);
      std::sort(run.begin(), run.end(), before);
      for (size_t k = 0; k < run.size(); k++)
        points += k == 0 || before(run[k - 1], run[k]);
      i = j;
    }
    stats->distinct_points = points;

    // Per-axis effective bits. An axis whose range equals max_range
    // consumes `bits` of resolution; a shorter axis consumes
    // bits + log2(range/max_range) - fewer. If range is zero, report 0.
    // Negative values indicate the axis is below grid resolution
    // (quantises entirely into the same bin).
    for (int d = 0; d < 3; d++) {
      stats->axis_range[d] = (d < dim) ? range[d] : 0.0;
      if (d < dim && range[d] > 0.0) {
        stats->effective_bits[d] = (double)bits + std::log2(range[d] / max_range);
      } else {
        stats->effective_bits[d] = 0.0;
      }
    }
  }
}

} // namespace op_renumber_impl

#endif // HILBERT_SFC_HPP
