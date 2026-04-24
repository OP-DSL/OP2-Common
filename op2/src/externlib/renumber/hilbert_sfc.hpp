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
 * Grid resolution is 20 bits per coordinate - yielding a 60-bit Hilbert
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
#include <utility>

namespace op_renumber_impl {

// Diagnostic statistics optionally filled in by hilbert_sfc_order. Allows
// the caller to detect under-resolved quantisation (many vertices sharing
// one Hilbert index) or strong anisotropy (one axis with far fewer
// effective bits than another).
struct hilbert_sfc_stats {
  int num_verts;             // total number of input vertices
  int distinct_indices;      // distinct Hilbert indices after quantisation
  int dim;                   // spatial dimension (2 or 3)
  int nominal_bits;          // bits per axis used in Skilling (== max across axes)
  double axis_range[3];      // per-axis physical range (max - min)
  double effective_bits[3];  // per-axis effective bits after normalisation;
                             // == nominal_bits for the longest axis, less for
                             // shorter axes. May be negative for extremely
                             // short axes (quantised below grid resolution).
};

// Skilling's AxestoTranspose. Converts n coordinates, each up to b bits, into
// the Hilbert transpose form in place. Identical to the published reference
// implementation (Skilling, 2004).
inline void hilbert_axes_to_transpose(std::uint32_t *X, int b, int n) {
  std::uint32_t M = 1U << (b - 1);
  std::uint32_t P, Q, t;

  // Inverse undo
  for (Q = M; Q > 1; Q >>= 1) {
    P = Q - 1;
    for (int i = 0; i < n; i++) {
      if (X[i] & Q) {
        X[0] ^= P;
      } else {
        t = (X[0] ^ X[i]) & P;
        X[0] ^= t;
        X[i] ^= t;
      }
    }
  }

  // Gray encode
  for (int i = 1; i < n; i++)
    X[i] ^= X[i - 1];
  t = 0;
  for (Q = M; Q > 1; Q >>= 1) {
    if (X[n - 1] & Q) t ^= Q - 1;
  }
  for (int i = 0; i < n; i++)
    X[i] ^= t;
}

// Interleave transposed bits (most-significant bit first) into a single
// scalar Hilbert index. Output is b*n bits wide.
inline std::uint64_t hilbert_transpose_to_index(const std::uint32_t *X, int b, int n) {
  std::uint64_t h = 0;
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < n; j++) {
      std::uint64_t bit = (X[j] >> (b - 1 - i)) & 1u;
      h = (h << 1) | bit;
    }
  }
  return h;
}

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

  // 20 bits per axis. 2D index <= 40 bits, 3D index <= 60 bits. Both fit
  // comfortably in a uint64_t with room to spare.
  const int bits = 20;
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
  std::vector<std::pair<std::uint64_t, int> > indexed(num_verts);
  for (int v = 0; v < num_verts; v++) {
    std::uint32_t X[3] = { 0, 0, 0 };
    for (int d = 0; d < dim; d++) {
      double norm = (coords[v * dim + d] - mn[d]) / max_range;
      if (norm < 0.0) norm = 0.0;
      if (norm > 1.0) norm = 1.0;
      X[d] = (std::uint32_t)(norm * (double)scale);
    }
    hilbert_axes_to_transpose(X, bits, dim);
    std::uint64_t h = hilbert_transpose_to_index(X, bits, dim);
    indexed[v] = std::make_pair(h, v);
  }

  std::sort(indexed.begin(), indexed.end());

  for (int i = 0; i < num_verts; i++) {
    permutation[indexed[i].second] = i;
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
      if (indexed[i].first != indexed[i - 1].first) distinct++;
    }
    stats->distinct_indices = distinct;

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
