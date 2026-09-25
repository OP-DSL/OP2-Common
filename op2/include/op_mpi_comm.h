#pragma once

/* Sparse data exchange.
 *
 * Every rank knows what it wants to send and to whom; no rank knows what is
 * coming. Discovering that with a collective costs O(P) memory at best and, as
 * find_neighbors_set did, O(P^2) at worst. NBX (Hoefler et al., "Scalable
 * Communication Protocols for Dynamic Sparse Data Exchange") solves it in
 * Theta(1) space and Theta(log P) time: synchronous sends tell a sender when its
 * message has been matched, a non-blocking barrier entered only after all of a
 * rank's sends are matched detects global completion, and probing receives
 * whatever turns up in the meantime. Only a small header per message goes
 * through that; the payloads follow in a second round, straight into place.
 *
 * Everything here needs only MPI-3.1.
 */

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace op::mpi {

/* Types that can travel through an exchange: trivially copyable, so they can be
 * sent straight out of the caller's memory, and not themselves an address,
 * because an address means nothing on another rank.
 *
 * The address check needs both halves. Rejecting pointers is obvious. Rejecting
 * views is not, and matters more: std::string_view and std::span are trivially
 * copyable and are not pointers, so on a plain trivially-copyable test they go
 * over the wire as their own pointer and length - garbage on arrival. What
 * separates them from a legitimate aggregate payload like std::array<double, 3>,
 * which is also a contiguous range, is that a view refers to storage it does not
 * own; std::ranges::view is precisely that predicate.
 *
 * Known limit: this catches types that *are* an address, not types that
 * *contain* one. `struct { int *p; double x; }` and std::reference_wrapper<int>
 * are trivially copyable, are neither pointers nor views, and are accepted - and
 * would put an address on the wire. Detecting that needs reflection to walk the
 * members, which C++20 does not have. Payloads are the caller's to get right;
 * the concept removes the easy mistakes, not every mistake. */
template <typename T>
concept Exchangeable =
    std::is_trivially_copyable_v<T> && std::same_as<T, std::remove_cvref_t<T>> &&
    !std::is_pointer_v<T> && !std::is_member_pointer_v<T> && !std::is_array_v<T> &&
    !std::ranges::view<T>;

/* Messages live in op::mpi::msg and come in four types. Which one you name says
 * both who owns the payload and what shape it is:
 *
 *   msg::BlockView<T>   a contiguous run of T that the caller keeps alive
 *   msg::Block<T>       a contiguous run of T that the message owns
 *   msg::ItemView<T>    one T that the caller keeps alive
 *   msg::Item<T>        one T, held by value
 *
 * There is deliberately no single type that can be any of them. Ownership is
 * what a reader needs to see, and a variant would bury it inside a constructor
 * call - as well as costing every message the size of its largest state, which
 * lands hardest on the per-element path where messages are most numerous.
 *
 * The `View` suffix is the standard library's marker for not owning, as in
 * string_view and mdspan, so it is the last thing you read.
 *
 * A call takes one message type, not a mixture. Nothing needs to mix them, and
 * keeping them separate is what lets `std::vector<msg::BlockView<int>>` say at
 * the declaration that every payload here must outlive the call.
 *
 * Note what the type system does not do: msg::ItemView and a msg::BlockView of
 * length one both borrow a single element, and building either into a growing
 * vector dangles when it reallocates. Owning types remove that hazard; the
 * borrowing ones only make it visible. */
namespace msg {

template <Exchangeable T>
class BlockView {
public:
  using element_type = T;

  /* Any contiguous, sized, borrowed range of T: an lvalue std::vector<T>,
     std::array<T, N>, or a std::span (a span is always a borrowed range, so this
     covers it - there is deliberately no separate std::span overload).

     The absent span overload is load-bearing. std::span<const T> can be built
     from an rvalue container, because its own range constructor permits it when
     the element type is const: `borrowed_range<R> || is_const_v<element_type>`.
     A `BlockView(int, std::span<const T>)` overload would therefore accept an
     rvalue container - an rvalue std::array, say - by implicit conversion, and
     dangle. */
  template <std::ranges::contiguous_range R>
    requires std::ranges::sized_range<R> && std::ranges::borrowed_range<R> &&
             std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R>>, T>
  BlockView(int dest, R &&values) noexcept
      : dest_{dest}, data_{std::ranges::data(values), std::ranges::size(values)} {}

  /* The same range, but a temporary: rejected with a readable error rather than
     left to fall through the constraints above, which fails with a dump of the
     whole requires clause and the actual reason - borrowed_range - buried in it.
     The explanation is a trailing comment on the declaration line because that is
     the line GCC and Clang print under "declared here". Deleted rather than a
     static_assert in a body, so std::is_constructible still answers false. */
  template <std::ranges::contiguous_range R>
    requires std::ranges::sized_range<R> && (!std::ranges::borrowed_range<R>) &&
             std::same_as<std::remove_cvref_t<std::ranges::range_value_t<R>>, T>
  BlockView(int, R &&) = delete;  // a temporary would dangle: keep it alive, or move it into a msg::Block

  BlockView(int dest, const T *values, std::size_t count) noexcept
      : dest_{dest}, data_{values, count} {}

  int dest() const noexcept { return dest_; }
  std::span<const T> data() const noexcept { return data_; }

private:
  int dest_;
  std::span<const T> data_;
};

template <Exchangeable T>
class Block {
public:
  using element_type = T;

  /* By value: an lvalue is copied, an rvalue moved. Naming this type is the
     statement that the message should own its payload. */
  Block(int dest, std::vector<T> values) noexcept : dest_{dest}, data_{std::move(values)} {}

  int dest() const noexcept { return dest_; }
  std::span<const T> data() const noexcept { return data_; }

private:
  int dest_;
  std::vector<T> data_;
};

template <Exchangeable T>
class ItemView {
public:
  using element_type = T;

  ItemView(int dest, const T &value) noexcept : dest_{dest}, value_{&value} {}

  /* Explanation on the declaration line, as for BlockView, so it is what the
     compiler prints under "declared here". */
  ItemView(int, const T &&) = delete;  // a temporary would dangle: keep it alive, or own it with msg::Item

  int dest() const noexcept { return dest_; }
  std::span<const T> data() const noexcept { return {value_, 1}; }

private:
  int dest_;
  const T *value_;
};

template <Exchangeable T>
class Item {
public:
  using element_type = T;

  Item(int dest, T value) noexcept : dest_{dest}, value_{value} {}

  int dest() const noexcept { return dest_; }

  /* Computed on demand, never cached: the element lives inside this object, so a
     stored span would dangle as soon as the message moved - which a
     std::vector<Item<T>> does whenever it reallocates. */
  std::span<const T> data() const noexcept { return {&value_, 1}; }

private:
  int dest_;
  T value_;
};

}  // namespace msg

/* What the exchange needs of a message, whichever type it is. Kept here rather
   than in msg:: so it does not read as msg::Message, and mirroring
   std::ranges::view, which sits beside the std::ranges::views family. */
template <typename M>
concept Message = requires(const M &m) {
  typename M::element_type;
  { m.dest() } -> std::convertible_to<int>;
  { m.data() } -> std::convertible_to<std::span<const typename M::element_type>>;
};

/* Whether to concatenate the messages bound for one destination before sending.
 *
 * This never changes the result - the receiver groups by source either way - so
 * it is purely a choice between few large messages and many small ones. Nothing
 * is ever discarded; coalescing concatenates, it does not deduplicate. */
enum class Coalesce : bool {
  /* A header and a payload per message object, the payload sent straight out
     of the caller's memory. No send-side copy. */
  no,
  /* A header and a payload per destination. If any destination is named more
     than once, every message's payload is packed into an internal buffer, in
     the order given; only when none repeats are they sent as they stand,
     without copying. */
  yes,
};

/* What arrived, grouped by the rank it came from.
 *
 * ranks is ascending and duplicate-free. counts[i] elements came from ranks[i]
 * and occupy data[disps[i] .. disps[i] + counts[i]); disps is a prefix sum with
 * disps[0] == 0. Messages a rank sent to itself appear as a group from its own
 * rank.
 *
 * Ascending ranks is not cosmetic - callers binary-search this order.
 *
 * counts and disps are int, like the halo lists they feed, so an exchange that
 * delivers more than INT_MAX elements to one rank stops the job rather than
 * wrapping. */
template <typename T>
struct Received {
  std::vector<int> ranks;
  std::vector<int> counts;
  std::vector<int> disps;

  /* Every element, grouped as above; null when nothing arrived. A unique_ptr
     rather than a std::vector because the exchange receives straight into it,
     and a vector cannot be given a size without initialising every element -
     a zero fill that measured up to 40% of a large exchange. Moving it out is
     how a consumer adopts the storage without a copy. */
  std::unique_ptr<T[]> data;

  /* How many neighbours sent something. A count, not the set - the set is
     `ranks`, which is why this is not called neighbours(). */
  int num_neighbours() const { return static_cast<int>(ranks.size()); }

  /* What neighbour i sent, where i is an index in [0, num_neighbours()) - the same i
     that indexes ranks, counts and disps, not a rank. The name says "neighbour"
     rather than "rank" precisely because the two are both int: an accessor named
     for the rank would return the wrong data silently instead of failing. No
     lookup-by-rank accessor exists because nothing needs one; add one if that
     changes. */
  std::span<const T> from_neighbour(int i) const {
    return {data.get() + disps[i], static_cast<std::size_t>(counts[i])};
  }

  /* Flat access, for callers that do not care who sent what. */
  const T *begin() const { return data.get(); }
  const T *end() const { return data.get() + size(); }
  std::size_t size() const {
    return counts.empty() ? 0 : static_cast<std::size_t>(disps.back()) + static_cast<std::size_t>(counts.back());
  }
};

namespace detail {

/* Per-communicator state for the exchanges below: a private communicator and a
 * call counter. Cached on the caller's own communicator as an MPI attribute, so
 * it is freed with its parent and cannot go stale if MPI recycles a handle.
 *
 * NBX probes with MPI_ANY_SOURCE, so it will match *any* message carrying its
 * tag - and a rank leaves the barrier as soon as every rank's sends have been
 * matched, while a straggler is still going round its receive loop. That opens
 * two ways for a probe to steal a message that is not its own, and both need
 * closing:
 *
 * - Against the caller's own traffic. The rank that left the barrier posts
 *   whatever it does next, and the straggler's probe takes it; the intended
 *   receive then never matches and both ranks hang. Callers here pick tags from
 *   set and dat indices, so no reserved tag would be safe - only a communicator
 *   nothing else sends on. Hence the dup.
 *
 * - Against the next exchange. Same window, but the header that arrives early
 *   belongs to exchange k+1. The straggler counts it into exchange k and waits
 *   for its payload, which the sender only posts in exchange k+1's delivery
 *   round - behind a barrier the straggler never reaches - so the job hangs.
 *   Alternating the tag between consecutive calls closes it: for a rank to be sending with tag k's value again it must
 *   have cleared exchange k+1's barrier, which cannot happen until the straggler
 *   has entered exchange k+1 and therefore left exchange k's loop.
 *
 * The counter is consistent across ranks because exchange is collective. */
struct CommState {
  MPI_Comm comm;
  unsigned generation;
};

inline CommState &comm_state(MPI_Comm comm) {
  static int keyval = MPI_KEYVAL_INVALID;

  if (keyval == MPI_KEYVAL_INVALID) {
    MPI_Comm_create_keyval(
        MPI_COMM_NULL_COPY_FN,
        [](MPI_Comm, int, void *value, void *) {
          auto *state = static_cast<CommState *>(value);
          MPI_Comm_free(&state->comm);
          delete state;
          return MPI_SUCCESS;
        },
        &keyval, nullptr);
  }

  void *cached = nullptr;
  int found = 0;
  MPI_Comm_get_attr(comm, keyval, &cached, &found);
  if (found) return *static_cast<CommState *>(cached);

  auto *state = new CommState{MPI_COMM_NULL, 0};
  MPI_Comm_dup(comm, &state->comm);
  MPI_Comm_set_attr(comm, keyval, state);
  return *state;
}

/* MPI counts are int, so a count that does not fit cannot be sent correctly, and
   truncating it would silently corrupt data on another rank. */
[[noreturn]] inline void fail(MPI_Comm comm, const char *what) {
  std::fprintf(stderr, "op::mpi::sparse::exchange: %s\n", what);
  MPI_Abort(comm, 1);
  std::abort(); /* MPI_Abort is not declared noreturn */
}

/* T as a single MPI datatype, so that MPI counts are in elements rather than
   bytes. Counting bytes would cap every message at 2 GB whatever T is; counting
   elements caps it at INT_MAX elements. Built once per T and never freed: it is
   needed until MPI_Finalize, and MPI does not require derived datatypes to be
   freed before then. */
template <Exchangeable T>
MPI_Datatype datatype() {
  static const MPI_Datatype type = [] {
    MPI_Datatype t;
    MPI_Type_contiguous(static_cast<int>(sizeof(T)), MPI_BYTE, &t);
    MPI_Type_commit(&t);
    return t;
  }();
  return type;
}

/* The exchange itself, in two rounds.
 *
 * Discovery: each message is announced by a small header - its element count and
 * element size - sent with Issend on the NBX tag, and the loop below receives
 * headers until every rank's have been matched. Only headers go through NBX. An
 * Issend completes only once its receive has started - on Open MPI a matching
 * probe alone does not count, measured - so a payload sent this way would have to
 * be received before the barrier, into a temporary, since its final position
 * depends on everything that has yet to arrive.
 *
 * Delivery: after the barrier every rank knows exactly who sends it what, so it
 * sizes the result once and receives each payload straight into its final
 * offset. Payloads travel on their own tag with a named source, so no probe can
 * take one. Nor can consecutive exchanges' payloads be confused: a rank finishes
 * its own sends before it can start the next exchange, and MPI does not let
 * messages between one pair overtake each other.
 *
 * Each element is written once on arrival, and nothing is held beyond the result
 * but one header per message. The price is a second round of messages. */
template <Message M>
Received<typename M::element_type> exchange_nbx(MPI_Comm comm, std::span<const M> messages) {
  using T = typename M::element_type;
  CommState &state = comm_state(comm);
  comm = state.comm;
  const int header_tag = static_cast<int>(state.generation++ & 1u);
  constexpr int payload_tag = 2;
  const MPI_Datatype type = datatype<T>();
  constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<int>::max());

  int my_rank = 0;
  MPI_Comm_rank(comm, &my_rank);

  /* The header carries the element size so that ranks disagreeing on T fail
     loudly instead of reinterpreting each other's bytes. */
  struct Header {
    int count;
    int elem_size;
  };

  struct Outgoing {
    int dest;
    std::span<const T> block;
    Header header;
  };

  /* Blocks for this rank are only counted here, and copied straight from the
     caller's messages into place once the result is sized. Empty blocks are
     dropped: one would arrive as a neighbour with no elements, which consumers
     would have to filter out. */
  std::vector<Outgoing> outgoing;
  outgoing.reserve(messages.size());
  std::size_t self_count = 0;
  for (const auto &m : messages) {
    const std::span<const T> block = m.data();
    if (block.empty()) continue;
    if (m.dest() == my_rank) {
      self_count += block.size();
      continue;
    }
    if (block.size() > int_max) fail(comm, "a message holds more than INT_MAX elements");
    outgoing.push_back({m.dest(), block, {static_cast<int>(block.size()), static_cast<int>(sizeof(T))}});
  }

  /* Posted only once `outgoing` is complete: each Issend reads its header in
     place, so the vector must not reallocate underneath it. */
  std::vector<MPI_Request> header_reqs(outgoing.size());
  for (std::size_t i = 0; i < outgoing.size(); ++i)
    MPI_Issend(&outgoing[i].header, 2, MPI_INT, outgoing[i].dest, header_tag, comm, &header_reqs[i]);

  /* `arrival` orders messages from one source; it sits in what would otherwise
     be padding, so the record stays 16 bytes. */
  struct Incoming {
    int source;
    int arrival;
    std::size_t count;
  };
  /* How many headers will arrive is unknown until the barrier; as many as were
     sent is exact for the symmetric patterns halo lists produce, and the self
     group is known. Nothing is reserved when nothing is expected. */
  std::vector<Incoming> incoming;
  incoming.reserve(outgoing.size() + (self_count > 0 ? 1 : 0));
  MPI_Request barrier_req = MPI_REQUEST_NULL;
  bool sends_done = false;

  for (;;) {
    int got = 0;
    MPI_Message message;
    MPI_Status status;
    MPI_Improbe(MPI_ANY_SOURCE, header_tag, comm, &got, &message, &status);

    if (got) {
      Header header;
      MPI_Mrecv(&header, 2, MPI_INT, &message, MPI_STATUS_IGNORE);
      if (header.elem_size != static_cast<int>(sizeof(T)))
        fail(comm, "ranks disagree on the element type: its size differs");
      incoming.push_back({status.MPI_SOURCE, static_cast<int>(incoming.size()),
                          static_cast<std::size_t>(header.count)});
    }

    if (sends_done) {
      int done = 0;
      MPI_Test(&barrier_req, &done, MPI_STATUS_IGNORE);
      if (done) break;
    } else {
      int all_matched = 0;
      MPI_Testall(static_cast<int>(header_reqs.size()), header_reqs.data(), &all_matched,
                  MPI_STATUSES_IGNORE);
      if (all_matched) {
        MPI_Ibarrier(comm, &barrier_req);
        sends_done = true;
      }
    }
  }

  if (self_count > 0) incoming.push_back({my_rank, static_cast<int>(incoming.size()), self_count});

  /* Order by source. Callers binary-search `ranks`, and NBX hands headers back
     in arrival order. Ties break on arrival, so that two messages from the same
     rank keep the order they were sent in, which MPI's non-overtaking rule gives
     the headers and which the payload receives below then follow. That is
     std::stable_sort's guarantee, but stable_sort allocates a merge buffer and
     std::sort does not. */
  std::sort(incoming.begin(), incoming.end(), [](const Incoming &a, const Incoming &b) {
    return a.source != b.source ? a.source < b.source : a.arrival < b.arrival;
  });

  Received<T> out;
  std::size_t groups = 0;
  for (std::size_t i = 0; i < incoming.size(); ++i)
    groups += i == 0 || incoming[i].source != incoming[i - 1].source;
  out.ranks.reserve(groups);
  out.counts.reserve(groups);
  out.disps.reserve(groups);

  std::size_t total = 0;
  for (const auto &in : incoming) {
    /* Checked before anything is added, so no int below can overflow. */
    if (in.count > int_max - total)
      fail(comm, "more than INT_MAX elements arrived, which int disps cannot index");

    /* Several messages from one rank concatenate into a single group, which is
       what lets the send side skip coalescing: `ranks` stays ascending and
       unique. Nothing is discarded - every element arrives, in send order. */
    if (!out.ranks.empty() && out.ranks.back() == in.source) {
      out.counts.back() += static_cast<int>(in.count);
    } else {
      out.ranks.push_back(in.source);
      out.counts.push_back(static_cast<int>(in.count));
      out.disps.push_back(static_cast<int>(total));
    }
    total += in.count;
  }
  /* Left uninitialised: every element is written below, by exactly one receive
     or self copy, since the offsets tile [0, total). Nothing allocated for an
     empty result, where make_unique_for_overwrite would still allocate. */
  if (total > 0) out.data = std::make_unique_for_overwrite<T[]>(total);

  /* Receives first, so that most payloads find a posted receive rather than
     waiting in MPI's unexpected queue. The receives from one rank are posted in
     the order its headers arrived, which is the order its payloads are sent. */
  std::vector<MPI_Request> payload_reqs;
  payload_reqs.reserve(incoming.size() + outgoing.size());
  std::size_t offset = 0;
  for (const auto &in : incoming) {
    T *into = out.data.get() + offset;
    if (in.source == my_rank) {
      /* This rank's own group: its blocks in the order given. */
      for (const auto &m : messages)
        if (m.dest() == my_rank) {
          const std::span<const T> block = m.data();
          into = std::copy(block.begin(), block.end(), into);
        }
    } else {
      payload_reqs.emplace_back();
      MPI_Irecv(into, static_cast<int>(in.count), type, in.source, payload_tag, comm,
                &payload_reqs.back());
    }
    offset += in.count;
  }
  for (const auto &o : outgoing) {
    payload_reqs.emplace_back();
    MPI_Isend(o.block.data(), o.header.count, type, o.dest, payload_tag, comm, &payload_reqs.back());
  }
  MPI_Waitall(static_cast<int>(payload_reqs.size()), payload_reqs.data(), MPI_STATUSES_IGNORE);

  return out;
}

/* True if no destination is named twice, so there is nothing to coalesce.
   Deliberately not reserved to messages.size(): on the one-element-per-message
   path that is millions of buckets, and the scan stops at the first repeat,
   which comes within the first P + 1 messages.

   try_emplace, not emplace, here and below: libstdc++'s emplace builds a node
   before looking the key up and frees it again when the key is already there,
   which on the one-element-per-message path is an allocation per element. */
template <Message M>
bool destinations_unique(std::span<const M> messages) {
  std::unordered_map<int, int> seen;
  for (const auto &m : messages)
    if (!seen.try_emplace(m.dest(), 0).second) return false;
  return true;
}

/* Concatenate the messages for each destination into one buffer, preserving the
 * order they were given in. Whatever message type came in, the result is
 * msg::BlockView messages viewing that buffer, so the buffer is returned
 * alongside them and must outlive them. */
template <Message M>
std::pair<std::unique_ptr<typename M::element_type[]>,
          std::vector<msg::BlockView<typename M::element_type>>>
coalesce_by_destination(std::span<const M> messages) {
  using T = typename M::element_type;
  std::unordered_map<int, int> slot_of;
  std::vector<int> dests;
  std::vector<std::size_t> counts;

  for (const auto &m : messages) {
    auto [at, inserted] = slot_of.try_emplace(m.dest(), static_cast<int>(dests.size()));
    if (inserted) {
      dests.push_back(m.dest());
      counts.push_back(0);
    }
    counts[at->second] += m.data().size();
  }

  std::vector<std::size_t> offsets(dests.size());
  std::size_t total = 0;
  for (std::size_t i = 0; i < dests.size(); ++i) {
    offsets[i] = total;
    total += counts[i];
  }

  /* Uninitialised, like the receive side: the copies below fill every element. */
  std::unique_ptr<T[]> storage;
  if (total > 0) storage = std::make_unique_for_overwrite<T[]>(total);
  std::vector<std::size_t> filled(dests.size(), 0);
  for (const auto &m : messages) {
    const int slot = slot_of.find(m.dest())->second;
    std::copy(m.data().begin(), m.data().end(), storage.get() + offsets[slot] + filled[slot]);
    filled[slot] += m.data().size();
  }

  std::vector<msg::BlockView<T>> packed;
  packed.reserve(dests.size());
  for (std::size_t i = 0; i < dests.size(); ++i)
    packed.emplace_back(dests[i], storage.get() + offsets[i], counts[i]);

  return {std::move(storage), std::move(packed)};
}

}  // namespace detail

/* The sparse exchange itself.
 *
 * Namespaced away from op::mpi because "exchange" already means halo exchange
 * throughout OP2 - op_mpi_halo_exchanges, op_trigger_halo_exchanges, OP_PARTIAL_EXCHANGE
 * and several hundred other uses - and this is not that. This is the dynamic
 * sparse data exchange of Hoefler et al.: setup-time discovery and delivery
 * where no rank knows in advance who will send to it. Written out, the call site
 * reads sparse::exchange, which is the name of the pattern. */
namespace sparse {

/* Send each message to its destination and return everything sent to this rank.
 *
 * Takes one message type from op::mpi::msg - not a mixture; the element type
 * comes from the messages, so it need not be spelled.
 *
 * Collective over comm in the sense that every rank must call it, but it moves
 * no data between ranks that do not exchange messages, and allocates nothing
 * proportional to the communicator.
 *
 * Any number of messages may name the same destination; their payloads are
 * concatenated, in the order given, into that rank's group in the result.
 * Whether they travel as one MPI message or several is what `coalesce` selects,
 * and it does not affect the result.
 *
 * Needs no tag from the caller, and is safe both alongside the caller's own
 * traffic and back to back with itself - see comm_state above. */
template <Message M>
Received<typename M::element_type> exchange(MPI_Comm comm, std::span<const M> messages,
                                            Coalesce coalesce = Coalesce::no) {
  /* Nothing to coalesce when no destination repeats, and saying so here rather
     than inside the packer matters: the messages are sent as they stand, so an
     owning message's payload is never copied just to discover it was already
     the only one for its destination. */
  if (coalesce == Coalesce::no || detail::destinations_unique(messages))
    return detail::exchange_nbx(comm, messages);

  auto [storage, packed] = detail::coalesce_by_destination(messages);
  return detail::exchange_nbx(
      comm, std::span<const msg::BlockView<typename M::element_type>>{packed});
}

/* Not a convenience: template argument deduction does not consider user-defined
   conversions, so a std::vector argument never matches the span parameter above
   even though std::span converts from one. Without this, every caller would have
   to spell out the span. */
template <Message M>
Received<typename M::element_type> exchange(MPI_Comm comm, const std::vector<M> &messages,
                                            Coalesce coalesce = Coalesce::no) {
  return exchange(comm, std::span<const M>{messages}, coalesce);
}

/* CSR layout: message i is values[offsets[i], offsets[i + 1]) sent to dests[i].
 *
 * For callers whose payload is already packed contiguously with an offsets
 * array, so they need not build a message list. The payload is borrowed, so it
 * must outlive the call. offsets.size() must be dests.size() + 1, non-decreasing,
 * with offsets.back() <= values.size(). */
template <Exchangeable T>
Received<T> exchange_csr(MPI_Comm comm, std::span<const int> dests,
                         std::span<const std::size_t> offsets,
                         std::span<const T> values,
                         Coalesce coalesce = Coalesce::no) {
  assert(offsets.size() == dests.size() + 1 && "offsets needs one entry per message, plus an end");
  assert((offsets.empty() || offsets.back() <= values.size()) && "offsets run past values");

  std::vector<msg::BlockView<T>> messages;
  messages.reserve(dests.size());
  for (std::size_t i = 0; i < dests.size(); ++i) {
    assert(offsets[i] <= offsets[i + 1] && "offsets must be non-decreasing");
    messages.emplace_back(dests[i], values.data() + offsets[i], offsets[i + 1] - offsets[i]);
  }

  return exchange(comm, std::span<const msg::BlockView<T>>{messages}, coalesce);
}

}  // namespace sparse

}  // namespace op::mpi
