#ifndef __OP_TIMING2_H
#define __OP_TIMING2_H

#include <extern/json.hpp>

#include <chrono>
#include <optional>
#include <functional>
#include <vector>
#include <string>
#include <string_view>

using json = nlohmann::json;

/*
 * Tree-based timing code for instrumentation of OP2 applications. See the op_timing2 class (and its C API) for the
 * public API methods to use.
 *
 * The OP2 code-generator will generate calls in kernel code for _enter_kernel() and sections inside.
 *
 * Expected usage from application:
 *   - op_timing2_start(name)
 *     - op_timing2_enter(name) (Optional - use to time application sections and separate out kernels)
 *     - op_timing2_exit()
 *     - ...
 *   - op_timing2_finish()
 *   - op_timing2_output()
 *
 * Two environment variables can be used:
 *   - OP_TIMING2_LEVEL={0,1,2,3} - Set the timing detail level
 *     - 0: Disabled
 *     - 1: Only time outer sections from _enter() in the application code (no kernel timing)
 *     - 2: Time outer sections from application code and overall kernel timings (no in-kernel timing)
 *     - 3: Time outer sections and detailed kernel timings (from code-generated _enter() sections)
 *
 *   - OP_TIMING2_JSON_OUTPUT=<filename> - Output the timing tree to the specified file in JSON format during the
 *                                         call to op_timing2_output().
 */

/* ----------------------------------------- op_timing2_clock ----------------------------------------- */

/* Helper struct to hold the timing information for each tree node */
struct op_timing2_clock {
  using clock = std::chrono::high_resolution_clock;

  std::size_t n = 0;

  clock::duration total;
  clock::duration min;
  clock::duration max;

  void submit(clock::duration duration);
  op_timing2_clock& operator+=(const op_timing2_clock& other);

  clock::duration average() const { return total / n; }
};

/* Format a duration auto-selecting units (s, ms, us, ns) into the specified string width */
std::string format_duration(const op_timing2_clock::clock::duration& d, unsigned width = 7);

/* Convert a op_timing2_clock to a string, optionally with a parent clock reference that is used to calculate a 
 * time percentage. The parent needs not be the actual parent of the node in the timing tree. */
std::string to_string(const op_timing2_clock& clock,
                      const std::optional<std::reference_wrapper<const op_timing2_clock>> parent = std::nullopt);


/* JSON conversion implementations */
void to_json(json& j, const op_timing2_clock& clock);
void from_json(const json& j, op_timing2_clock& clock);

/* ----------------------------------------- op_timing2_node ----------------------------------------- */

/* Node type for the timing tree nodes */
enum class op_timing2_node_type { standard, kernel };

/* Timing tree node */
struct op_timing2_node {
  std::string name;

  op_timing2_node_type type = op_timing2_node_type::standard;
  op_timing2_clock clock;

  std::size_t num_ranks = 1;

  std::vector<op_timing2_node> children;

  op_timing2_node(): name{"unknown"} {}
  op_timing2_node(std::string_view name): name{name} {}

  /* Check if the node has an immediate child of the given name (and type) */
  bool has_child(std::string_view name,
      std::optional<op_timing2_node_type> child_type = std::nullopt);

  /* Get an immediate child of the node with the given name (and type). This will create a new node if a match is
   * not found. Child names must be unique, regardless of type */
  op_timing2_node& get_child(std::string_view name,
                             std::optional<op_timing2_node_type> child_type = std::nullopt);

  /* Get a child of the node with the given scope path (and type). The scope must contain at least one name, and
   * children will be created on demand if they do not exist. */
  op_timing2_node& get_child(std::vector<std::string> scope,
                             std::optional<op_timing2_node_type> child_type = std::nullopt);

  /* Combine with another node, probably from another MPI rank, adding the timing statistics together. Children
   * present in the second node will be created and combined into this node */
  op_timing2_node& operator+=(const op_timing2_node& other);

  /* Pretty-print the node to stdout, with an optional parent for time percentage output from the clocks. The parent
   * needs not be the node's actual parent */
  void output(unsigned indent = 0,
              const std::optional<std::reference_wrapper<const op_timing2_node>> parent = std::nullopt);
};

/* JSON conversion implementations */
void to_json(json& j, const op_timing2_node& node);
void from_json(const json& j, op_timing2_node& node);

/* ----------------------------------------- op_timing2 ----------------------------------------- */

/* Timing detail level, set before timing init/start */
enum class op_timing2_level {
  disabled = 0,    // No timing
  simple,          // Only user-defined outer sections, doesn't time kernels
  kernel,          // Includes whole-kernel timing, no in-kernel sections (default)
  kernel_detailed  // All sections, including kernel sections
};

/* The timing class, instantiated as a singleton "timing" - interact through that */
class op_timing2 {
private:
  op_timing2_level level = op_timing2_level::kernel;

  std::vector<std::reference_wrapper<op_timing2_node>> current_scope;
  std::vector<op_timing2_clock::clock::time_point> current_starts;
  unsigned extra_depth = 0; // Depth into sections not enabled by the current level

  op_timing2_node root;

  bool started = false;

  bool finished = false;
  bool combined = false;

public:
  /* Returns the singleton timing instance */
  static op_timing2& instance();

  /* Sets the timing level (only before calling start()) - see op_timing2_level */
  void set_level(const op_timing2_level new_level);

  /* Initialises timing (with application name), required unless level set to disabled */
  void start(std::string_view name);

  /* Enter a timing section, starting the timer for the section. Sync controls if deviceSync() will be called */
  void enter(std::string_view name, bool sync = true);

  /* Enter a kernel section - called by translator-generated kernels, don't call manually */
  void enter_kernel(std::string_view name, std::string_view target, std::string_view variant);

  /* Helper to stop the previous section, and start a new one, equivalent to exit() then enter(name) */
  void next(std::string_view name);

  /* Finish a kernel section, recording elapsed time. Sync controls if deviceSync() will be called */
  void exit(bool sync = true);

  /* End timing, closing all remaining open sections. */
  void finish();

  /* Pretty-print timing statistics, and output JSON to OP_TIMING2_JSON_OUTPUT if it's defined. The timing trees
   * will be combined across MPI ranks the first time this is called if needed. */
  void output();

  /* Output the timing tree to the specified path in JSON format. The timing trees will be combined across MPI ranks
   * first if needed. */
  void output_json(std::string_view filename);

private:
  /* Combines the trees onto the MPI root. Trees are serialised to JSON then MessagePack, transmitted, then
   * deserialised and merged with operator+= */
  void combine();

  /* Pretty-prints the tree, starting with an outer summary, then a kernel summary for each kernel containing node
   * and then finally per-kernel details depending on the timing level */
  void print_summary();

  /* Pretty-prints the non-kernel nodes, accumulating a list of nodes that have immediate kernel children */
  void print_walk_non_kernel(const op_timing2_node& node,
                             const std::vector<std::string>& parent_path,
                             std::vector<std::vector<std::string>>& nodes_with_kernels,
                             unsigned indent = 0);

  /* Pretty-prints a table and details of the given nodes immediate kernel children */
  void print_kernel_summary(const std::vector<std::string>& path, unsigned longest_name);
};

/* C/Fortran timing API functions wrapping the public op_timing2 methods */
extern "C" {

void op_timing2_start(const char* name);

void op_timing2_enter(const char* name);
void op_timing2_enter_kernel(const char* name, const char* target, const char* variant);

void op_timing2_next(const char* name);

void op_timing2_exit();
void op_timing2_finish();

void op_timing2_output();
void op_timing2_output_json(const char* filename);

}

#endif
