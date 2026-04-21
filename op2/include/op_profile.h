#ifndef __OP_PROFILE_H
#define __OP_PROFILE_H

#include <extern/json.hpp>

#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using json = nlohmann::json;

/*
 * Tree-based profiling code for instrumentation of OP2 applications. See the op_profile class for the
 * implementation API, and op_profile_c.h for the C/Fortran entry points.
 *
 * The OP2 code-generator will generate calls for _enter_kernel() and sections inside.
 *
 * Expected usage from application:
 *   - op_profile_start(name)
 *     - op_profile_enter(name) (Optional - use to profile application sections and separate out kernels)
 *     - op_profile_exit()
 *     - ...
 *   - op_profile_end()
 *   - op_profile_output()
 *
 * Two environment variables can be used:
 *   - OP_PROFILE_LEVEL={0,1,2,3} - Set the profiling detail level
 *     - 0: Disabled
 *     - 1: Only profile outer sections from _enter() in the application code (no kernel profiling)
 *     - 2: Profile outer sections from application code and overall kernel profiling (no in-kernel sections)
 *     - 3: Profile outer sections and detailed kernel sections (from code-generated _enter() sections)
 *
 *   - OP_PROFILE_JSON_OUTPUT=<filename> - Output the profiling tree to the specified file in JSON format during the
 *                                         call to op_profile_output().
 */

/* ----------------------------------------- op_profile_clock ----------------------------------------- */

struct op_profile_clock {
  using clock = std::chrono::high_resolution_clock;

  std::size_t n = 0;

  clock::duration total;
  clock::duration min;
  clock::duration max;

  void submit(clock::duration duration);
  op_profile_clock& operator+=(const op_profile_clock& other);

  clock::duration average() const { return total / n; }
};

std::string format_duration(const op_profile_clock::clock::duration& d, unsigned width = 7);

std::string to_string(const op_profile_clock& clock,
                      const std::optional<std::reference_wrapper<const op_profile_clock>> parent = std::nullopt);

void to_json(json& j, const op_profile_clock& clock);
void from_json(const json& j, op_profile_clock& clock);

/* ----------------------------------------- op_profile_node ----------------------------------------- */

enum class op_profile_node_type { standard, kernel };

struct op_profile_node {
  std::string name;

  op_profile_node_type type = op_profile_node_type::standard;
  op_profile_clock clock;

  std::size_t num_ranks = 1;

  std::vector<op_profile_node> children;

  op_profile_node(): name{"unknown"} {}
  op_profile_node(std::string_view name): name{name} {}

  bool has_child(std::string_view name,
                 std::optional<op_profile_node_type> child_type = std::nullopt);

  op_profile_node& get_child(std::string_view name,
                             std::optional<op_profile_node_type> child_type = std::nullopt);

  op_profile_node& get_child(std::vector<std::string> scope,
                             std::optional<op_profile_node_type> child_type = std::nullopt);

  op_profile_node& operator+=(const op_profile_node& other);

  void output(unsigned indent = 0,
              const std::optional<std::reference_wrapper<const op_profile_node>> parent = std::nullopt);
};

void to_json(json& j, const op_profile_node& node);
void from_json(const json& j, op_profile_node& node);

/* ----------------------------------------- op_profile ----------------------------------------- */

enum class op_profile_level {
  disabled = 0,
  simple,
  kernel,
  kernel_detailed
};

class op_profile {
private:
  op_profile_level level = op_profile_level::simple;

  std::vector<std::reference_wrapper<op_profile_node>> current_scope;
  std::vector<op_profile_clock::clock::time_point> current_starts;
  unsigned extra_depth = 0;

  op_profile_node root;

  bool started = false;
  bool ended = false;
  bool combined = false;
  bool runtime_disabled = false;
  bool local_error = false;

public:
  static op_profile& instance();

  void set_level(const op_profile_level new_level);

  void start(std::string_view name);
  void enter(std::string_view name, bool sync = true);
  void enter_kernel(std::string_view name, std::string_view target, std::string_view variant);
  void next(std::string_view name);
  void exit(bool sync = true);
  void end();
  void output();
  void output_json(std::string_view filename);

private:
  void combine();
  void print_summary();
  void print_walk_non_kernel(const op_profile_node& node,
                             const std::vector<std::string>& parent_path,
                             std::vector<std::vector<std::string>>& nodes_with_kernels,
                             unsigned indent = 0);
  void print_kernel_summary(const std::vector<std::string>& path, unsigned longest_name);
};

#endif
