#include <op_timing2.h>
#include <op_lib_core.h>

#ifdef OPMPI
#include <op_mpi_core.h>
#endif

#include <cassert>
#include <cstring>

#include <sstream>
#include <fstream>

/* (De-)serialisation support for std::chrono::durations, assumes that the rep/period 
 * you are deserialising to matches that which was used for serialisation. */
namespace nlohmann {
  template<typename Rep, typename Period>
  struct adl_serializer<std::chrono::duration<Rep, Period>> {
    static void to_json(json& j, const std::chrono::duration<Rep, Period>& duration) {
      j = duration.count();
    }

    static void from_json(const json& j, std::chrono::duration<Rep, Period>& duration) {
      duration = std::chrono::duration<Rep, Period>(j.template get<Rep>());
    }
  };
}

/* ----------------------------------------- op_timing2_clock ----------------------------------------- */

void op_timing2_clock::submit(op_timing2_clock::clock::duration duration) {
  ++n;

  if (n == 1) {
    total = duration;
    min = duration;
    max = duration;

    return;
  }

  total += duration;

  if (duration < min) min = duration;
  if (duration > max) max = duration;
}

op_timing2_clock& op_timing2_clock::operator+=(const op_timing2_clock& other) {
  n += other.n;

  total += other.total;

  if (other.min < min) min = other.min;
  if (other.max < max) max = other.max;

  return *this;
}

std::string format_duration(const op_timing2_clock::clock::duration& d, unsigned width) { 
    const char *unit[4] = {"s", "ms", "us", "ns"};
    const int unit_width[4] {1, 2, 2, 2};

    double v = std::chrono::duration<double>(d).count();

    // Find the appropropriate output units
    int i = 0;
    for (; i < 4; ++i) {
        if (v >= 1) break;
        v *= 1000.0;
    }

    const int remaining_width = width - unit_width[i];

    // Calculate how many digits left of the decimal point there are
    int left_width = 1;
    double v2 = v;
    while (v2 >= 10) {
        left_width++;
        v2 /= 10;
    }

    // The rest of width can be used for decimal places if remaining width > 1
    int right_width = std::max(remaining_width - left_width - 1, 0);

    int padding = 0;
    if (right_width == 0)
        padding = std::max(remaining_width - left_width, 0);

    constexpr int OUTPUT_LEN = 1024;
    char output[OUTPUT_LEN];
    assert(width + 1 <= OUTPUT_LEN);

    std::snprintf(output, width + 1, "%*s%*.*f%s",
        padding, "", left_width, right_width, v, unit[i]);

    return std::string(output);
}

std::string to_string(const op_timing2_clock& clock,
    const std::optional<std::reference_wrapper<const op_timing2_clock>> parent) {
  std::ostringstream oss;
  oss << "*" << clock.n << " total: " << format_duration(clock.total);

  // Calculate and output the total time as a percentage of the optionally provided parent total
  if (parent.has_value()) {
    auto to_s = [](auto t) { return std::chrono::duration<double>(t).count(); };
    auto pct = 100.0 * to_s(clock.total) / to_s(parent->get().total);

    oss.precision(1);
    oss << std::fixed;
    oss << " (" << pct << "%)";
  }

  if (clock.n > 1) { 
    oss << " (average: " << format_duration(clock.average())
            << ", min: " << format_duration(clock.min)
            << ", max: " << format_duration(clock.max) << ")";
  }

  return oss.str();
}

void to_json(json& j, const op_timing2_clock& clock) {
  j = json{
    {"n", clock.n},
    {"total", clock.total},
    {"min", clock.min},
    {"max", clock.max},
  };
}

void from_json(const json& j, op_timing2_clock& clock) {
  j.at("n").get_to(clock.n);
  j.at("total").get_to(clock.total);
  j.at("min").get_to(clock.min);
  j.at("max").get_to(clock.max);
}

/* ----------------------------------------- op_timing2_node ----------------------------------------- */

bool op_timing2_node::has_child(std::string_view name, std::optional<op_timing2_node_type> child_type) {
  for (auto& child: children) {
    if (child.name == name) {
      if (child_type.has_value()) assert(child.type == *child_type);
      return true;
    }
  }

  return false;
}

op_timing2_node& op_timing2_node::get_child(std::string_view name, std::optional<op_timing2_node_type> child_type) {
  for (auto& child: children) {
    if (child.name == name) {
      if (child_type.has_value()) assert(child.type == *child_type);
      return child;
    }
  }

  // Create the child if we don't have one already, setting the type if it was provided
  children.push_back(op_timing2_node(name));
  if (child_type.has_value()) children.back().type = *child_type;

  return children.back();
}

op_timing2_node& op_timing2_node::get_child(std::vector<std::string> scope,
    std::optional<op_timing2_node_type> child_type) {
  assert(scope.size() >= 1);

  // Recursively call get_child popping the first element off the scope
  if (scope.size() == 1) return get_child(scope[0], child_type);
  return get_child(scope[0]).get_child(std::vector<std::string>(scope.begin() + 1, scope.end()), child_type);
}

op_timing2_node& op_timing2_node::operator+=(const op_timing2_node& other) {
  clock += other.clock;
  num_ranks += other.num_ranks;

  // Missing children in this tree will be created by get_child
  for (auto& child: other.children)
    get_child(child.name, child.type) += child;

  return *this;
}

void op_timing2_node::output(unsigned indent,
    const std::optional<std::reference_wrapper<const op_timing2_node>> parent) {
  std::printf("%*s%s %s\n", indent, "", name.c_str(),
      to_string(clock, parent.has_value() ? std::optional(parent->get().clock) : std::nullopt).c_str());

  for (auto& child: children)
    child.output(indent + 4, parent.has_value() ? parent : *this);
}

void to_json(json& j, const op_timing2_node& node) {
  j = json{
    {"name", node.name},
    {"type", node.type},
    {"clock", node.clock},
    {"num_ranks", node.num_ranks},
    {"children", node.children},
  };
}

void from_json(const json& j, op_timing2_node& node) {
  j.at("name").get_to(node.name);
  j.at("type").get_to(node.type);
  j.at("clock").get_to(node.clock);
  j.at("num_ranks").get_to(node.num_ranks);
  j.at("children").get_to(node.children);
}

/* ----------------------------------------- op_timing2 ----------------------------------------- */

op_timing2& op_timing2::instance() {
  static auto timing = op_timing2{};
  return timing;
}

void op_timing2::set_level(op_timing2_level new_level) {
  assert(!started);
  level = new_level;
}

void op_timing2::start(std::string_view name) {
  assert(!started);
  assert(current_scope.size() == 0);

  char *level_str = getenv("OP_TIMING2_LEVEL");
  if (level_str != nullptr) {
    int level_int = -1;

    try {
      level_int = std::stoi(level_str);
    } catch (...) {};

    if (level_int < 0 || level_int > static_cast<int>(op_timing2_level::kernel_detailed))
      std::printf("warning: OP_TIMING2_LEVEL set to unsupported value: %s\n", level_str);
    else
      level = static_cast<op_timing2_level>(level_int);
  }

  if (level == op_timing2_level::disabled) return;

  started = true;
  deviceSync();

  root = op_timing2_node(name);

  current_scope.push_back(root);
  current_starts.push_back(op_timing2_clock::clock::now());
}

void op_timing2::enter(std::string_view name, bool sync) {
  if (level == op_timing2_level::disabled) return;

  assert(started && !finished);
  assert(current_scope.size() > 0);

  // Check if we should actually start a timer
  if (extra_depth > 0 ||
      (level < op_timing2_level::kernel_detailed &&
       current_scope.back().get().type == op_timing2_node_type::kernel)) {
    extra_depth++;
    return;
  }

  auto& parent = current_scope.back().get();
  auto& node = parent.get_child(name);

  if (sync) deviceSync();

  current_scope.push_back(node);
  current_starts.push_back(op_timing2_clock::clock::now());
}

void op_timing2::enter_kernel(std::string_view name, std::string_view target, std::string_view variant) {
  if (level == op_timing2_level::disabled) return;

  assert(started && !finished);
  assert(current_scope.size() > 0);

  if (level < op_timing2_level::kernel) {
    extra_depth++;
    return;
  }

  auto full_name = std::string(name);

  full_name += "/"; full_name += target;
  full_name += "/"; full_name += variant;
  deviceSync();

  enter(full_name);
  current_scope.back().get().type = op_timing2_node_type::kernel;
}

void op_timing2::next(std::string_view name) {
  exit();
  enter(name, false);
}

void op_timing2::exit(bool sync) {
  if (level == op_timing2_level::disabled) return;

  assert(started && !finished);
  assert(current_scope.size() > 0);

  if (extra_depth > 0) {
    extra_depth--;
    return;
  }

  if (sync) deviceSync();

  auto& node = current_scope.back().get();
  node.clock.submit(op_timing2_clock::clock::now() - current_starts.back());

  current_scope.pop_back();
  current_starts.pop_back();
}

void op_timing2::finish() {
  if (level == op_timing2_level::disabled) return;

  assert(started && !finished);
  assert(current_scope.size() > 0);

  deviceSync();
  while (current_scope.size() > 0)
    exit(false);

  finished = true;
}

void op_timing2::combine() {
  if (level == op_timing2_level::disabled) return;

  assert(finished);
  if (combined) return;

#ifdef OPMPI
  if (op_is_root()) {
    // Gather and deserialise the MessagePacked trees from other ranks, then combine with +=
    std::size_t size;
    std::vector<std::uint8_t> msg;

    int comm_size;
    MPI_Comm_size(OP_MPI_WORLD, &comm_size);

    for (int rank = 1; rank < comm_size; ++rank) {
      MPI_Recv(&size, sizeof(std::size_t), MPI_BYTE, rank, 0, OP_MPI_WORLD, MPI_STATUS_IGNORE);
      msg.resize(size);

      MPI_Recv(msg.data(), size, MPI_BYTE, rank, 0, OP_MPI_WORLD, MPI_STATUS_IGNORE);
      json other_root_json = json::from_msgpack(msg);
      auto other_root = other_root_json.template get<op_timing2_node>();

      root += other_root;
    }
  } else {
    // root nodes -> JSON -> MessagePack -> send to MPI root
    json root_json = root;
    std::vector<std::uint8_t> msg = json::to_msgpack(root_json);
    std::size_t size = msg.size();

    MPI_Send(&size, sizeof(std::size_t), MPI_BYTE, 0, 0, OP_MPI_WORLD);
    MPI_Send(msg.data(), size, MPI_BYTE, 0, 0, OP_MPI_WORLD);
  }
#endif

  combined = true;
}

void op_timing2::output() {
  if (level == op_timing2_level::disabled) return;

  assert(finished);
  combine();

#ifdef OPMPI
  if (!op_is_root()) return;
#endif

  print_summary();

  char *json_filename = getenv("OP_TIMING2_JSON_OUTPUT");
  if (json_filename != NULL)
    output_json(json_filename);
}

void op_timing2::output_json(std::string_view filename) {
  if (level == op_timing2_level::disabled) return;

  assert(finished);
  combine();

#ifdef OPMPI
  if (!op_is_root()) return;
#endif

  auto output = std::ofstream{std::string(filename)};
  if (!output) {
    std::printf("Unable to open timing JSON output file: %.*s (%s)\n",
        static_cast<int>(filename.length()), filename.data(),
        std::strerror(errno));

    return;
  }

  std::printf("Writing timing JSON to: %.*s\n", static_cast<int>(filename.length()), filename.data());
  json root_json = root;
  output << root_json;
}

void op_timing2::print_summary() {
  // Output the non-kernel sections, and simultaneously gather a list of nodes which have
  // immediate kernel children
  std::vector<std::vector<std::string>> nodes_with_kernels;
  print_walk_non_kernel(root, {}, nodes_with_kernels, 0);
  std::printf("\n");

  // We need to calculate the longest name to reliably format the kernel table
  unsigned longest_name = 0;
  for (auto& path: nodes_with_kernels) {
    unsigned path_len = 0;
    for (auto& path_elem: path)
      path_len += path_elem.length() + 1;

    if (path_len > longest_name)
      longest_name = path_len;

    std::vector<std::string> scope = {path.begin() + 1, path.end()};
    auto& node = scope.size() == 0 ? root : root.get_child(scope);
    for (auto& child: node.children)
      if (child.name.length() + 4 > longest_name) longest_name = child.name.length() + 4;
  }

  // Summarise each node with immediate kernel children
  for (auto& path: nodes_with_kernels)
    print_kernel_summary(path, longest_name);
}

void op_timing2::print_walk_non_kernel(const op_timing2_node& node,
    const std::vector<std::string>& parent_path,
    std::vector<std::vector<std::string>>& nodes_with_kernels,
    unsigned indent) {
  bool has_kernel_child = false;
  for (auto& child: node.children)
    if (child.type == op_timing2_node_type::kernel) has_kernel_child = true;

  std::vector<std::string> current_path = parent_path;
  current_path.push_back(node.name);

  if (has_kernel_child)
    nodes_with_kernels.push_back(current_path);

  std::printf("%*s%s %s\n", indent, "", node.name.c_str(), to_string(node.clock).c_str());

  for (auto& child: node.children) {
    if (child.type == op_timing2_node_type::kernel) continue;
    print_walk_non_kernel(child, current_path, nodes_with_kernels, indent + 4);
  }
}

void op_timing2::print_kernel_summary(const std::vector<std::string>& path, unsigned longest_name) {
  // Print the header, starting with the node path
  int path_len = 0;
  for (size_t i = 0; i < path.size(); ++i) {
    if (i == path.size() - 1) {
      std::printf("%s", path[i].c_str());
      path_len += path[i].length();
    } else {
      std::printf("%s/", path[i].c_str());
      path_len += path[i].length() + 1;
    }
  }

  // And then the column headers for the table
  std::printf("%*s     num    total      avg      min      max", longest_name + 4 - path_len, "");
  if (level >= op_timing2_level::kernel_detailed) std::printf("    %%kern");
  std::printf("\n");

  // Fetch the node so we can print its children
  std::vector<std::string> scope = {path.begin() + 1, path.end()};
  auto& node = scope.size() == 0 ? root : root.get_child(scope);

  // Gather all kernel children
  std::vector<std::reference_wrapper<op_timing2_node>> kernel_nodes;
  for (auto& child: node.children) {
    if (child.type != op_timing2_node_type::kernel) continue;
    kernel_nodes.push_back(child);
  }

  // Sort by descending total time
  std::sort(kernel_nodes.begin(), kernel_nodes.end(),
      [](auto n1, auto n2) { return n1.get().clock.total > n2.get().clock.total; });

  const auto limit = 20; // Max number of children to print
  auto n = 0;

  auto to_s = [](auto t) { return std::chrono::duration<double>(t).count(); };
  for (auto& child: kernel_nodes) {
    if (n >= limit) {
      std::printf("    (%ld more)\n", kernel_nodes.size() - limit);
      break;
    }

    // Print each kernel child row, with kernel % if level = kernel_detailed
    auto kern_pct = std::string("");
    if (level >= op_timing2_level::kernel_detailed) {
      auto computation_node = child.get().get_child("Computation");
      auto kernel_node = computation_node.has_child("Kernel") ?
        computation_node.get_child("Kernel") : computation_node;

      double pct = 100.f * to_s(kernel_node.clock.total) / to_s(child.get().clock.total);

      char buf[10];
      snprintf(buf, 10, "    %5.1f", pct);
      kern_pct = std::string(buf);
    }

    std::printf("    %s%*s    %8ld  %s  %s  %s  %s%s\n", child.get().name.c_str(),
      (int) (longest_name - (child.get().name.length() + 4)), "",
      child.get().clock.n,
      format_duration(child.get().clock.total).c_str(),
      format_duration(child.get().clock.average()).c_str(),
      format_duration(child.get().clock.min).c_str(),
      format_duration(child.get().clock.max).c_str(),
      kern_pct.c_str()
    );

    ++n;
  }

  std::printf("\n");

  if (level < op_timing2_level::kernel_detailed) return;

  // Print the full tree for the top detailed_limit kernels
  const auto detailed_limit = 4;
  n = 0;

  for (auto& child: kernel_nodes) {
    if (n >= detailed_limit) break;

    child.get().output(4);
    std::printf("\n");

    ++n;
  }
}

extern "C" {

void op_timing2_start(const char* name) { op_timing2::instance().start(name); }

void op_timing2_enter(const char* name) { op_timing2::instance().enter(name); }
void op_timing2_enter_kernel(const char* name, const char* target, const char* variant) {
  op_timing2::instance().enter_kernel(name, target, variant);
}

void op_timing2_next(const char* name) { op_timing2::instance().next(name); }

void op_timing2_exit() { op_timing2::instance().exit(); }
void op_timing2_finish() { op_timing2::instance().finish(); }

void op_timing2_output() { op_timing2::instance().output(); }
void op_timing2_output_json(const char* filename) { op_timing2::instance().output_json(filename); }

}
