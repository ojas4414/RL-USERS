#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// DFS-based topological sort — runs ONCE at startup to validate
// the funnel DAG has no cycles and produces the canonical order
std::vector<std::string> topological_sort(
    const std::unordered_map<std::string, std::vector<std::string>>& graph
) {
    std::unordered_map<std::string, bool> visited;
    std::unordered_map<std::string, bool> temp_mark;
    std::vector<std::string> order;

    std::function<void(const std::string&)> dfs = [&](const std::string& node) {
        if (temp_mark.count(node) && temp_mark[node])
            throw std::runtime_error("Cycle detected at: " + node);
        if (visited.count(node) && visited[node]) return;

        temp_mark[node] = true;
        if (graph.count(node)) {
            for (const auto& neighbor : graph.at(node)) {
                dfs(neighbor);
            }
        }
        temp_mark[node] = false;
        visited[node]   = true;
        order.push_back(node);
    };

    for (const auto& [node, _] : graph) {
        if (!visited.count(node) || !visited[node]) {
            dfs(node);
        }
    }

    std::reverse(order.begin(), order.end());
    return order;
}

bool is_action_allowed(
    const std::vector<std::string>& agent_history,
    const std::string& action,
    const std::unordered_map<std::string, std::vector<std::string>>& prereqs
) {
    if (prereqs.count(action) == 0) return true;

    for (const auto& required : prereqs.at(action)) {
        bool found = false;
        for (const auto& h : agent_history) {
            if (h == required) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

PYBIND11_MODULE(funnel_core, m) {
    m.def("topological_sort",  &topological_sort);
    m.def("is_action_allowed", &is_action_allowed);
}