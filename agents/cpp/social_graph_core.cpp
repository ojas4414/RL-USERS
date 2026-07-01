#include <unordered_map>
#include <vector>
#include <queue>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::unordered_map<int,double> bfs_influence(
    const std::unordered_map<int, std::vector<int>>& graph,
    int source_agent,
    int max_depth  =3
){
    std::unordered_map<int, int> visited;
    std::queue<std::pair<int, int>> queue;

    visited[source_agent]=0;
    queue.push({source_agent,0});

    while (!queue.empty()){
        auto [current, depth]= queue.front();
        queue.pop();

        if (depth>= max_depth) continue;

        if (graph.count(current)==0) continue;

        for (int neighbor : graph.at(current)) {
            if (visited.count(neighbor) == 0) {
                visited[neighbor] = depth + 1;
                queue.push({neighbor, depth + 1});
            }
        }
    }

    std::unordered_map<int, double> signals;
    for (auto& [agent_id, depth] : visited) {
        if (agent_id == source_agent) continue;
        signals[agent_id] = 1.0 / std::pow(2.0, depth);
    }
    return signals;
}

PYBIND11_MODULE(social_graph_core, m) {
    m.def("bfs_influence", &bfs_influence,
          py::arg("graph"),
          py::arg("source_agent"),
          py::arg("max_depth") = 3);


    }
