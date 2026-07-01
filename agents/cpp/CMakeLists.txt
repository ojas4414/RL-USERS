cmake_minimum_required(VERSION 3.15)
project(rl_users_cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

pybind11_add_module(scheduler_core        scheduler_core.cpp)
pybind11_add_module(social_graph_core     social_graph_core.cpp)
pybind11_add_module(funnel_core           funnel_core.cpp)

pybind11_add_module(agent_inference_engine agent_inference_engine.cpp)
target_link_libraries(agent_inference_engine PRIVATE ${TORCH_LIBRARIES})