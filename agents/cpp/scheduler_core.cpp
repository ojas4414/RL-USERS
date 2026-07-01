#include <vector>
#include <queue>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include<string>


struct Event{
    double timestamp;
    long counter;
    int agent_id;
    std::string action;

};

struct CompareEvent{
    bool operator()(const Event& a, const Event& b){ 
        if (a.timestamp != b.timestamp){
            return a.timestamp > b.timestamp;
        }
        return a.counter > b.counter;

    }
};


class Scheduler{
private:
    std::priority_queue<Event,std::vector<Event>,CompareEvent>heap;
    long counter=0;

public:
    void push(double timestamp,int agent_id,std::string action ){
        Event e;
        e.timestamp=timestamp;
        e.counter=counter++;
        e.agent_id=agent_id;
        e.action=action;
        heap.push(e);
    }

    bool empty(){
        return heap.empty();
    }

    std::tuple<double, int , std:: string> pop(){
        if (heap.empty()){
            return {-1.0,-1,""};
        }
        Event e = heap.top();
        heap.pop();
        return {e.timestamp,e.agent_id,e.action};
    }
};


namespace py =pybind11;

PYBIND11_MODULE(scheduler_core, m) {
    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init<>())
        .def("push", &Scheduler::push)
        .def("pop",  &Scheduler::pop)
        .def("empty", &Scheduler::empty);
}

