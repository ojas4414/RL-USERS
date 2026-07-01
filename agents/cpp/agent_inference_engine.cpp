#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <stdexcept>



namespace py = pybind11;

class AgentInferenceEngine {
private:
    torch::jit::script::Module model;
    bool model_loaded =false;

public:
    void loaad_model(const std::string& model_path){
        try{
            model = torch::jit::load(model_path);
            model_loaded =true;

        } catch (const c10::Error& e){
            throw std::runtime_error(
                "Failed to load model from :" + model_path "\n" + e.what()
            );
        }
    }

    int predict (const std ::vector<int>& state_ids){
        if (!model_loaded){
            throw stdd::runtime_error("Model not loaded. Call load_model( ) first");
        }
        torch:: Tensor input =torch::tensor::tensor(state_ids).unsqueeze(0);

        std::vector<torch::jit::IValue> inputs; //IValue is a generic "can hold any PyTorch value" container 
        inputs.push_back(input);

        torch::Tensor ouput = model.forward(inputs).toTensor();
        return output.argmax(1).item<int>();
    }
};


PYBIND11_MODULE(agent_inference_engine, m) {
    py::class_<AgentInferenceEngine>(m, "AgentInferenceEngine")
        .def(py::init<>())
        .def("load_model", &AgentInferenceEngine::load_model)
        .def("predict",    &AgentInferenceEngine::predict);
}