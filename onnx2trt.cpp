#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "ioHelper.h"
#include <fstream>  
#include <iostream>  
#include <string>   
#include <memory>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;
const string onnx_path = "./bert-base-uncased/model-sim.onnx";
const string trt_path = "./bert-base-uncased/model.plan";


ICudaEngine* createCudaEngine(string const& onnx_path, int batch_size) {
    unique_ptr<IBuilder, Destroy<IBuilder>> builder(createInferBuilder(gLogger));
    builder->setMaxBatchSize(batch_size);

    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{builder->createNetworkV2(explicit_batch)};
    unique_ptr<IParser, Destroy<IParser>> parser{createParser(*network, gLogger)};
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }

    config->setMaxWorkspaceSize(1 << 30);

    /* 动态shape */
    auto profile = builder->createOptimizationProfile();
    u_int8_t num_inputs = static_cast<u_int8_t>(network->getNbInputs());
    for (u_int8_t i = 0; i < num_inputs; i++) {
        auto input_tensor = network->getInput(i);
        profile->setDimensions(input_tensor->getName(), OptProfileSelector::kMIN, Dims2(1, 6));
        profile->setDimensions(input_tensor->getName(), OptProfileSelector::kOPT, Dims2(1, 64));
        profile->setDimensions(input_tensor->getName(), OptProfileSelector::kMAX, Dims2(1, 256));
    }
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}


void saveEngine(const unique_ptr<ICudaEngine, Destroy<ICudaEngine>> &engine, string const& trt_path) {
    unique_ptr<IHostMemory, nvinfer1::Destroy<IHostMemory>> trtModelStream{engine->serialize()};

    std::ofstream out(trt_path.c_str(), std::ios::binary);
    if (!out.is_open()) {
        std::cout << "Can not open output file!" <<std:: endl;
    }
    out.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    out.close();
}


int main(){
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    engine.reset(createCudaEngine(onnx_path, 1));
    saveEngine(engine, trt_path);
    return 0;
}
