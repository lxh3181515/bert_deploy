#include "NvInfer.h"
#include "ioHelper.h"
#include "cudaWrapper.h"
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <cassert>
#include <numeric>
#include <time.h>


using namespace std;
using namespace nvinfer1;
using namespace cudawrapper;


static Logger gLogger;
const string trt_path = "./bert-base-uncased/model.plan";
const int batch_size = 1;
const int seq_len = 16;


void launchInference(IExecutionContext* const& context, 
                     cudaStream_t stream, 
                     vector<vector<int>> const& input_tensor, 
                     vector<float>& output_tensor, 
                     void** bindings)
{
    for (int i = 0; i < input_tensor.size(); i++) {
        cudaMemcpyAsync(bindings[i], input_tensor[i].data(), input_tensor[i].size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    }
    context->enqueueV2(bindings, stream, nullptr);
    cudaMemcpyAsync(output_tensor.data(), bindings[3], output_tensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
}


ICudaEngine* loadEngine(const string &trt_path) {
	std::ifstream file(trt_path, std::ios::binary);
	char* trt_stream = NULL;
	int size = 0;
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trt_stream = new char[size];
		assert(trt_stream);
		file.read(trt_stream, size);
		file.close();
	}
    else {
        return nullptr;
    }

    unique_ptr<IRuntime, Destroy<IRuntime>> runtime{createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    return runtime->deserializeCudaEngine(trt_stream, size);    
}


int main() {
    void* bindings[4]{0};
    vector<vector<int>> input_tensor(3);
    vector<float> output_tensor;
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr};
    CudaStream stream;

    engine.reset(loadEngine(trt_path));
    if (!engine) {
        return -1;
    }
    context.reset(engine->createExecutionContext());

    u_int8_t num_bindings = engine->getNbBindings();
    assert(engine->getNbBindings() == 4);
    for (u_int8_t i = 0; i < num_bindings; ++i)
    {
        if (engine->bindingIsInput(i)) {
            cudaMalloc(&bindings[i], batch_size * seq_len * sizeof(int));
            input_tensor.resize(batch_size * seq_len);
        }
        else {
            Dims dims{engine->getBindingDimensions(i)};
            size_t size = batch_size * seq_len * dims.d[2];
            cudaMalloc(&bindings[i], size * sizeof(float));
            output_tensor.resize(size);
        }
    }

    input_tensor = vector<vector<int>>{
        {101, 1996, 3007, 1997, 2605, 1010, 103, 1010, 3397, 1996, 1041, 13355, 2884, 3578, 1012, 102},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

    Dims2 inputDims{batch_size, seq_len};
    for (int i = 0; i < input_tensor.size(); i++) {
        context->setBindingDimensions(i, inputDims);
    }

    clock_t ts = clock();
    for (int i = 0; i < 100; i++) {
        // launchInference(context.get(), stream, input_tensor, output_tensor, bindings);
        for (int i = 0; i < input_tensor.size(); i++) {
            cudaMemcpyAsync(bindings[i], input_tensor[i].data(), input_tensor[i].size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        context->enqueueV2(bindings, stream, nullptr);
        cudaMemcpyAsync(output_tensor.data(), bindings[3], output_tensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    cout << "Average time cost: " << (clock() - ts) * 10.0 / CLOCKS_PER_SEC << "ms" << endl;

    cout << "Output tensor: ";
    for (int i = 0; i < 16; i++) {
        cout << output_tensor[i] << " ";
    }
    cout << endl;
}
