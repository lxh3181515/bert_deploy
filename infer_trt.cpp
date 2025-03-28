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
const string trt_path = "./bert-base-uncased/model_sim.plan";
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


// 将一维 vector 转换为二维 vector
vector<vector<float>> reshape(const vector<float>& vec, size_t rows, size_t cols) {
    // 检查输入是否合法
    if (vec.size() != rows * cols) {
        throw std::invalid_argument("Cannot reshape: dimensions do not match the vector size.");
    }

    // 创建二维 vector
    vector<vector<float>> result(rows, vector<float>(cols));

    // 填充二维 vector
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = vec[i * cols + j];
        }
    }

    return result;
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
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

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

    cout << "Output:\n";
    std::vector<std::vector<float>> output = reshape(output_tensor, 16, 30522);
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < 3; j++) {
            cout << output[i][j] << " ";
        }
        cout << "... ";
        for (int j = output[0].size() - 3; j < output[0].size(); j++) {
            cout << output[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Output sum:";
    float sum = 0.0;
    for (int i = 0; i < output_tensor.size(); i++) {
        sum += output_tensor[i];
    }
    cout << sum << endl;
}
