#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <chrono>
#include <iostream>
#include <random>

#include "Metal.hpp"

const int N = 256;

class Executor {
  public:
    Executor()
    {
        device_ = MTL::CreateSystemDefaultDevice();

        auto default_library = device_->newDefaultLibrary();

        if (!default_library) {
            std::cerr << "Failed to load default library.";
            std::exit(-1);
        }

        auto function_name = NS::String::string("multiply_matrix_and_vector",
                                                NS::ASCIIStringEncoding);
        auto function = default_library->newFunction(function_name);

        if (!function) {
            std::cerr << "Failed to find the adder function.";
            std::exit(-1);
        }

        NS::Error* error;
        pipeline_state_ = device_->newComputePipelineState(function, &error);
        command_queue_ = device_->newCommandQueue();

        matrix_buffer_ = device_->newBuffer(N * N * sizeof(float),
                                            MTL::ResourceStorageModeShared);
        vector_buffer_ = device_->newBuffer(N * sizeof(float),
                                            MTL::ResourceStorageModeShared);
        result_buffer_ = device_->newBuffer(N * sizeof(float),
                                            MTL::ResourceStorageModeShared);
    }

    ~Executor()
    {
        /*
        delete device_;
        delete pipeline_state_;
        delete command_queue_;
        delete matrix_buffer_;
        delete vector_buffer_;
        delete result_buffer_;
        */
    }

    auto Compute() -> void
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        MTL::CommandBuffer* buffer = command_queue_->commandBuffer();
        MTL::ComputeCommandEncoder* encoder = buffer->computeCommandEncoder();

        encoder->setComputePipelineState(pipeline_state_);
        encoder->setBuffer(matrix_buffer_, 0, 0);
        encoder->setBuffer(vector_buffer_, 0, 1);
        encoder->setBuffer(result_buffer_, 0, 2);

        MTL::Size grid_size = MTL::Size(N, 1, 1);
        MTL::Size thread_group_size = MTL::Size(N, 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);

        encoder->endEncoding();

        buffer->commit();
        auto t1 = std::chrono::high_resolution_clock::now();

        // buffer->waitUntilCompleted();
        int counter_error = 0;
        int counter_committed = 0;
        int counter_scheduled = 0;
        int counter_enqueued = 0;
        int counter_not_enqueued = 0;
        int counter_completed = 0;
        int counter = 0;

        while (buffer->status() != MTL::CommandBufferStatusCompleted) {
            ++counter;
            switch (buffer->status()) {
            case MTL::CommandBufferStatusError:
                ++counter_error;
                break;
            case MTL::CommandBufferStatusCommitted:
                ++counter_committed;
                break;
            case MTL::CommandBufferStatusScheduled:
                ++counter_scheduled;
                break;
            case MTL::CommandBufferStatusNotEnqueued:
                ++counter_not_enqueued;
                break;
            case MTL::CommandBufferStatusEnqueued:
                ++counter_enqueued;
                break;
            case MTL::CommandBufferStatusCompleted:
                ++counter_completed;
                break;
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        auto duration1 =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        auto duration2 =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Error counter: " << counter_error << std::endl;
        std::cout << "Commi counter: " << counter_committed << std::endl;
        std::cout << "Sched counter: " << counter_scheduled << std::endl;
        std::cout << "Enque counter: " << counter_enqueued << std::endl;
        std::cout << "NotEn counter: " << counter_not_enqueued << std::endl;
        std::cout << "Compl counter: " << counter_completed << std::endl;
        std::cout << "Total counter: " << counter << std::endl;
        std::cout << std::endl;
        std::cout << "Preparation: " << duration1.count() << "us" << std::endl;
        std::cout << "Computation: " << duration2.count() << "us" << std::endl;
    }

    auto GenerateMatrix(std::default_random_engine& rng) -> void
    {
        std::normal_distribution<float> normal(0, 1);
        float* data = static_cast<float*>(matrix_buffer_->contents());
        for (int i = 0; i < N; ++i) {
            data[i] = normal(rng);
        }
    }

    auto GenerateVector(std::default_random_engine& rng) -> void
    {
        std::normal_distribution<float> normal(0, 1);
        float* data = static_cast<float*>(matrix_buffer_->contents());
        for (int i = 0; i < N; ++i) {
            data[i] = normal(rng);
        }
    }

  private:
    MTL::Device* device_;
    MTL::ComputePipelineState* pipeline_state_;
    MTL::CommandQueue* command_queue_;
    MTL::ComputeCommandEncoder* command_encoder_;
    MTL::Buffer* matrix_buffer_;
    MTL::Buffer* vector_buffer_;
    MTL::Buffer* result_buffer_;
};

int main(int argc, const char* argv[])
{
    std::cout << "Let's multiply!" << std::endl;
    std::default_random_engine rng(0);

    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();

    Executor executor;
    executor.GenerateMatrix(rng);
    executor.GenerateVector(rng);
    executor.Compute();
    executor.GenerateMatrix(rng);
    executor.GenerateVector(rng);
    executor.Compute();
    executor.GenerateMatrix(rng);
    executor.GenerateVector(rng);
    executor.Compute();
    executor.GenerateMatrix(rng);
    executor.GenerateVector(rng);
    executor.Compute();
    executor.GenerateMatrix(rng);
    executor.GenerateVector(rng);
    executor.Compute();
    std::cout << "Execution finished." << std::endl;

    p_pool->release();

    return 0;
}
