#include "add_cuda.h"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <sstream>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        throw std::runtime_error(cudaGetErrorString(status));                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        throw std::runtime_error(cusparseGetErrorString(status));              \
    }                                                                          \
}

template<typename value_t, typename size_t>
__global__ void add_kernel(value_t *__restrict__ r, const value_t *__restrict__ a, const value_t *__restrict__ b, const size_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = a[i] + b[i];
    }
}

float *gpu_add_f32(const float *x, const float *y, const uint64_t n) {
    float *result;
    CHECK_CUDA( cudaMalloc(&result, n * sizeof(float)) );

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    add_kernel<float, uint64_t><<<blocksPerGrid, threadsPerBlock>>>(result, x, y, n);

    CHECK_CUDA( cudaGetLastError() );

    return result;
}
