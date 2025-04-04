#include "cuda_utils.h"

#include <cuda_runtime.h>

// Wrapper function for cudaFree
// __host__ void cudaFreeWrapper(void* ptr) noexcept {
//     // cudaError_t error = cudaFree(ptr);
//     // if (error != cudaSuccess) {
//     //     throw std::runtime_error(cudaGetErrorString(error));
//     // }
//     cudaFree(ptr);
// }
