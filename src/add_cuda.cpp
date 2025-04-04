#include "add_cuda.hpp"

GPUVector<float> nb_gpu_add_f32(const GPUVector<float> a, const GPUVector<float> b) {
    const uint64_t n = a.shape(0);
    float *result = gpu_add_f32(a.data(), b.data(), n);
    return make_gpu_vector(result, n);
}
