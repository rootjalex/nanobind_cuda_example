#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

// #include "cuda_utils.h"

inline void cudaFreeWrapper(void* ptr) noexcept {
    // cudaError_t error = cudaFree(ptr);
    // if (error != cudaSuccess) {
    //     throw std::runtime_error(cudaGetErrorString(error));
    // }
    // cudaFree(ptr);
}

namespace nb = nanobind;

template<typename T>
using GPUVector = nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::cuda>;

template<typename T>
using CPUVector = nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

template<typename T>
GPUVector<T> make_gpu_vector(T* ptr, const size_t n) {
    nb::capsule owner(ptr, cudaFreeWrapper);

    const size_t shape[1] = {n};

    return GPUVector<T>(ptr, /* ndim = */ 1, shape, owner, /* strides */ nullptr, nb::dtype<T>(), /* explicitly set device type */ nb::device::cuda::value);
}
