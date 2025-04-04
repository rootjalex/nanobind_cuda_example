#pragma once

#include "add_cuda.h"
#include "nb_utils.hpp"

GPUVector<float> nb_gpu_add_f32(const GPUVector<float> a, const GPUVector<float> b);