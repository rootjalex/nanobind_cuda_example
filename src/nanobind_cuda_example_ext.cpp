#include <nanobind/nanobind.h>

#include "add_cuda.hpp"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(nanobind_cuda_example_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind";
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("gpu_add_f32", &nb_gpu_add_f32);
}
