cmake_minimum_required(VERSION 3.15...3.26)

message("Building for Redwood")
set(CMAKE_CUDA_ARCHITECTURES 89)
project(nanobind_cuda_example LANGUAGES CUDA CXX)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject] torch
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
# message(CMAKE_CUDA_ARCHITECTURES)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--use_fast_math")


set(nanobind_cuda_example_CUDA_INCLUDES
    # Include folders go here
    ${CUDAToolkit_INCLUDE_DIRS}
)

set(nanobind_cuda_example_CUDA_SRC
  src/add_cuda.cu
  src/cuda_utils.cu
)

set(nanobind_cuda_example_CUDA_DEPENDENCIES
    # CUDA::cusparse
)

add_library(nanobind_cuda_example_CUDA_KERNELS STATIC ${nanobind_cuda_example_CUDA_SRC})
set_target_properties(nanobind_cuda_example_CUDA_KERNELS
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(nanobind_cuda_example_CUDA_KERNELS PRIVATE ${nanobind_cuda_example_CUDA_DEPENDENCIES} ${Python_LIBRARIES})
target_include_directories(nanobind_cuda_example_CUDA_KERNELS PRIVATE ${nanobind_cuda_example_CUDA_INCLUDES})

# All source files
set(nanobind_cuda_example_CPP_SRC
    src/nanobind_cuda_example_ext.cpp
    src/add_cuda.cpp
)

# Add the CUDA flags (Redwood GPU arch)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_89 -dc")

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  nanobind_cuda_example_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  # src/nanobind_cuda_example_ext.cpp
  ${nanobind_cuda_example_CPP_SRC}
  ${nanobind_cuda_example_CUDA_SRC}
)

# Link the CUDA libraries (optional, only needed if you require specific CUDA libraries)
# target_link_libraries(nanobind_cuda_example_ext PRIVATE CUDA::cudart)
target_link_libraries(nanobind_cuda_example_ext PRIVATE ${nanobind_cuda_example_CUDA_DEPENDENCIES})

# Install directive for scikit-build-core
install(TARGETS nanobind_cuda_example_ext LIBRARY DESTINATION nanobind_cuda_example)
