cmake_minimum_required (VERSION 2.8)
project (cuda-convnet)

find_package (CUDA REQUIRED)
find_package (BLAS REQUIRED)
find_package (PythonLibs REQUIRED)
find_package (PythonInterp REQUIRED)

find_path(BLAS_INCLUDE_DIRS cblas.h)

find_path(CUDA_COMMON_INCLUDE_DIRS
  helper_cuda.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
    "/Developer/NVIDIA/CUDA-6.0"
    "/usr/local/cuda"
  PATH_SUFFIXES "samples/common/inc"
  DOC "Location of helper_cuda.h"
  NO_DEFAULT_PATH
)

if(APPLE)
  LIST(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/clang)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
endif(APPLE)

#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_32,code=sm_32)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND CMAKE_CXX_FLAGS -fPIC)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
  "import numpy as n; print(n.get_include());"
  RESULT_VARIABLE _NUMPY_SEARCH_SUCCESS
  OUTPUT_VARIABLE _NUMPY_VALUES
  ERROR_VARIABLE _NUMPY_ERROR_VALUE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_NUMPY_SEARCH_SUCCESS MATCHES 0)
  string(REGEX REPLACE ";" "\\\\;" _NUMPY_VALUES ${_NUMPY_VALUES})
  string(REGEX REPLACE "\n" ";" _NUMPY_VALUES ${_NUMPY_VALUES})
  list(GET _NUMPY_VALUES 0 NUMPY_INCLUDE_DIRS)
  string(REGEX REPLACE "\\\\" "/" NUMPY_INCLUDE_DIRS ${NUMPY_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "NumPy import failure:\n${_NUMPY_ERROR_VALUE}")
endif()

include_directories (
  ${BLAS_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_COMMON_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}/numpy
  include/common
  include/nvmatrix
  include/cudaconv2
  include/
)

add_subdirectory (src/common)
add_subdirectory (src/nvmatrix)
add_subdirectory (src/cudaconv2)

CUDA_ADD_LIBRARY (convnet SHARED
  src/convnet.cu
  src/cost.cu
  src/data.cu
  src/layer.cu
  src/layer_kernels.cu
  src/neuron.cu
  src/pyconvnet.cu
  src/util.cu
  src/weights.cu
  src/worker.cu
)

set_target_properties (convnet
  PROPERTIES
  PREFIX ""
  SUFFIX "_.so"
)

TARGET_LINK_LIBRARIES (convnet
  common
  nvmatrix
  cudaconv2
  ${PYTHON_LIBRARIES}
  ${BLAS_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

CUDA_BUILD_CLEAN_TARGET()
