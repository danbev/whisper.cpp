# Cross-compile whisper.cpp for Windows ARM64 on an x64 host. Includes the
# Clang toolchain ggml requires on ARM (MSVC is explicitly rejected there -
# see ggml/src/ggml-cpu/CMakeLists.txt) and layers CUDA support on top: nvcc
# compiles CUDA sources using an ARM64-targeting MSVC host compiler,
# independent of the Clang toolchain used for the rest of the project.
include( ${CMAKE_CURRENT_LIST_DIR}/arm64-windows-llvm.cmake )

if ( DEFINED CUDAToolkit_ROOT )
    file( TO_CMAKE_PATH "${CUDAToolkit_ROOT}" CUDA_ROOT )
elseif ( DEFINED ENV{CUDA_PATH} )
    file( TO_CMAKE_PATH "$ENV{CUDA_PATH}" CUDA_ROOT )
else()
    message( FATAL_ERROR "Set CUDAToolkit_ROOT or CUDA_PATH to a Windows CUDA Toolkit with ARM64 target libraries" )
endif()

if ( DEFINED ENV{VCToolsInstallDir} )
    file( TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" MSVC_TOOLS_ROOT )
    set( CMAKE_CUDA_HOST_COMPILER "${MSVC_TOOLS_ROOT}/bin/Hostx64/arm64/cl.exe" CACHE FILEPATH "" )
endif()

set( CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" )
set( CMAKE_CUDA_FLAGS_INIT "-target-dir=arm64" )

# FindCUDAToolkit selects lib/x64 from the host architecture on Windows.
set( CUDA_CUDART              "${CUDA_ROOT}/lib/arm64/cudart.lib"   CACHE FILEPATH "" )
set( CUDA_cudart_LIBRARY      "${CUDA_ROOT}/lib/arm64/cudart.lib"   CACHE FILEPATH "" )
set( CUDA_cublas_LIBRARY      "${CUDA_ROOT}/lib/arm64/cublas.lib"   CACHE FILEPATH "" )
set( CUDA_cublasLt_LIBRARY    "${CUDA_ROOT}/lib/arm64/cublasLt.lib" CACHE FILEPATH "" )
set( CUDA_cuda_driver_LIBRARY "${CUDA_ROOT}/lib/arm64/cuda.lib"     CACHE FILEPATH "" )
