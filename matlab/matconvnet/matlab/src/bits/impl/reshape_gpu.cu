// @file reshape_gpu.cu
// @brief Reshape implementation (GPU)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#include "reshape.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ------------------------------------------------------------ */
/*                                                Reshape (GPU) */
/* ------------------------------------------------------------ */

template<typename T> __global__ void
reshape_gpu_kernel(
  T* output,
  const T* data,
  const int nthreads
)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    output[index] = data[index];//TODO: check gradient
  }
}

template<> vl::Error
vl::impl::reshape_forward<vl::GPU, float>(float* output, float const* data, 
                                       size_t ndim, size_t ndata)
{
  int nthreads = ndim * ndata;

  reshape_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (output, data, nthreads);

  cudaError_t status = cudaPeekAtLastError();
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ------------------------------------------------------------ */
/*                                        ReshapeBackward (GPU) */
/* ------------------------------------------------------------ */

template <typename T> __global__ void
reshape_backward_gpu_kernel(
  T* derData,
  const T* derOutput,
  const int nthreads)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    derData[index] = derOutput[index];
  }
}

template<> vl::Error
vl::impl::reshape_backward<vl::GPU, float>(float* derData, float const* derOutput,
                                        size_t ndim, size_t ndata)
{
  int nthreads = ndim * ndata;
  
  reshape_backward_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (derData, derOutput, nthreads);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
