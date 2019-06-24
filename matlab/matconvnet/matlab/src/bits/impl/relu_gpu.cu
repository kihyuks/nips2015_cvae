// @file relu_gpu.cu
// @brief Relu non-linearity implementation (GPU)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#include "relu.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ------------------------------------------------------------ */
/*                                                   Relu (GPU) */
/* ------------------------------------------------------------ */

template<typename T> __global__ void
relu_gpu_kernel(
  T* output,
  const T* data,
  const int nthreads,
  const int width,
  const int height)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  T zero = (T)0;
  if (index < nthreads) {
    //int x = index;
    //int y = x / width;
    //int z = y / height;
    //x %= width;
    //y %= height;

    //output += z * (width*height);

    if (data[index] > zero)
      output[index] = data[index];
    else
      output[index] = zero;
  }

}

template<> vl::Error
vl::impl::relu_forward<vl::GPU, float>(float* output, float const* data, 
                                       size_t height, size_t width, size_t depth)
{
  int nthreads = width * height * depth;

  relu_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (output, data,
   nthreads, height, width);

  cudaError_t status = cudaPeekAtLastError();
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ------------------------------------------------------------ */
/*                                           ReluBackward (GPU) */
/* ------------------------------------------------------------ */

template <typename T> __global__ void
relu_backward_gpu_kernel(
  T* derData,
  const T* data,
  const T* derOutput,
  const int nthreads,
  const int width,
  const int height)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  T zero = (T)0;
  if (index < nthreads) {
    
    //int x = index;
    //int y = x / width;
    //int z = y / height;
    //x %= width;
    //y %= height;

    if (data[index] > zero)
      derData[index] = derOutput[index];
    else
      derData[index] = zero;
  }
}

template<> vl::Error
vl::impl::relu_backward<vl::GPU, float>(float* derData, float const* data, float const* derOutput,
                                        size_t height, size_t width, size_t depth)
{
  int nthreads = width * height * depth;
  
  relu_backward_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (derData, data, derOutput,
   nthreads, height, width);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}
