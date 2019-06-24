// @file upsampling_gpu.cu
// @brief Upsampling block implementation (GPU)
// @author Xinchen Yan
// Created by Xinchen, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)
//#include <iostream>
#include "upsampling.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ---------------------------------------------------------------- */
/*                                                 Upsampling (GPU) */
/* ---------------------------------------------------------------- */

//using namespace std;

template<typename T> __global__ void 
upsampling_gpu_kernel
(T* upsampled,
 const T* data,
 const bool sparse,
 const int nthreads,
 const int width,
 const int height,
 const int strideX,
 const int strideY)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int upsampledWidth = width * strideX;
    int upsampledHeight = height * strideY;
    int x = index ;
    int y = x / width ;
    int z = y / height ;
    x %= width ;
    y %= height ;

    upsampled += z * (upsampledWidth*upsampledHeight) ;

    int x1 = x * strideX;
    int y1 = y * strideY;
    int x2 = (x + 1) * strideX;
    int y2 = (y + 1) * strideY;
    for (int v = y1 ; v < y2 ; ++v) {
      for (int u = x1 ; u < x2 ; ++u) {
        if (sparse)
          upsampled[v * upsampledWidth + u] = (T)0;
        else
          upsampled[v * upsampledWidth + u] = data[index];
      }
    }
    upsampled[y1 * upsampledWidth + x1] = data[index] ;
  }
}


template<> vl::Error
vl::impl::upsampling_forward<vl::GPU, float>(float* upsampled,
                                      float const* data,
                                      bool sparse,
                                      size_t height, size_t width, size_t depth,
                                      size_t strideY, size_t strideX)
{
  int nthreads = width * height * depth;

  //std::cerr << sparse << std::endl;
  upsampling_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (upsampled, data,
   sparse,
   nthreads, height, width,
   strideY, strideX);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ---------------------------------------------------------------- */
/*                                         UpsamplingBackward (GPU) */
/* ---------------------------------------------------------------- */

//#ifdef VLNN_CAFFELIKE_BPPOLL
template <typename T> __global__ void 
upsampling_backward_gpu_kernel(
    T* derData,
    const T* derUpsampled,
    const bool sparse,
    const int nthreads,
    const int width,
    const int height,
    const int strideX,
    const int strideY)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    
    int upsampledWidth = width * strideX;
    int upsampledHeight = height * strideY;

    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;
    
    derUpsampled += z * (upsampledWidth * upsampledHeight);
    int x1 = x_data * strideX;
    int y1 = y_data * strideY;
    int x2 = (x_data + 1) * strideX;
    int y2 = (y_data + 1) * strideY;
    if (sparse) {
      derData[index] += derUpsampled[y1 * upsampledWidth + x1];
    } else {
      for (int v = y1 ; v < y2 ; ++v) {
        for (int u = x1 ; u < x2 ; ++u) { 
          derData[index] += derUpsampled[v * upsampledWidth + u];
        }
      }
      derData[index] /= (T)(strideX * strideY);    
    } 
  }
}
//#endif

template<> vl::Error
vl::impl::upsampling_backward<vl::GPU, float>(float* derData,
                                              float const* derUpsampled,
                                              bool sparse,
                                              size_t height, size_t width, size_t depth,
                                              size_t strideY, size_t strideX)
{
  int nthreads = width * height * depth;
  
  upsampling_backward_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (derData, derUpsampled,
   sparse,
   nthreads,
   height, width,
   strideY, strideX);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

