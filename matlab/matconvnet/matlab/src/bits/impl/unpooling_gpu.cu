// @file unpooling_gpu.cu
// @brief Unpooling block implementation (GPU)
// @author Xinchen Yan
// Modified by Xinchen, Feb. 19, 2015 (in compatible with matconvnet 1.0.9)

#include "unpooling.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ---------------------------------------------------------------- */
/*                                                Unpooling (GPU) */
/* ---------------------------------------------------------------- */
using namespace std;
template<typename T> __global__ void 
unpooling_gpu_kernel
(T* unpooled,
 const T* data,
 const int nthreads,
 const int width,
 const int height,
 const int strideX,
 const int strideY)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int unpooledWidth = width * strideX;
    int unpooledHeight = height * strideY;
    int x = index ;
    int y = x / width ;
    int z = y / height ;
    x %= width ;
    y %= height ;

    unpooled += z * (unpooledWidth*unpooledHeight) ;

    int x1 = x * strideX;
    int y1 = y * strideY;
    int x2 = (x + 1) * strideX;
    int y2 = (y + 1) * strideY;
    for (int v = y1 ; v < y2 ; ++v) {
      for (int u = x1 ; u < x2 ; ++u) {
        unpooled[v * unpooledWidth + u] = (T)0;
      }
    }
    unpooled[y1 * unpooledWidth + x1] = data[index] ;
  }
}


template<> vl::Error
vl::impl::unpooling_forward<vl::GPU, float>(float* unpooled,
                                     float const* data,
                                     size_t height, size_t width, size_t depth,
                                     size_t strideY, size_t strideX)
{
  int nthreads = width * height * depth;

  unpooling_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (unpooled, data,
   nthreads, height, width,
   strideY, strideX);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

/* ---------------------------------------------------------------- */
/*                                         UnpoolingBackward (GPU) */
/* ---------------------------------------------------------------- */

//#ifdef VLNN_CAFFELIKE_BPPOLL
template <typename T> __global__ void 
unpooling_backward_gpu_kernel(
    T* derData,
    const T* derUnpooled,
    const int nthreads,
    const int width,
    const int height,
    const int strideX,
    const int strideY)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    
    int unpooledWidth = width * strideX;
    int unpooledHeight = height * strideY;

    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;
    
    derUnpooled += z * (unpooledWidth * unpooledHeight);
    int x1 = x_data * strideX;
    int y1 = y_data * strideY;
    
    derData[index] += derUnpooled[y1 * unpooledWidth + x1];
  }
}
//#endif

template<> vl::Error
vl::impl::unpooling_backward<vl::GPU, float>(float* derData,
                                          float const* derUnpooled,
                                          size_t height, size_t width, size_t depth,
                                          size_t strideY, size_t strideX)
{
  int nthreads = width * height * depth;
  
  unpooling_backward_gpu_kernel<float>
  <<< divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
  (derData, derUnpooled,
   nthreads,
   height, width,
   strideY, strideX);

  cudaError_t status = cudaPeekAtLastError() ;
  return (status == cudaSuccess) ? vl::vlSuccess : vl::vlErrorCuda ;
}

