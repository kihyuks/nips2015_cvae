// @file nnunpooling_cudnn.cu
// @brief Unpooling block CuDNN-based implementation.
// @author Xinchen
// Modified by Xinchen Yan, Feb. 19, 2015

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnunpooling_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnunpooling_cudnn.hpp"
#include "../datacu.hpp"
#include <assert.h>

using namespace vl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
goto done ; \
} }

/* nnunpooling_forward_cudnn */
template<> vl::Error
vl::impl::nnunpooling_forward_cudnn<float>(Context& context,
                                           Tensor output,
                                           Tensor data,
                                           int strideY, int strideX)

{
  assert(output) ;
  assert(data) ;
  
  cudnnTensorDescriptor_t outputDesc, dataDesc ;
  cudnn
//TODO

}



