// @file nnunpooling_cudnn.hpp
// @brief Unpooling block CuDNN-based implementation.
// @author Xinchen Yan
// Modified by Xinchen, Feb. 19, 2015 (in compatible with matconvnet 1.0.9)

#ifndef __vl__nnunpooling_cudnn__
#define __vl__nnunpooling_cudnn__

#include "../nnunpolling.hpp"
#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<typename type> vl::Error
  nnunpooling_forward_cudnn(Context& context,
                            Tensor output,
                            Tensor data,
                            int strideY, int strideX);

  template<typename type> vl::Error
  nnunpooling_backward_cudnn(Context& context,
                             Tensor derData,
                             Tensor derOutput,
                             int strideY, int strideX);

  /* specialisations */
  template<> vl::Error
  nnunpooling_forward_cudnn<float>(Context& context,
                                   Tensor output,
                                   Tensor data,
                                   int strideY, int strideX);

  template<> vl::Error
  nnunpooling_backward_cudnn(Context& context,
                             Tensor derData,
                             Tensor derOutput,
                             int strideY, int strideX);

} }

#endif /* defined(__vl__nnunpooling_cudnn__) */


